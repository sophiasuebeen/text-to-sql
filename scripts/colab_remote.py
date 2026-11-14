"""Remote implementation for colab_training.py; executed inside Colab."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RESULT_PREFIX = "TEXT_TO_SQL_RESULT="
PROJECT = Path("/content/text-to-sql")
PID_FILE = Path("/content/text-to-sql-full-e1.pid")


def training_pids() -> list[int]:
    """Find actual training commands instead of trusting a recycled PID."""
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if ((b"-m" in args and b"text_to_sql.model.train" in args)
                or any(arg.endswith(b"/train_with_checkpoint_policy.py") for arg in args)):
            found.append(int(proc.name))
    return sorted(found)


def status(run: Path) -> dict:
    state = run / "status.json"
    checkpoints = run / "resume"
    completed = sorted(
        int(p.name.split("-")[-1])
        for p in checkpoints.glob("checkpoint-*")
        if p.name.split("-")[-1].isdigit() and (p / "COMPLETE.json").is_file()
    )
    log = run / "training.log"
    # Bound the read even for a long-running training log.
    tail = ""
    if log.exists():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 16000))
            tail = "\n".join(stream.read().decode(errors="replace").splitlines()[-20:])
    return {
        "action": "status", "pids": training_pids(),
        "saved_status": json.loads(state.read_text()) if state.exists() else None,
        "checkpoint_markers": completed,
        "final_marker": (run / "final/COMPLETE.json").is_file(),
        "log_tail": tail,
    }


def supervise(process, run: Path, checkpoint: str) -> dict:
    """Keep the real training execution active in the notebook kernel."""
    invocation = run / "invocations" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{process.pid}"
    invocation.mkdir(parents=True, exist_ok=False)
    record = {"pid": process.pid, "checkpoint": checkpoint, "checkpoint_save_steps": 20,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    (invocation / "launch.json").write_text(json.dumps(record, indent=2))
    print("TEXT_TO_SQL_STARTED=" + json.dumps(record), flush=True)
    while process.poll() is None:
        snapshot = status(run)
        print("TEXT_TO_SQL_PROGRESS=" + json.dumps({
            "pid": process.pid, "saved_status": snapshot["saved_status"],
            "checkpoint_markers": snapshot["checkpoint_markers"],
            "log_tail": snapshot["log_tail"],
        }), flush=True)
        time.sleep(30)
    record.update(ended_utc=datetime.now(timezone.utc).isoformat(),
                  returncode=process.returncode, saved_status=status(run)["saved_status"])
    (invocation / "exit.json").write_text(json.dumps(record, indent=2))
    if process.returncode != 0:
        raise RuntimeError(f"Training exited with code {process.returncode}; inspect {invocation / 'exit.json'} and training.log")
    state = (record["saved_status"] or {}).get("state")
    if state not in ("paused", "complete"):
        raise RuntimeError(f"Training exited without a paused/complete state: {state}")
    return {"action": state, **record}


def resume(run: Path, hours: float) -> dict:
    import fcntl
    import shutil
    import tarfile

    # Hold this same-VM lock throughout setup and pass it to the training child.
    lock = open("/content/text-to-sql-full-e1.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return {"action": "already_running", "pids": training_pids()}
    existing = training_pids()
    if existing:
        return {"action": "already_running", "pids": existing}
    if not (run / "identity.json").is_file():
        raise RuntimeError("Saved identity missing; refusing to start a fresh run")
    if (run / "final/COMPLETE.json").exists():
        return {"action": "complete", "message": "Final export exists; training not restarted"}

    saved = json.loads((run / "identity.json").read_text())
    # Check the GPU before spending time on installation or model downloads.
    import torch
    assert torch.cuda.is_available(), "A GPU session is required"
    for key, value in {
        "torch": torch.__version__, "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
    }.items():
        if saved["dependencies"][key] != value:
            raise RuntimeError(f"Environment mismatch for {key}: {value}")

    PROJECT.mkdir(exist_ok=True)
    with tarfile.open("/content/text-to-sql-resume.tar.gz") as archive:
        archive.extractall(PROJECT, filter="data")
    shutil.copy2(run / "train_source.py", PROJECT / "src/text_to_sql/model/train.py")
    shutil.copy2(run / "train_config.yaml", PROJECT / "configs/train-full-t4-e1.yaml")
    deps = saved["dependencies"]
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-e", str(PROJECT),
        f"transformers=={deps['transformers']}", f"peft=={deps['peft']}",
        f"bitsandbytes=={deps['bitsandbytes']}", "accelerate==1.15.0",
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Use a fresh interpreter: pip may have replaced modules imported by the kernel.
    preflight = '''
import json, os, sys
from pathlib import Path
os.chdir('/content/text-to-sql')
sys.path.insert(0, '/content/text-to-sql/src')
from text_to_sql.model.train import load_config, run_identity, _dependency_versions
from text_to_sql.model.checkpoints import latest_complete_checkpoint
run = Path(sys.argv[1])
saved = json.loads((run / 'identity.json').read_text())
cfg = load_config('configs/train-full-t4-e1.yaml')
assert str(Path(cfg.drive_dir) / cfg.run_name) == str(run), 'Config output path mismatch'
current = run_identity(cfg, _dependency_versions())
assert current == saved, f'Run identity mismatch: {current}'
checkpoint = latest_complete_checkpoint(run / 'resume', current)
assert checkpoint is not None, 'No verified compatible checkpoint; refusing fresh training'
print('VERIFIED_CHECKPOINT=' + str(checkpoint))
'''
    checked = subprocess.run(
        [sys.executable, "-c", preflight, str(run)], cwd=PROJECT,
        capture_output=True, text=True,
    )
    if checked.returncode:
        raise RuntimeError(checked.stdout + checked.stderr)
    checkpoint = next(
        line.split("=", 1)[1] for line in checked.stdout.splitlines()
        if line.startswith("VERIFIED_CHECKPOINT=")
    )
    with (run / "training.log").open("a", buffering=1) as log:
        log.write(f"\n=== Automated resume from {checkpoint}; budget {hours} hours; save every 20 steps ===\n")
        process = subprocess.Popen([
            sys.executable, "-u", str(PROJECT / "scripts/train_with_checkpoint_policy.py"),
            "--save-steps", "20",
            "--config", "configs/train-full-t4-e1.yaml", "--session-hours", str(hours),
        ], cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, pass_fds=(lock.fileno(),))
    PID_FILE.write_text(str(process.pid))
    return supervise(process, run, checkpoint)


def main(config: dict) -> None:
    try:
        run = Path(config["run"])
        if not run.is_dir():
            raise RuntimeError("Run directory unavailable; mount Drive and check the path")
        result = status(run) if config["action"] == "status" else resume(run, config["hours"])
    except Exception as error:
        result = {"action": "error", "error": str(error)}
    # Explicit acknowledgement: some CLI versions print remote exceptions but exit 0.
    print(RESULT_PREFIX + json.dumps(result))
