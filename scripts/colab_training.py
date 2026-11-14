"""Control the recorded full-data Colab run from a Mac; never retry lost sessions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "/content/drive/MyDrive/text-to-sql/runs/qlora-r16-lr2e-4-full-e1"
PREFIX = "TEXT_TO_SQL_RESULT="


def parse_result(output: str) -> dict:
    matches = [line.split(PREFIX, 1)[1] for line in output.splitlines() if PREFIX in line]
    if not matches:
        raise RuntimeError("No remote acknowledgement; session may be lost. Not retrying or creating another GPU session.")
    result = json.loads(matches[-1])
    if result.get("action") == "error":
        raise RuntimeError(result.get("error", "Remote operation failed"))
    return result


def cli(*args: str, code: str | None = None) -> str:
    if code is not None:
        # Stream real progress while keeping the remote kernel execution open.
        process = subprocess.Popen(
            ["colab", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, cwd=ROOT, bufsize=1,
        )
        process.stdin.write(code)
        process.stdin.close()
        lines = []
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        if process.wait():
            raise RuntimeError("".join(lines[-40:]) or "Colab execution failed")
        return "".join(lines)
    result = subprocess.run(
        ["colab", *args], input=code, text=True,
        capture_output=code is not None, cwd=ROOT,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or f"colab failed: {args}").strip())
    return result.stdout or ""


def remote(args, action: str) -> dict:
    source = (ROOT / "scripts/colab_remote.py").read_text()
    config = {"action": action, "run": args.run, "hours": args.session_hours}
    code = source + "\nmain(" + repr(config) + ")\n"
    return parse_result(cli("exec", "-s", args.session, "--timeout",
                            str(int(args.session_hours * 3600) + 3600) if action == "resume" else "120", code=code))


def display(result: dict) -> None:
    tail = result.pop("log_tail", None)
    print(json.dumps(result, indent=2))
    if tail:
        print("\n=== Recent training log ===\n" + tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["resume", "status"])
    parser.add_argument("--session", default="sql-resume")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--session-hours", type=float, default=2)
    parser.add_argument("--create", action="store_true", help="Create one T4 session; never retry automatically")
    parser.add_argument("--skip-mount", action="store_true", help="Use an already mounted Drive; status still verifies the run directory")
    args = parser.parse_args()
    if not math.isfinite(args.session_hours) or not 0 < args.session_hours <= 2:
        parser.error("--session-hours must be greater than zero and at most 2")
    if args.create and args.action != "resume":
        parser.error("--create is only supported with resume")
    try:
        if args.action == "status":
            display(remote(args, "status"))
            return
        if args.create:
            cli("new", "-s", args.session, "--gpu", "T4")
        if not args.skip_mount:
            cli("drivemount", "-s", args.session)
        before = remote(args, "status")
        if before["pids"] or before["final_marker"]:
            print("Training is active or a final export exists; setup and launch skipped.")
            display(before)
            return
        with tempfile.TemporaryDirectory(prefix="text-to-sql-colab-") as temp:
            archive = Path(temp) / "text-to-sql-resume.tar.gz"
            with tarfile.open(archive, "w:gz") as target:
                for name in ["src", "scripts", "configs", "pyproject.toml", "README.md", "uv.lock"]:
                    target.add(ROOT / name, arcname=name, filter=lambda info: None if
                               "__pycache__" in Path(info.name).parts or info.name.endswith(".pyc") else info)
            cli("upload", "-s", args.session, str(archive), "/content/text-to-sql-resume.tar.gz")
        display(remote(args, "resume"))
    except (RuntimeError, OSError, ValueError) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
