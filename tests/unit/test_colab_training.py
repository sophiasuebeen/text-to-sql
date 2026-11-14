"""Offline checks for the Colab controller; never allocate GPU sessions."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


controller = load_script("colab_training")
worker = load_script("colab_remote")


def test_missing_acknowledgement_is_failure():
    with pytest.raises(RuntimeError, match="Not retrying"):
        controller.parse_result("Session lost (404/401)")


def test_remote_exception_is_failure_even_if_cli_exits_zero():
    with pytest.raises(RuntimeError, match="Identity mismatch"):
        controller.parse_result('TEXT_TO_SQL_RESULT={"action":"error","error":"Identity mismatch"}')


def test_acknowledgement_amid_cli_output():
    assert controller.parse_result(
        'CLI update notice\nTEXT_TO_SQL_RESULT={"action":"started","pid":42}\n'
    ) == {"action": "started", "pid": 42}


@pytest.mark.parametrize("active,final", [(True, False), (False, True)])
def test_existing_training_or_export_does_not_upload_or_launch(monkeypatch, active, final):
    monkeypatch.setattr("sys.argv", ["colab_training.py", "resume"])
    calls = []
    monkeypatch.setattr(controller, "cli", lambda *args, **kwargs: calls.append(args))

    def fake_remote(args, action):
        assert action == "status", "Must not launch another training process"
        return {"pids": [42] if active else [], "final_marker": final}

    monkeypatch.setattr(controller, "remote", fake_remote)
    controller.main()
    assert calls == [("drivemount", "-s", "sql-resume")]


def test_lost_session_stops_without_creating_replacement(monkeypatch):
    monkeypatch.setattr("sys.argv", ["colab_training.py", "resume"])
    calls = []
    monkeypatch.setattr(controller, "cli", lambda *args, **kwargs: calls.append(args))

    def fail(*args):
        raise RuntimeError("Session lost")

    monkeypatch.setattr(controller, "remote", fail)
    with pytest.raises(SystemExit) as error:
        controller.main()
    assert error.value.code == 1
    assert calls == [("drivemount", "-s", "sql-resume")]


def test_remote_nonzero_exit_is_not_success(monkeypatch):
    monkeypatch.setattr(controller.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="", stderr="Access denied"))
    with pytest.raises(RuntimeError, match="Access denied"):
        controller.cli("sessions")


def test_status_reports_markers_without_claiming_checksums_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "training_pids", lambda: [42])
    (tmp_path / "resume/checkpoint-50").mkdir(parents=True)
    (tmp_path / "resume/checkpoint-50/COMPLETE.json").write_text("{}")
    (tmp_path / "resume/checkpoint-75").mkdir()
    (tmp_path / "status.json").write_text(json.dumps({"state": "training", "step": 60}))
    (tmp_path / "training.log").write_text("\n".join(str(i) for i in range(100)))
    result = worker.status(tmp_path)
    assert result["checkpoint_markers"] == [50]
    assert result["pids"] == [42]
    assert result["saved_status"]["step"] == 60
    assert result["log_tail"].splitlines() == [str(i) for i in range(80, 100)]


def test_missing_drive_is_reported_in_acknowledgement(tmp_path, capsys):
    worker.main({"action": "status", "run": str(tmp_path / "missing"), "hours": 2})
    with pytest.raises(RuntimeError, match="mount Drive"):
        controller.parse_result(capsys.readouterr().out)


def test_worker_lock_blocks_duplicate_setup(tmp_path, monkeypatch):
    import fcntl

    lock_path = tmp_path / "training.lock"
    with lock_path.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(worker, "open", lambda *args, **kwargs: lock_path.open("a"), raising=False)
        monkeypatch.setattr(worker, "training_pids", lambda: [42])
        assert worker.resume(tmp_path, 2) == {"action": "already_running", "pids": [42]}


def test_supervisor_waits_and_records_budget_pause(tmp_path, monkeypatch, capsys):
    process = SimpleNamespace(pid=42, returncode=None)
    process.poll = lambda: process.returncode
    monkeypatch.setattr(worker, "status", lambda run: {
        "saved_status": {"state": "paused", "step": 100},
        "checkpoint_markers": [100], "log_tail": "100/377",
    })
    sleeps = []

    def advance(seconds):
        sleeps.append(seconds)
        process.returncode = 0

    monkeypatch.setattr(worker.time, "sleep", advance)
    result = worker.supervise(process, tmp_path, "checkpoint-50")
    assert sleeps == [30]
    assert result["action"] == "paused"
    assert result["saved_status"]["step"] == 100
    exit_path, = (tmp_path / "invocations").glob("*/exit.json")
    assert json.loads(exit_path.read_text())["returncode"] == 0
    assert "TEXT_TO_SQL_PROGRESS=" in capsys.readouterr().out


def test_supervisor_records_failed_process(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=42, returncode=-9, poll=lambda: -9)
    monkeypatch.setattr(worker, "status", lambda run: {"saved_status": {"state": "training"}})
    with pytest.raises(RuntimeError, match="code -9"):
        worker.supervise(process, tmp_path, "checkpoint-50")
    exit_path, = (tmp_path / "invocations").glob("*/exit.json")
    assert json.loads(exit_path.read_text())["returncode"] == -9


def test_checkpoint_policy_overrides_restored_interval(monkeypatch, capsys):
    import sys

    policy = load_script("train_with_checkpoint_policy")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainerCallback=object))
    args = SimpleNamespace(save_steps=25)
    state = SimpleNamespace(save_steps=25, global_step=50)
    control = SimpleNamespace(should_save=False)
    assert policy.checkpoint_callback(20).on_train_begin(args, state, control) is control
    assert args.save_steps == state.save_steps == 20
    assert not control.should_save
    assert "next periodic save at 60" in capsys.readouterr().out


def test_checkpoint_wrapper_preserves_existing_callbacks_and_training_args(monkeypatch):
    import sys
    from text_to_sql.model import train

    policy = load_script("train_with_checkpoint_policy")
    sentinel = object()
    budget = object()
    received = []

    def original_callbacks(*args, **kwargs):
        received.append(kwargs["budget"])
        return [sentinel]

    monkeypatch.setattr(train, "_make_callbacks", original_callbacks)
    monkeypatch.setattr(policy, "checkpoint_callback", lambda interval: ("checkpoint_policy", interval))
    original_argv = ["wrapper", "--save-steps", "20", "--config", "saved.yaml", "--session-hours", "2"]
    monkeypatch.setattr(sys, "argv", original_argv)

    def fake_main():
        assert sys.argv[1:] == ["--config", "saved.yaml", "--session-hours", "2"]
        assert train._make_callbacks(budget=budget) == [sentinel, ("checkpoint_policy", 20)]

    monkeypatch.setattr(train, "main", fake_main)
    policy.main()
    assert received == [budget]
    assert train._make_callbacks is original_callbacks
    assert sys.argv is original_argv


def test_skip_mount_still_checks_existing_training(monkeypatch):
    monkeypatch.setattr("sys.argv", ["colab_training.py", "resume", "--skip-mount"])
    calls = []
    monkeypatch.setattr(controller, "cli", lambda *args, **kwargs: calls.append(args))
    checked = []

    def inspect(args, action):
        checked.append(action)
        return {"pids": [42], "final_marker": False}

    monkeypatch.setattr(controller, "remote", inspect)
    controller.main()
    assert calls == []
    assert checked == ["status"]
