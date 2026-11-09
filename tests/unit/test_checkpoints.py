import json

import pytest

from text_to_sql.checkpoints import (
    MARKER,
    ChecksumMismatch,
    complete_checkpoints,
    latest_complete_checkpoint,
    prune_checkpoints,
    publish_checkpoint,
    verify_manifest,
    write_manifest,
)

ID = {"run_name": "r", "config_hash": "abc"}


def make_local(tmp_path, step, payload="w"):
    d = tmp_path / "local" / f"checkpoint-{step}"
    (d / "sub").mkdir(parents=True)
    (d / "adapter.bin").write_text(payload * 10)
    (d / "sub" / "optimizer.pt").write_text(f"opt{step}")
    return d


def test_manifest_detects_changes(tmp_path):
    d = make_local(tmp_path, 1)
    write_manifest(d)
    verify_manifest(d)
    (d / "adapter.bin").write_text("tampered")
    with pytest.raises(ChecksumMismatch):
        verify_manifest(d)


def test_publish_verifies_then_marks(tmp_path):
    dest = publish_checkpoint(make_local(tmp_path, 100), tmp_path / "drive", ID)
    assert (dest / MARKER).is_file()
    assert json.loads((dest / MARKER).read_text())["identity"] == ID
    assert (dest / "sub" / "optimizer.pt").read_text() == "opt100"


def test_publish_replaces_partial_upload(tmp_path):
    partial = tmp_path / "drive" / "checkpoint-100"
    partial.mkdir(parents=True)
    (partial / "junk").write_text("half-copied")  # no marker
    dest = publish_checkpoint(make_local(tmp_path, 100), tmp_path / "drive", ID)
    assert not (dest / "junk").exists() and (dest / MARKER).exists()


def test_latest_complete_respects_identity_and_markers(tmp_path):
    drive = tmp_path / "drive"
    for step in (100, 200):
        publish_checkpoint(make_local(tmp_path, step), drive, ID)
    incomplete = drive / "checkpoint-300"
    incomplete.mkdir()
    assert latest_complete_checkpoint(drive, ID).name == "checkpoint-200"
    assert latest_complete_checkpoint(drive, {"run_name": "other"}) is None


def test_latest_complete_skips_corrupted(tmp_path):
    drive = tmp_path / "drive"
    for step in (100, 200):
        publish_checkpoint(make_local(tmp_path, step), drive, ID)
    (drive / "checkpoint-200" / "adapter.bin").write_text("bitrot")
    assert latest_complete_checkpoint(drive, ID).name == "checkpoint-100"


def test_steps_sort_numerically(tmp_path):
    drive = tmp_path / "drive"
    for step in (900, 1000, 80):
        publish_checkpoint(make_local(tmp_path, step), drive, ID)
    assert [p.name for p in complete_checkpoints(drive)] == ["checkpoint-80", "checkpoint-900", "checkpoint-1000"]


# ---- prune_checkpoints: the deletion policy


def test_prune_keeps_newest_n(tmp_path):
    drive = tmp_path / "drive"
    for step in (100, 200, 300, 400):
        publish_checkpoint(make_local(tmp_path, step), drive, ID)
    deleted = prune_checkpoints(drive, keep=2)
    assert sorted(p.name for p in deleted) == ["checkpoint-100", "checkpoint-200"]
    assert [p.name for p in complete_checkpoints(drive)] == ["checkpoint-300", "checkpoint-400"]


def test_prune_never_touches_incomplete(tmp_path):
    drive = tmp_path / "drive"
    publish_checkpoint(make_local(tmp_path, 100), drive, ID)
    uploading = drive / "checkpoint-200"
    uploading.mkdir()
    (uploading / "partial").write_text("x")
    assert prune_checkpoints(drive, keep=1) == []
    assert (drive / "checkpoint-100").exists() and uploading.exists()


def test_prune_with_fewer_than_keep_deletes_nothing(tmp_path):
    drive = tmp_path / "drive"
    publish_checkpoint(make_local(tmp_path, 100), drive, ID)
    assert prune_checkpoints(drive, keep=2) == []
    assert prune_checkpoints(tmp_path / "missing", keep=2) == []


def test_prune_refuses_keep_zero(tmp_path):
    drive = tmp_path / "drive"
    publish_checkpoint(make_local(tmp_path, 100), drive, ID)
    with pytest.raises(ValueError):
        prune_checkpoints(drive, keep=0)
    assert (drive / "checkpoint-100").exists()


def test_prune_ignores_non_checkpoint_dirs(tmp_path):
    drive = tmp_path / "drive"
    for step in (100, 200):
        publish_checkpoint(make_local(tmp_path, step), drive, ID)
    (drive / "final").mkdir()
    (drive / "final" / MARKER).write_text("{}")
    prune_checkpoints(drive, keep=1)
    assert (drive / "final").exists() and (drive / "checkpoint-200").exists()
