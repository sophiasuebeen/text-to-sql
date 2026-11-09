"""Checkpoint durability on Google Drive (see docs/model-card.md).

Life of a checkpoint:
1. Trainer writes it locally (fast disk).
2. ``publish_checkpoint`` writes a SHA-256 manifest, copies the directory to a
   unique Drive path, re-hashes the copy, and only then writes a COMPLETE marker
   carrying the run identity.
3. ``prune_checkpoints`` deletes older *complete* resume checkpoints beyond the
   newest N - never the only valid one, never incomplete ones still uploading.
4. On restart, ``latest_complete_checkpoint`` returns the newest checkpoint whose
   marker matches the current run identity; anything else is ignored.

Pure standard library: tested on the Mac without torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

MANIFEST = "MANIFEST.sha256.json"
MARKER = "COMPLETE.json"
_STEP_RE = re.compile(r"^checkpoint-(\d+)$")


class ChecksumMismatch(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    """Relative path -> sha256 for every file except our own manifest/marker."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in (MANIFEST, MARKER):
            out[p.relative_to(root).as_posix()] = _sha256(p)
    return out


def write_manifest(root: Path) -> dict[str, str]:
    manifest = hash_tree(root)
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def verify_manifest(root: Path) -> None:
    expected = json.loads((root / MANIFEST).read_text())
    actual = hash_tree(root)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
        raise ChecksumMismatch(f"missing={missing[:3]} extra={extra[:3]} changed={changed[:3]}")


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    os.replace(tmp, path)


def is_complete(ckpt: Path, identity: dict[str, Any] | None = None) -> bool:
    marker = ckpt / MARKER
    if not marker.is_file():
        return False
    if identity is not None and json.loads(marker.read_text()).get("identity") != identity:
        return False
    return True


def publish_checkpoint(local_ckpt: Path, dest_root: Path, identity: dict[str, Any]) -> Path:
    """Copy local_ckpt to dest_root/<name>, verify, then mark complete. Returns dest path."""
    write_manifest(local_ckpt)
    dest = dest_root / local_ckpt.name
    if dest.exists():
        if is_complete(dest, identity):
            return dest  # already published (e.g. epoch-end save at a step already saved)
        shutil.rmtree(dest)  # leftover partial upload from an interrupted session
    dest_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_ckpt, dest)
    verify_manifest(dest)
    _atomic_write_json(dest / MARKER, {"identity": identity, "files": len(hash_tree(dest))})
    return dest


def checkpoint_step(path: Path) -> int | None:
    m = _STEP_RE.match(path.name)
    return int(m.group(1)) if m else None


def complete_checkpoints(root: Path, identity: dict[str, Any] | None = None) -> list[Path]:
    """Complete checkpoints under root, oldest first by global step."""
    if not root.is_dir():
        return []
    found = [p for p in root.iterdir() if p.is_dir() and checkpoint_step(p) is not None and is_complete(p, identity)]
    return sorted(found, key=checkpoint_step)


def latest_complete_checkpoint(root: Path, identity: dict[str, Any]) -> Path | None:
    for ckpt in reversed(complete_checkpoints(root, identity)):
        try:
            verify_manifest(ckpt)
            return ckpt
        except (ChecksumMismatch, FileNotFoundError, json.JSONDecodeError):
            continue  # corrupted on Drive: fall back to the previous one
    return None


def prune_checkpoints(root: Path, keep: int) -> list[Path]:
    """Delete older complete resume checkpoints, keeping the newest ``keep``.

    Returns the deleted paths.
    """
    if(keep < 1):
        raise ValueError(f"keep={keep} must be at least 1") #keep 0 would delete nothing
    current = complete_checkpoints(root)
    for ckpt in current[:-keep]: # all except the last `keep` checkpoints
        shutil.rmtree(ckpt)
    
    return current[:-keep] # Return the deleted paths
