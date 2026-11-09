"""Spider loading, database-level train/dev/test splits, and split/manifest checks."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..io import sha256_file


@dataclass(frozen=True)
class Example:
    example_id: str
    split: str  # "train" | "dev" | "test"
    db_id: str
    question: str
    query: str
    source: str  # "<file>#<index>" for traceability

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


def load_spider_file(path: str | Path, id_prefix: str) -> list[dict[str, Any]]:
    """Load a Spider question file; IDs are stable positions in the pinned file."""
    path = Path(path)
    rows = json.loads(path.read_text())
    return [
        {"id": f"{id_prefix}-{i:05d}", "db_id": r["db_id"], "question": r["question"],
         "query": r["query"], "source": f"{path.name}#{i}"}
        for i, r in enumerate(rows)
    ]


def split_databases(train_db_ids: Iterable[str], dev_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Hold out ceil(fraction * n) whole training databases as the dev split."""
    dbs = sorted(set(train_db_ids))
    n_dev = max(1, math.ceil(dev_fraction * len(dbs)))
    dev = sorted(random.Random(seed).sample(dbs, n_dev))
    dev_set = set(dev)
    return [d for d in dbs if d not in dev_set], dev


def assign_splits(train_rows: Sequence[dict], test_rows: Sequence[dict], dev_fraction: float, seed: int) -> list[Example]:
    _train_dbs, dev_dbs = split_databases((r["db_id"] for r in train_rows), dev_fraction, seed)
    dev_set = set(dev_dbs)
    out = [
        Example(r["id"], "dev" if r["db_id"] in dev_set else "train", r["db_id"], r["question"], r["query"], r["source"])
        for r in train_rows
    ]
    out += [Example(r["id"], "test", r["db_id"], r["question"], r["query"], r["source"]) for r in test_rows]
    check_disjoint(out)
    return out


def select_dev_subset(examples: Sequence[Example], n: int, seed: int) -> list[str]:
    """Seeded subset of up to n dev IDs, round-robin across databases for coverage."""
    by_db: dict[str, list[str]] = defaultdict(list)
    for e in examples:
        if e.split == "dev":
            by_db[e.db_id].append(e.example_id)
    rng = random.Random(seed)
    queues = []
    for db in sorted(by_db):
        ids = sorted(by_db[db])
        rng.shuffle(ids)
        queues.append(ids)
    rng.shuffle(queues)
    chosen: list[str] = []
    while len(chosen) < n and any(queues):
        for q in queues:
            if q and len(chosen) < n:
                chosen.append(q.pop())
    return sorted(chosen)


# ---------------------------------------------------------------- checks


class SplitError(ValueError):
    pass


def check_disjoint(examples: Sequence[Example]) -> None:
    ids = Counter(e.example_id for e in examples)
    dupes = [i for i, c in ids.items() if c > 1]
    if dupes:
        raise SplitError(f"duplicate example IDs: {dupes[:5]}")
    dbs: dict[str, set[str]] = defaultdict(set)
    for e in examples:
        dbs[e.split].add(e.db_id)
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = dbs[a] & dbs[b]
        if overlap:
            raise SplitError(f"{a}/{b} share databases: {sorted(overlap)[:5]}")


def db_path(db_root: str | Path, db_id: str) -> Path:
    return Path(db_root) / db_id / f"{db_id}.sqlite"


def build_database_manifest(db_root: str | Path, db_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    manifest, missing = {}, []
    for db_id in sorted(set(db_ids)):
        p = db_path(db_root, db_id)
        if not p.is_file():
            missing.append(db_id)
            continue
        manifest[db_id] = {"path": str(p.relative_to(db_root)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
    if missing:
        raise FileNotFoundError(f"{len(missing)} databases missing, e.g. {missing[:5]}")
    return manifest


