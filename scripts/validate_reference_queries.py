"""Phase 1: build splits/manifests and validate the reference SQL queries.

    python scripts/validate_reference_queries.py --config configs/data.yaml [--allow-unpinned]

Outputs (in output_dir):
  databases.json         db_id -> path, sha256, bytes
  examples.jsonl         train/dev rows in full; test rows as IDs + db_id only
  dev_subset.json        the fixed seeded dev selection
  exclusions.jsonl       example_id, split, db_id, reason (no questions/SQL)
  integrity_summary.json counts by split and reason, run identity

Exit code 2 if any reference query result fails comparison with itself.
This indicates a comparator defect, not an invalid data example.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from text_to_sql import EVALUATOR_VERSION
from text_to_sql.data.splits import assign_splits, build_database_manifest, db_path, load_spider_file, select_dev_subset
from text_to_sql.evaluation.reference import (
    AUDIT_METHOD,
    NEEDS_ADJUDICATION,
    SELF_COMPARE_FAILED,
    ReferenceCache,
    ReferenceRecord,
    audit_reference,
    reference_cache_key,
)
from text_to_sql.io import write_jsonl
from text_to_sql.sql.executor import Limits

REQUIRED_PINS = [("bundle", "url"), ("bundle", "sha256"), ("hf_dataset", "revision")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--allow-unpinned", action="store_true", help="dry run only; outputs are not freezable")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    unpinned = [".".join(k) for k in REQUIRED_PINS if cfg[k[0]][k[1]] is None]
    if unpinned and not args.allow_unpinned:
        print(f"refusing to run: unpinned {unpinned}", file=sys.stderr)
        return 1

    root = Path(cfg["spider_root"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    limits = Limits(**cfg["limits"])

    train_rows = [r for f in cfg["train_files"] for r in load_spider_file(root / f, "train")]
    test_rows = [r for f in cfg["test_files"] for r in load_spider_file(root / f, "test")]
    examples = assign_splits(train_rows, test_rows, cfg["dev_db_fraction"], cfg["split_seed"])

    manifest = build_database_manifest(root / "database", (e.db_id for e in examples))
    (out / "databases.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    write_jsonl(
        out / "examples.jsonl",
        (e.to_json() if e.split != "test" else {"example_id": e.example_id, "split": "test", "db_id": e.db_id}
         for e in examples),
    )
    dev_subset = select_dev_subset(examples, cfg["dev_subset_size"], cfg["dev_subset_seed"])
    (out / "dev_subset.json").write_text(json.dumps(dev_subset, indent=1))

    adjudications = yaml.safe_load(Path(cfg["adjudications"]).read_text()) or {}
    cache = ReferenceCache(out / "reference_cache.jsonl")

    keyed = [(e, reference_cache_key(manifest[e.db_id]["sha256"], e.query, limits)) for e in examples]
    todo = {k: e for e, k in keyed if cache.get(k) is None}
    print(f"{len(examples)} examples, {len(todo)} reference queries to validate", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:  # each audit spawns its own worker process
        futures = {k: pool.submit(audit_reference, db_path(root / "database", e.db_id), e.query, limits) for k, e in todo.items()}
        for i, (k, fut) in enumerate(futures.items(), 1):
            cache.put(k, fut.result())
            if i % 500 == 0:
                print(f"  audited {i}/{len(todo)}", flush=True)

    exclusions, counts, defects = [], Counter(), []
    for e, k in keyed:
        rec: ReferenceRecord = cache.get(k)
        eligible, reason = rec.eligible, rec.reason
        if reason == NEEDS_ADJUDICATION and e.example_id in adjudications:
            adj = adjudications[e.example_id]
            eligible, reason = bool(adj["eligible"]), None if adj["eligible"] else "adjudicated_ineligible"
        if reason == SELF_COMPARE_FAILED:
            defects.append(e.example_id)
        counts[(e.split, "eligible" if eligible else reason)] += 1
        if not eligible:
            exclusions.append({"example_id": e.example_id, "split": e.split, "db_id": e.db_id, "reason": reason})
    write_jsonl(out / "exclusions.jsonl", exclusions)

    summary = {
        "evaluator_version": EVALUATOR_VERSION,
        "audit_method": AUDIT_METHOD,
        "sqlite_version": sqlite3.sqlite_version,
        "limits": cfg["limits"],
        "pins": {"bundle": cfg["bundle"], "hf_dataset": cfg["hf_dataset"]},
        "unpinned_dry_run": bool(unpinned),
        "n_databases": len(manifest),
        "counts": {f"{s}/{r}": n for (s, r), n in sorted(counts.items())},
        "pending_adjudication": sum(1 for x in exclusions if x["reason"] == NEEDS_ADJUDICATION),
        "comparator_defects": defects,
    }
    (out / "integrity_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["counts"], indent=1))
    return 2 if defects else 0


if __name__ == "__main__":
    sys.exit(main())
