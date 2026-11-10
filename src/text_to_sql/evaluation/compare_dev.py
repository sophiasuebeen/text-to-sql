"""Provisional paired baseline/adapter execution accuracy on the frozen dev subset.

This command never reads test questions. Outputs include the current exclusions
and unresolved decisions; this is a diagnostic, not a final benchmark result.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from .. import EVALUATOR_VERSION
from ..data.schema import serialize_schema
from ..data.splits import db_path
from ..io import read_jsonl, sha256_file, write_jsonl
from ..model.predict import Prediction, SQLPredictor, export_identity
from ..sql.executor import Limits
from .metrics import accuracy, macro_accuracy, paired_cluster_bootstrap
from .reference import ReferenceCache, reference_cache_key
from .score import score_prediction


def select_examples(phase1: Path, max_examples: int | None = None) -> tuple[list[dict], list[dict]]:
    """Select the same frozen eligible dev IDs for both models, never replace failures."""
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive")
    selected = set(json.loads((phase1 / "dev_subset.json").read_text()))
    exclusions = read_jsonl(phase1 / "exclusions.jsonl")
    excluded = {r["example_id"] for r in exclusions}
    examples = sorted(
        (r for r in read_jsonl(phase1 / "examples.jsonl")
         if r["split"] == "dev" and r["example_id"] in selected and r["example_id"] not in excluded),
        key=lambda r: r["example_id"],
    )
    if max_examples is not None:
        examples = examples[:max_examples]
    if not examples:
        raise ValueError("no eligible examples in the frozen dev subset")
    return examples, exclusions


def select_training_examples(
    *, adapter: Path, training_config: Path, phase1: Path, spider_root: Path,
) -> tuple[list[dict], list[dict], dict]:
    """Reconstruct the smoke training rows, then apply scoring-only exclusions."""
    from transformers import AutoTokenizer

    from ..model.train import load_config, prepare_training_data

    cfg = load_config(training_config, smoke=True)
    identity = export_identity(adapter)
    if cfg.semantic_hash() != identity["config_hash"]:
        raise ValueError("training config does not match the saved adapter's config hash")
    if (cfg.model_name, cfg.model_revision) != (identity["model"], identity["model_revision"]):
        raise ValueError("training base model differs from the saved adapter")
    if Path(cfg.phase1_dir).resolve() != phase1.resolve() or Path(cfg.spider_root).resolve() != spider_root.resolve():
        raise ValueError("training and evaluation data locations differ")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    _features, report, trained_rows = prepare_training_data(cfg, tokenizer)
    stored_report = json.loads((adapter.parent / "data_report.json").read_text())
    # Support reports written before the terminology rename.
    stored_report = {k.replace("gold", "reference"): v for k, v in stored_report.items()}
    if stored_report != asdict(report):
        raise ValueError("reconstructed training data report differs from the saved training run")
    exclusions = read_jsonl(phase1 / "exclusions.jsonl")
    excluded = {r["example_id"] for r in exclusions}
    examples = [r for r in trained_rows if r["example_id"] not in excluded]
    if not examples:
        raise ValueError("none of the reconstructed training rows is eligible for execution scoring")
    info = {
        "training_config_hash": cfg.semantic_hash(), "training_data_report": asdict(report),
        "n_trained_examples": len(trained_rows), "n_scored_training_examples": len(examples),
        "n_training_examples_excluded_from_scoring": len(trained_rows) - len(examples),
        "trained_example_ids": [r["example_id"] for r in trained_rows],
        "reconstruction_note": "same config, saved tokenizer and current data; report matches saved run; "
                               "original training run did not save per-example IDs or input hashes",
    }
    print(f"Reconstructed {len(trained_rows)} trained examples; "
          f"{len(examples)} eligible for execution scoring", flush=True)
    return examples, exclusions, info


def compare_dev(
    *, adapter: Path, data_config: Path, output: Path,
    max_examples: int | None = None, max_context: int = 2048, max_new_tokens: int = 256,
    resume: bool = False,
    train_smoke_config: Path | None = None,
) -> dict:
    cfg = yaml.safe_load(data_config.read_text())
    phase1 = Path(cfg["output_dir"])
    selection_info = None
    if train_smoke_config is not None:
        if max_examples is not None:
            raise ValueError("do not limit the reconstructed training selection with max_examples")
        examples, exclusions, selection_info = select_training_examples(
            adapter=adapter, training_config=train_smoke_config,
            phase1=phase1, spider_root=Path(cfg["spider_root"]),
        )
    else:
        examples, exclusions = select_examples(phase1, max_examples)
    integrity = json.loads((phase1 / "integrity_summary.json").read_text())
    if integrity["comparator_defects"] or integrity["unpinned_dry_run"]:
        raise ValueError("reference validation has defects or was an unpinned dry run")
    if integrity["evaluator_version"] != EVALUATOR_VERSION:
        raise ValueError("reference results were produced by a different evaluator version")
    if {"bundle": cfg["bundle"], "hf_dataset": cfg["hf_dataset"]} != integrity["pins"]:
        raise ValueError("data configuration pins differ from the reference validation run")
    limits = Limits(**integrity["limits"])
    manifest = json.loads((phase1 / "databases.json").read_text())
    cache = ReferenceCache(phase1 / "reference_cache.jsonl")
    references, schemas = {}, {}
    for example in examples:
        db_id = example["db_id"]
        db = db_path(Path(cfg["spider_root"]) / "database", db_id)
        if db_id not in schemas:
            if sha256_file(db) != manifest[db_id]["sha256"]:
                raise ValueError(f"database changed since reference validation: {db_id}")
            schemas[db_id] = serialize_schema(db)
        key = reference_cache_key(manifest[db_id]["sha256"], example["query"], limits)
        reference = cache.get(key)
        if reference is None or not reference.eligible:
            raise ValueError(f"missing eligible reference cache record: {example['example_id']}")
        references[example["example_id"]] = reference

    if resume:
        if not (output / "run.json").is_file():
            raise ValueError("resume requires an existing run.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
    predictor = SQLPredictor(adapter, max_context=max_context, max_new_tokens=max_new_tokens)
    metadata = {
        "provisional_dev_diagnostic": train_smoke_config is None,
        "adapter_identity": predictor.identity,
        "evaluator_version": EVALUATOR_VERSION,
        "reference_validation": integrity,
        "input_hashes": {name: sha256_file(phase1 / name) for name in
                         ["examples.jsonl", "dev_subset.json", "exclusions.jsonl", "databases.json"]},
        "example_ids": [r["example_id"] for r in examples],
        "max_context": max_context, "max_new_tokens": max_new_tokens,
        "decoding": "greedy", "quantization": "4-bit NF4, double quantization",
        "batch_size": 1, "warmup_examples": 0,
        "latency": "synchronized generation seconds; includes first-call overhead; excludes SQL execution",
    }
    if selection_info is not None:
        metadata["training_memorization_diagnostic"] = selection_info
    completed = []
    if resume:
        if json.loads((output / "run.json").read_text()) != metadata:
            raise ValueError("cannot resume: model, data, selection, or generation settings changed")
        predictions = output / "predictions.jsonl"
        if predictions.exists():
            completed = read_jsonl(predictions)
        if len(completed) > len(examples):
            raise ValueError("resume log contains more rows than the selected examples")
        for row, example in zip(completed, examples):
            for field in ["example_id", "db_id", "question"]:
                if row[field] != example[field]:
                    raise ValueError("resume log is not an ordered prefix of the selected examples")
            if row["reference_sql"] != example["query"]:
                raise ValueError("resume reference SQL changed")
            # Validate both serialized predictions before appending anything.
            for mode in ["baseline", "adapter"]:
                Prediction(**row[mode]["prediction"])
        print(f"Resuming: {len(completed)}/{len(examples)} paired predictions already saved", flush=True)
    else:
        write_jsonl(output / "exclusions.jsonl", exclusions)
        (output / "run.json").write_text(json.dumps(metadata, indent=2))
    started = time.monotonic()
    flags = {"baseline": [], "adapter": []}
    # Re-score saved predictions with the current evaluator without regenerating.
    for row in completed:
        db = db_path(Path(cfg["spider_root"]) / "database", row["db_id"])
        for mode in flags:
            score = score_prediction(db, row[mode]["prediction"]["sql"], references[row["example_id"]], limits)
            row[mode]["score"] = asdict(score)
            flags[mode].append(score.correct)
    if completed:
        # Atomic rewrite preserves the saved predictions while updating their scores.
        write_jsonl(output / "predictions.jsonl", completed)
    with (output / "predictions.jsonl").open("a") as f:
        for i, example in enumerate(examples, 1):
            if i <= len(completed):
                continue
            db = db_path(Path(cfg["spider_root"]) / "database", example["db_id"])
            row = {"example_id": example["example_id"], "db_id": example["db_id"],
                   "question": example["question"], "reference_sql": example["query"]}
            for mode in flags:
                prediction = predictor.predict(schemas[example["db_id"]], example["question"],
                                               baseline=mode == "baseline")
                score = score_prediction(db, prediction.sql, references[example["example_id"]], limits)
                flags[mode].append(score.correct)
                row[mode] = {"prediction": asdict(prediction), "score": asdict(score)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{i}/{len(examples)} baseline={sum(flags['baseline'])}/{i} "
                  f"adapter={sum(flags['adapter'])}/{i}", flush=True)
    summary = {
        "provisional_dev_diagnostic": train_smoke_config is None, "n_examples": len(examples),
        "baseline_accuracy": accuracy(flags["baseline"]),
        "adapter_accuracy": accuracy(flags["adapter"]),
        "baseline_macro_accuracy": macro_accuracy([r["db_id"] for r in examples], flags["baseline"]),
        "adapter_macro_accuracy": macro_accuracy([r["db_id"] for r in examples], flags["adapter"]),
        "paired_bootstrap": asdict(paired_cluster_bootstrap(
            [r["db_id"] for r in examples], flags["baseline"], flags["adapter"], seed=20261110,
        )),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "timing_scope": "this invocation only, including rescoring saved predictions",
        "resumed_examples": len(completed),
        "peak_gpu_mem_gib": None,
    }
    if selection_info is not None:
        summary["training_memorization_diagnostic"] = selection_info
    import torch
    summary["peak_gpu_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--output", type=Path, required=True, help="new directory for this diagnostic run")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--resume", action="store_true", help="reuse completed pairs in the same output directory")
    ap.add_argument("--train-smoke-config", type=Path,
                    help="instead of dev, reconstruct the smoke training examples with this exact config")
    args = ap.parse_args()
    print(json.dumps(compare_dev(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
