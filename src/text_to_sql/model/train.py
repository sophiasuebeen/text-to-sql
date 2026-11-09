"""QLoRA fine-tuning (smoke run, then full training). Runs in Colab on a GPU.

    python -m text_to_sql.model.train --config configs/train.yaml            # full run
    python -m text_to_sql.model.train --config configs/train.yaml --smoke    # ~200 examples

Re-running the same command after a disconnect resumes from the newest complete,
identity-compatible checkpoint on Drive (optimizer, scheduler, RNG, scaler and
data position are restored by the Trainer).

Layout on Drive, under output.drive_dir/<run_name>/:
  resume/checkpoint-<step>/   full resumable state; newest N kept
  epochs/epoch-<k>/           adapter-only export at each epoch end; never pruned,
                              scored on dev afterwards to pick the best checkpoint
  final/                      adapter + tokenizer export after training
  status.json                 heartbeat for optional monitoring

Only stdlib/yaml at module level; ML libraries are imported inside functions so
the config/data code is testable on the Mac and in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .. import EVALUATOR_VERSION
from .checkpoints import (
    complete_checkpoints,
    latest_complete_checkpoint,
    prune_checkpoints,
    publish_checkpoint,
    write_manifest,
)
from ..data.prepare import build_supervised_example, pad_batch
from ..io import read_jsonl
from ..data.schema import SERIALIZER_VERSION, serialize_schema
from ..data.splits import db_path
from .prompts import PROMPT_VERSION

# ---------------------------------------------------------------- config


@dataclass(frozen=True)
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class TrainingSettings:
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 3
    max_steps: int = -1
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 10
    save_steps: int = 100
    keep_resume_checkpoints: int = 2

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    seed: int
    model_name: str
    model_revision: str | None
    phase1_dir: str
    spider_root: str
    max_length: int
    local_dir: str
    drive_dir: str
    max_train_examples: int | None = None
    lora: LoraSettings = field(default_factory=LoraSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)

    def semantic_hash(self) -> str:
        """Hash of everything that changes the result (paths excluded)."""
        d = asdict(self)
        for k in ("local_dir", "drive_dir", "phase1_dir", "spider_root"):
            d.pop(k)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


class ConfigError(ValueError):
    pass


def load_config(path: str | Path, *, smoke: bool = False, allow_unpinned: bool = False) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text())
    lora = LoraSettings(**{**raw.get("lora", {}), "target_modules": tuple(raw.get("lora", {}).get(
        "target_modules", LoraSettings.target_modules))})
    training = TrainingSettings(**raw.get("training", {}))
    cfg = TrainConfig(
        run_name=raw["run_name"],
        seed=raw["seed"],
        model_name=raw["model"]["name"],
        model_revision=raw["model"].get("revision"),
        phase1_dir=raw["data"]["phase1_dir"],
        spider_root=raw["data"]["spider_root"],
        max_length=raw["context"]["max_length"],
        local_dir=raw["output"]["local_dir"],
        drive_dir=raw["output"]["drive_dir"],
        max_train_examples=raw["data"].get("max_train_examples"),
        lora=lora,
        training=training,
    )
    if smoke:
        smoke_cfg = raw.get("smoke", {})
        cfg = TrainConfig(**{
            **cfg.__dict__,
            "run_name": cfg.run_name + "-smoke",
            "max_train_examples": smoke_cfg.get("max_train_examples", 200),
            "training": TrainingSettings(**{**asdict(training), **smoke_cfg.get("training", {})}),
        })
    validate_config(cfg, allow_unpinned=allow_unpinned)
    return cfg


def validate_config(cfg: TrainConfig, *, allow_unpinned: bool = False) -> None:
    if cfg.model_revision is None and not allow_unpinned:
        raise ConfigError("model.revision must be pinned to a commit hash (reproducibility)")
    if cfg.training.effective_batch_size != 16:
        raise ConfigError(
            f"effective batch size is {cfg.training.effective_batch_size}, plan fixes 16; "
            "for OOM, lower per_device_train_batch_size and raise gradient_accumulation_steps"
        )
    if cfg.training.save_steps <= 0 or cfg.training.keep_resume_checkpoints < 1:
        raise ConfigError("save_steps must be > 0 and keep_resume_checkpoints >= 1")
    if cfg.max_length <= 0:
        raise ConfigError("context.max_length must be positive")


# ---------------------------------------------------------------- identity


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def run_identity(cfg: TrainConfig, dependency_versions: dict[str, str]) -> dict[str, Any]:
    """Everything that must match for a checkpoint to be resumed."""
    return {
        "run_name": cfg.run_name,
        "config_hash": cfg.semantic_hash(),
        "model": cfg.model_name,
        "model_revision": cfg.model_revision,
        "code_commit": git_commit(),
        "prompt_version": PROMPT_VERSION,
        "serializer_version": SERIALIZER_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "dependencies": dependency_versions,
    }


# ---------------------------------------------------------------- data


@dataclass
class DataReport:
    candidates: int = 0
    dropped_gold_invalid: int = 0
    dropped_overlength: int = 0
    kept: int = 0
    max_kept_length: int = 0


# Gold that cannot be parsed/executed is not valid supervision. Ambiguous ordering
# or unordered LIMIT only affect scoring, so those examples still train.
_INVALID_FOR_TRAINING = ("gold_parse_error", "gold_sql_error", "gold_timeout", "gold_resource_limit",
                         "gold_unsupported_value", "gold_disallowed", "gold_worker_error")


def load_training_features(cfg: TrainConfig, tokenizer) -> tuple[list[dict[str, list[int]]], DataReport]:
    phase1 = Path(cfg.phase1_dir)
    rows = [r for r in read_jsonl(phase1 / "examples.jsonl") if r["split"] == "train"]
    excluded = {x["example_id"]: x["reason"] for x in read_jsonl(phase1 / "exclusions.jsonl")}
    rows.sort(key=lambda r: r["example_id"])
    if cfg.max_train_examples is not None:
        # Seeded, so the smoke subset is the same in every session (needed for resume).
        import random

        rows = sorted(random.Random(cfg.seed).sample(rows, min(cfg.max_train_examples, len(rows))),
                      key=lambda r: r["example_id"])

    report = DataReport(candidates=len(rows))
    schemas: dict[str, str] = {}
    features = []
    for r in rows:
        if excluded.get(r["example_id"]) in _INVALID_FOR_TRAINING:
            report.dropped_gold_invalid += 1
            continue
        if r["db_id"] not in schemas:
            schemas[r["db_id"]] = serialize_schema(db_path(Path(cfg.spider_root) / "database", r["db_id"]))
        f = build_supervised_example(tokenizer, schemas[r["db_id"]], r["question"], r["query"])
        if len(f["input_ids"]) > cfg.max_length:
            report.dropped_overlength += 1  # dropped and counted, never truncated
            continue
        features.append(f)
        report.max_kept_length = max(report.max_kept_length, len(f["input_ids"]))
    report.kept = len(features)
    return features, report


# ---------------------------------------------------------------- status heartbeat


def write_status(path: Path, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"heartbeat_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


# ---------------------------------------------------------------- GPU training


def _dependency_versions() -> dict[str, str]:
    import bitsandbytes
    import peft
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }


def build_model_and_tokenizer(cfg: TrainConfig):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training needs a CUDA GPU (use a Colab GPU runtime)")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4 -> fp16

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        torch_dtype=compute_dtype,
        device_map={"": 0},
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    return model, tokenizer, compute_dtype


def _make_callbacks(cfg: TrainConfig, identity: dict[str, Any], run_root: Path, tokenizer):
    from transformers import TrainerCallback

    status_path = run_root / "status.json"

    class DriveCheckpointCallback(TrainerCallback):
        """Publish each local checkpoint to Drive, verified, then prune."""

        def on_epoch_end(self, args, state, control, **kw):
            control.should_save = True  # checkpoint at every epoch boundary too
            return control

        def on_save(self, args, state, control, model=None, **kw):
            local = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_status(status_path, run=cfg.run_name, state="checkpointing", step=state.global_step)
            publish_checkpoint(local, run_root / "resume", identity)
            prune_checkpoints(run_root / "resume", cfg.training.keep_resume_checkpoints)

            if state.epoch is not None and abs(state.epoch - round(state.epoch)) < 1e-6:
                k = int(round(state.epoch))
                export = Path(args.output_dir) / f"epoch-{k}"
                model.save_pretrained(export)
                tokenizer.save_pretrained(export)
                (export / "identity.json").write_text(json.dumps({**identity, "epoch": k, "step": state.global_step}))
                publish_checkpoint(export, run_root / "epochs", identity)
                shutil.rmtree(export)
            write_status(status_path, run=cfg.run_name, state="training", step=state.global_step)

    class StatusCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            loss = (logs or {}).get("loss")
            if loss is not None and loss != loss:  # NaN
                write_status(status_path, run=cfg.run_name, state="failed", step=state.global_step,
                             error="non-finite loss")
                control.should_training_stop = True
                return control
            write_status(status_path, run=cfg.run_name, state="training", step=state.global_step, loss=loss)

    return [DriveCheckpointCallback(), StatusCallback()]


def train(cfg: TrainConfig) -> Path:
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(cfg.seed)
    identity = run_identity(cfg, _dependency_versions())
    run_root = Path(cfg.drive_dir) / cfg.run_name
    local_out = Path(cfg.local_dir) / cfg.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "identity.json").write_text(json.dumps(identity, indent=1))

    model, tokenizer, compute_dtype = build_model_and_tokenizer(cfg)
    features, report = load_training_features(cfg, tokenizer)
    (run_root / "data_report.json").write_text(json.dumps(asdict(report), indent=1))
    print(f"training data: {asdict(report)}")
    if not features:
        raise RuntimeError("no training examples left after filtering")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate(batch):
        return {k: torch.tensor(v) for k, v in pad_batch(batch, pad_id).items()}

    resume = latest_complete_checkpoint(run_root / "resume", identity)
    resume_local = None
    if resume is not None:
        resume_local = local_out / resume.name
        if not resume_local.exists():
            shutil.copytree(resume, resume_local)
        print(f"resuming from {resume} (step {resume.name.split('-')[-1]})")
    elif complete_checkpoints(run_root / "resume"):
        raise RuntimeError("Drive has checkpoints for this run name with a different identity; "
                           "refusing to start over silently. Use a new run_name or restore the matching code.")

    t = cfg.training
    args = TrainingArguments(
        output_dir=str(local_out),
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_ratio=t.warmup_ratio,
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps,
        logging_steps=t.logging_steps,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.keep_resume_checkpoints,  # local disk only; Drive pruning is ours
        optim=t.optim,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        seed=cfg.seed,
        data_seed=cfg.seed,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=features,
        data_collator=collate,
        callbacks=_make_callbacks(cfg, identity, run_root, tokenizer),
    )

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=str(resume_local) if resume_local else None)

    final = local_out / "final"
    trainer.model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    (final / "identity.json").write_text(json.dumps(identity, indent=1))
    write_manifest(final)
    if (run_root / "final").exists():
        shutil.rmtree(run_root / "final")
    publish_checkpoint(final, run_root, identity)

    write_status(
        run_root / "status.json", run=cfg.run_name, state="complete", step=trainer.state.global_step,
        peak_gpu_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        session_seconds=round(time.time() - started),
    )
    return run_root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--smoke", action="store_true", help="short run on a seeded ~200-example subset")
    ap.add_argument("--allow-unpinned", action="store_true", help="debug only: unpinned model revision")
    args = ap.parse_args()
    cfg = load_config(args.config, smoke=args.smoke, allow_unpinned=args.allow_unpinned)
    print(f"run {cfg.run_name}: config hash {cfg.semantic_hash()}")
    print(f"done: {train(cfg)}")


if __name__ == "__main__":
    main()
