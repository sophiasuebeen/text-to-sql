# Model card

> Status: **not trained yet.** This card records the planned setup; results are
> added after the final evaluation.

## Base model

`Qwen/Qwen2.5-Coder-1.5B-Instruct` (Apache-2.0), pinned to a commit revision in
`configs/train.yaml`.

## Method: QLoRA

| Setting | Value |
|---|---|
| Quantization | 4-bit NF4, double quantization; fp16 compute on T4, bf16 where supported |
| LoRA | r=16, alpha=32, dropout 0.05, on `q_proj, k_proj, v_proj, o_proj` |
| Optimizer | paged AdamW 8-bit, learning rate 2e-4, cosine schedule, 3% warmup |
| Batch | 4 per device × 4 accumulation = effective 16 (fixed; lower per-device batch on OOM) |
| Epochs | up to 3, gradient checkpointing, fixed seed |
| Loss | only SQL tokens and EOS are supervised; prompt tokens are masked |

## Checkpoints

Saved every 100 optimizer steps and at each epoch boundary. Each checkpoint is
checksummed, copied to Google Drive, verified, and only then marked complete.
Resume uses the newest complete checkpoint with a matching run identity (config
hash, model revision, code commit, dependency versions). The newest two resume
checkpoints are kept; epoch-end adapter exports are never pruned.

The best checkpoint is chosen by dev execution accuracy, not by loss.

## Intended use

A learning and portfolio project: SQLite text-to-SQL over schemas that fit in
the context window. Not evaluated on other SQL dialects or arbitrary schemas.

## Results

_To be added after the final test evaluation._
