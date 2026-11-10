# Model card

> Status: **smoke training completed; full training and final test pending.**
> See [experiment log](experiments.md) for settings, preliminary comparisons,
> failures, and artifact locations.

## Base model

`Qwen/Qwen2.5-Coder-1.5B-Instruct` (Apache-2.0). Training requires a pinned model
commit in the run configuration; the repository template leaves it unset. The
smoke run's actual revision is preserved in its exported `identity.json` on Drive.

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

Preliminary smoke model (195 training examples, 3 epochs):

| Evaluation | Baseline | Adapter | Difference |
|---|---:|---:|---:|
| Fixed validation subset, 278 eligible questions / 14 DBs | 60.43% | 60.79% | +0.36 pp |
| Reconstructed training examples, 184 eligible questions / 85 DBs | 54.89% | 64.13% | +9.24 pp |

The validation paired database-bootstrap 95% interval for the difference is
−4.63 to +5.93 percentage points; improvement is not established. Training-set
accuracy is a memorization diagnostic, not evidence of generalization. Reference
adjudications remain unresolved and excluded, so these are provisional results.
Final test results have not been produced. See the [experiment log](experiments.md)
for the complete context and source locations.
