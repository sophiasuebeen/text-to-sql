# Architecture

This project fine-tunes a small code model to translate English questions into SQLite
queries, and measures the improvement with an execution-based grader.

## Where things run

| Environment | Runs | Never runs |
|---|---|---|
| Mac / CI | schema, prompts, SQL policy, executor, grading, data splits, config and checkpoint logic | torch, transformers, peft, bitsandbytes |
| Colab GPU | model loading, QLoRA training, generation | — |

ML libraries are imported inside functions only, and CI fails if they are installed.

## Package layout

```
src/text_to_sql/
  io.py                  checksums and JSONL helpers
  model/prompts.py       the single prompt used for training, baseline and evaluation
  model/train.py         QLoRA training, config-driven and resumable
  model/checkpoints.py   verified checkpoint publishing, resume selection, pruning
  sql/policy.py          what SQL is allowed: output extraction, parsing, runtime authorizer
  sql/executor.py        sandboxed execution in a resource-bounded worker process
  evaluation/compare.py  canonical values and result comparison
  evaluation/reference.py     reference-query integrity audit and cache
  evaluation/score.py    score one prediction against audited reference
  evaluation/metrics.py  accuracy and the paired database-cluster bootstrap
  data/schema.py         SQLite file -> deterministic CREATE TABLE text
  data/splits.py         Spider loading, database-level splits, disjointness and manifests
  data/prepare.py        tokenization and label masking for training
scripts/validate_reference_queries.py  builds manifests and audits every reference query
```

## Path of one example

```
SQLite database ──data/schema──────▶ schema text ─┐
question ──────────────────────────────────────────┴─model/prompts──▶ chat messages
                                                                          │
                                                       model (base or base + adapter)
                                                                          │
raw output ──sql/policy.extract_sql──▶ one SQL statement (or rejection)
           ──sql/executor.execute_sql──▶ canonical rows (or timeout/denied/error)
           ──evaluation/score──▶ correct? vs audited reference ──evaluation/metrics──▶ report
```

## Defense in depth for untrusted SQL

1. **Static:** exactly one SQLite read statement (`sql/policy.py`, sqlglot).
2. **Runtime:** read-only connection, extension loading off, authorizer allowing only
   reads and approved functions (`sql/policy.authorizer`).
3. **Resources:** separate process per query with wall-clock kill, SQLite heap and
   value-length limits, row and byte budgets (`sql/executor.py`).
