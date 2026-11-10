# Evaluation

**Primary metric: execution accuracy.** The predicted SQL and the reference SQL run on
the real database; a prediction is correct when both return the same result.
Different SQL that returns the right answer counts as correct.

This is a custom metric. It is not directly comparable to published Spider scores.

## Comparison rules (`evaluation/compare.py`)

| Rule | Behavior |
|---|---|
| Columns | Compared by position; names ignored; column counts must match, even for empty results |
| Row order | Multiset comparison (duplicates kept), unless the audited reference has a top-level `ORDER BY` |
| Numbers | Exact decimal of the stored value, rounded to 6 places, round-half-even; `1` equals `1.0`; `-0` equals `0`; integers never pass through float |
| Text | Case-sensitive |
| NULL | Equals only NULL |
| Types | Typed values, so text `"1"` differs from number `1`; mixed-type columns never need sorting |
| Non-finite | NaN and infinity are unsupported; the outcome is recorded as such |

## Reference query validation (`evaluation/reference.py`)

Every reference query is audited before any model is scored. Exclusions are data-validity
decisions only and are frozen before results are viewed.

- **Ordered reference:** the audit re-runs the query with its `ORDER BY` keys projected and
  `LIMIT`/`OFFSET` removed. If tied rows inside the returned window differ, a
  different tie order would change the answer → `ambiguous_order`.
- **`LIMIT` without `ORDER BY`:** excluded as `unordered_limit` unless the limit cuts nothing.
- **Unresolvable cases** (e.g. `DISTINCT` ordered by a non-output column) →
  `needs_adjudication`, decided manually in `configs/adjudications.yaml`.
- **Reference execution failures** → `reference_<status>`.
- **Self-comparison failure** → a comparator defect to fix, never a data exclusion.

Known limit: ties under a non-binary collation (e.g. `NOCASE`) are not detected.

Run `python scripts/validate_reference_queries.py --config configs/data.yaml` to
build the manifests and validate the reference queries. New cache records use
`reference_cache.jsonl` and exclusion reasons use the `reference_` prefix.
Existing `gold_cache.jsonl` records and `gold_` exclusion reasons are still read
for compatibility, so completed audits do not need to be repeated after the rename.

## Prediction outcomes

Each prediction gets one structured outcome: `ok`, `parse_error`, `sql_error`,
`disallowed`, `timeout`, `resource_limit`, `unsupported_value` or `worker_error`.
Anything other than `ok` with matching rows is incorrect. A prompt too long for
the context window is also a failure and stays in the denominator for both models.

## Execution limits (`sql/executor.Limits`)

30 s wall clock, 512 MiB SQLite heap, 100,000 rows, 16 MiB of result values,
1 MiB per value.

## Statistics (`evaluation/metrics.py`)

- Question-weighted accuracy (primary) and macro accuracy across databases.
- **Paired database-cluster bootstrap** for adapter − baseline: resample databases
  with replacement (10,000 resamples, fixed seed), take every question of each
  sampled database, and report the 95% percentile interval. This measures
  uncertainty from which databases were held out, not variation across training seeds.

## Splits

Databases, not questions, are split, so test schemas are never seen in training:
train and dev come from Spider's training databases (about 10% of databases held
out as dev), and Spider's validation set is the test set, scored once at the end.

## Try a saved adapter

Run on the Colab GPU (these are remote shell commands, not Mac commands):

```bash
python -u -m text_to_sql.model.predict \
  --adapter /content/drive/MyDrive/text-to-sql/runs/qlora-r16-lr2e-4-t4-b2-smoke/final \
  --db /path/to/example.sqlite \
  --question 'How many people are at least 30 years old?' \
  --mode both --execute
```

The model needs both the actual database schema and a question. `--mode both`
prints baseline and adapter predictions; `--mode adapter` uses just the trained
adapter. `--execute` is optional and uses the existing read-only SQL sandbox.
Execution previews contain at most 20 rows in the evaluator's typed canonical
format. Generating executable SQL does not by itself establish correctness.

The saved `final/` directory contains LoRA adapter weights and the tokenizer,
not a standalone copy of the base model. The loader verifies its completion
marker/checksums, loads the recorded base-model commit in 4-bit NF4, then attaches
the adapter. Baseline inference disables that adapter on the same model. Both
conditions use the same tokenizer, schema, prompt, precision, greedy decoding,
and context/output budgets. The default total budget is 2048 tokens, including
up to 256 generated tokens. Overflow and truncated output count as failures.

## Compare baseline and adapter on dev

```bash
python -u -m text_to_sql.evaluation.compare_dev \
  --adapter /content/drive/MyDrive/text-to-sql/runs/qlora-r16-lr2e-4-t4-b2-smoke/final \
  --data-config configs/data.yaml \
  --max-examples 20 \
  --output /content/drive/MyDrive/text-to-sql/artifacts/smoke-dev-first20
```

This is a provisional smoke diagnostic, not a final test result. It selects the
first 20 eligible IDs in the frozen dev subset (not a representative sample),
never test questions. Remove `--max-examples 20` to use all eligible examples in
that same frozen subset. Unresolved reference decisions remain excluded and are
recorded alongside the results; finalize the data-validity policy before official
scoring. This preliminary run consumes dev evaluation compute and should be
included in the experiment budget.

Each condition is scored by SQL execution results, not SQL string equality.
The denominator is identical, including generation/execution failures and prompt
overflows. Saved artifacts are `run.json` (identity, selected IDs, input hashes,
decoding policy), `exclusions.jsonl`, incremental `predictions.jsonl`, and
`summary.json` (execution accuracy, macro accuracy, paired database bootstrap,
elapsed time, and peak GPU memory). The baseline here is the same 4-bit model,
not an FP16 baseline. Small-subset intervals are diagnostic only.

Use a new output directory per run. To continue an interrupted run, use the same
`--output` directory and add `--resume`. Model identity, input hashes, selected IDs,
and generation settings must match. Completed prediction pairs are re-scored with
the current evaluator without regenerating SQL; only remaining pairs are generated.
The saved log must be an intact ordered prefix; corrupt or partial JSON lines are
rejected without silently dropping examples. Summary timing covers the current
invocation and `resumed_examples` records how much generation was reused.
Unclosed quotes and other SQL tokenization failures count as incorrect predictions,
not fatal evaluation errors. GPU inference must be
verified on Colab; Mac unit tests cover orchestration using a model test double.

## Check learning on the smoke training examples

Use the same comparison command with
`--train-smoke-config configs/train-smoke-t4.yaml` and a new output directory.
This reconstructs the original seeded 200-example selection and re-applies the
training tokenizer, invalid-reference filtering, and full-sequence length filter.
The config hash must match the saved adapter, and the reconstructed filtering
report must match the saved training report. The original run did not save exact
per-example input hashes, so this is a reconstruction from pinned data/current
artifacts, not an independent proof that training inputs were unchanged.

Only reconstructed training examples eligible for deterministic execution scoring
are scored. Some ordering-ambiguous queries were valid training supervision but
cannot be used for this metric; the output records trained IDs and counts before
and after scoring exclusions. Generation overflows stay in the scoring denominator.
This is a training memorization diagnostic, not dev/test generalization accuracy.
Do not pass `--max-examples` in this mode. Results use the same paired predictions
format and support `--resume`; the dev evaluation remains unchanged.
