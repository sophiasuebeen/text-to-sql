# Evaluation

**Primary metric: execution accuracy.** The predicted SQL and the gold SQL run on
the real database; a prediction is correct when both return the same result.
Different SQL that returns the right answer counts as correct.

This is a custom metric. It is not directly comparable to published Spider scores.

## Comparison rules (`evaluation/compare.py`)

| Rule | Behavior |
|---|---|
| Columns | Compared by position; names ignored; column counts must match, even for empty results |
| Row order | Multiset comparison (duplicates kept), unless the audited gold has a top-level `ORDER BY` |
| Numbers | Exact decimal of the stored value, rounded to 6 places, round-half-even; `1` equals `1.0`; `-0` equals `0`; integers never pass through float |
| Text | Case-sensitive |
| NULL | Equals only NULL |
| Types | Typed values, so text `"1"` differs from number `1`; mixed-type columns never need sorting |
| Non-finite | NaN and infinity are unsupported; the outcome is recorded as such |

## Gold integrity audit (`evaluation/gold.py`)

Every gold query is audited before any model is scored. Exclusions are data-validity
decisions only and are frozen before results are viewed.

- **Ordered gold:** the audit re-runs the query with its `ORDER BY` keys projected and
  `LIMIT`/`OFFSET` removed. If tied rows inside the returned window differ, a
  different tie order would change the answer → `ambiguous_order`.
- **`LIMIT` without `ORDER BY`:** excluded as `unordered_limit` unless the limit cuts nothing.
- **Unresolvable cases** (e.g. `DISTINCT` ordered by a non-output column) →
  `needs_adjudication`, decided manually in `configs/adjudications.yaml`.
- **Gold execution failures** → `gold_<status>`.
- **Self-comparison failure** → a comparator defect to fix, never a data exclusion.

Known limit: ties under a non-binary collation (e.g. `NOCASE`) are not detected.

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
