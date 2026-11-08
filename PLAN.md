# Fine-tune a small Qwen model for text-to-SQL with QLoRA

## Context
You want to learn fine-tuning end to end and show a credible before-and-after result. The model takes a **SQLite schema within its context budget plus a question** and writes SQL. Generalization is measured on held-out Spider databases; this does not establish support for arbitrary schemas or dialects. No performance measure was set, so this plan defines one.

Decisions so far: unseen SQLite schemas, free Colab GPU, learning/portfolio goal.

Constraints found:
- **Mac (M2, 8 GB RAM):** used for code and lightweight tests; model weights, datasets and GPU work stay in Colab/Drive. About 7 GiB was available during the November 8, 2026 review. Recheck before setup and keep at least 2 GiB free.
- **`xlangai/spider` on Hugging Face has questions and SQL only** (`db_id`, `question`, `query`, tokens). It has no schemas or databases, and we need both, so the Spider database bundle must be downloaded separately.

## Choices
| Item | Choice | Why |
|---|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Apache-2.0, trained for code, 3.1 GB, no thinking-mode handling |
| Optional comparison | `Qwen/Qwen3-0.6B` | Shows the size vs accuracy tradeoff; only after the main run works |
| Method | QLoRA: 4-bit NF4 + LoRA | Target a free T4; confirm fit at the selected context/batch size in the smoke run |
| Data | Spider (CC BY-SA 4.0) | Real SQLite databases, so queries can be executed, and its validation databases are separate from its training databases |
| Dialect | SQLite | Matches Spider; MySQL is out of scope |

Rejected for now:
- **`b-mc2/sql-create-context`, `gretelai/synthetic_text_to_sql`:** no real databases to execute against.
- **BIRD:** harder and a larger download; a possible follow-up.

## How performance is measured
**Primary metric: execution accuracy.** Run the predicted and correct SQL on the real database and check that they return the same rows. Different SQL that returns the right answer counts as correct.

Comparison rules (expanded into locked fixtures in `SPEC.md`):
- Compare column counts even for empty results. Columns compare by position; names are ignored. Unordered rows compare as multisets, retaining duplicates.
- A top-level gold `ORDER BY` requires sequence comparison. Audit gold ordering before scoring: tied keys are eligible only if permutations within ties cannot change the returned row sequence; otherwise exclude as ambiguous. `LIMIT`/`OFFSET` without deterministic selection is also excluded. Complex cases that cannot be established automatically require a documented integrity adjudication before model results are opened.
- Text is case-sensitive; NULL matches only NULL. Canonicalize finite numeric values to decimal values quantized to six decimal places with round-half-even, including integers (`1` equals `1.0`); preserve integer precision during conversion. This is a defined rounding metric, not a tolerance metric. Lock boundary, negative-zero, mixed-type and duplicate-row fixtures. Reject non-finite numeric results as unsupported.
- Accept exactly one read query. Use an isolated worker process, a read-only SQLite URI, extension loading disabled, and an authorizer allowlist for reads, SELECTs, approved functions and recursive SELECTs. Deny writes, `ATTACH`/`DETACH`, PRAGMAs and transaction/control statements.
- Enforce a 30-second wall-clock deadline by terminating the worker, plus an internal progress handler. Bound worker memory (initially 512 MiB), output to 100,000 rows and 16 MiB of serialized result values, including an individual-cell bound within that budget. Fetch incrementally; clean up connections/workers on every outcome. Prediction violations count as incorrect; gold violations follow the frozen exclusion policy.
- Store a structured outcome for parse errors, SQL errors, timeout, resource limit and successful execution. Cache gold outcomes keyed by database checksum, query, SQLite version, evaluator version and resource-limit configuration.

**Secondary metrics:**
- Share of predictions that execute without error
- Accuracy by Spider hardness level
- Inference latency
- Error categories from a manual review of about 50 failures

**Data splits, keeping databases separate to measure unseen-schema generalization:**
- **Train:** Spider train minus held-out databases
- **Dev:** about 10% of train databases, for tuning and choosing a checkpoint
- **Test:** Spider validation, run **once** at the end

**Success means:** the adapter beats the untouched base model on test, using the same prompt and inference settings, and a paired database-cluster bootstrap 95% confidence interval on the difference excludes 0. The primary score weights eligible questions equally. For each of 10,000 resamples, sample database IDs with replacement, include all questions from each sampled database (including repeated clusters), and recompute the paired difference. Use a fixed recorded seed and percentile interval. Also report macro accuracy across databases and the number of databases. This interval measures held-out-database sampling uncertainty, not variation across training seeds; the main run uses one seed. An inconclusive or negative result is still reported, with no test-driven retraining.

**Eligibility and test access:** freeze example IDs, database IDs and exclusion reasons before baseline/adapter comparison. Gold execution failures, unsupported ordering and gold resource violations are dataset exclusions, reported by split. Prompt/context overflow on dev or test counts as a model failure in the primary denominator for both models; an optional supported-context subset is reported separately. Drop overlength training examples and report counts. Never truncate schemas silently. Only train/dev token lengths may determine context settings.

Phase 1 may run a fixed automated test integrity job that validates database files, gold execution and ordering; only counts and exclusion records are exposed to training. Any required adjudication is limited to data validity, documented and frozen before scores are available. Test questions, predictions and model failure analysis remain closed until Phase 4; unavoidable gold inspection during integrity adjudication is recorded separately and cannot inform tuning. Do not use test lengths or failures to choose settings. If the evaluator is found defective after unblinding, invalidate and disclose the affected result; any corrected rerun is explicitly labeled, not described as a fresh untouched test.

## Implementation
Code is written and unit-tested on the Mac, tracked in local Git, and run in a thin Colab notebook using an approved remote or a source archive upload. Checkpoints and results go to Google Drive.

```
text-to-sql/
  src/schema.py     # SQLite -> CREATE TABLE text, with primary and foreign keys
  src/prompts.py    # one prompt template for training, baseline and evaluation
  src/data.py       # database-level splits, tokenizing, masking prompt tokens from labels
  src/evaluate.py   # safe execution, row comparison, metrics, bootstrap
  src/train.py      # QLoRA training, config-driven, resumable
  src/generate.py   # greedy generation, validate one SQL statement
  configs/*.yaml  tests/  notebooks/train.ipynb  README.md
```

### Reproducibility and input/output contract
- Commit `uv.lock` for Mac dependencies and a separate pinned Colab dependency set, including Python, PyTorch, CUDA compatibility, Transformers, PEFT and bitsandbytes. Record actual GPU/runtime versions. Pin model and tokenizer commit revisions, dataset revision, source bundle URL/checksums, split seed, code commit and run configuration.
- Serialize schemas deterministically: stable table/key order, original column order, escaped quoted identifiers, SQLite types, primary/foreign keys including composite keys. Test spaces, reserved words, quotes and unusual types. Freeze this serializer for baseline and adapter.
- Use the pinned tokenizer's chat template with one fixed system/user format. Training includes the assistant SQL and EOS; mask all system/user/template-prefix and padding tokens with `-100`. Keep EOS supervised even when its ID equals the padding ID. Verify boundaries by decoding real tokenized examples; do not assume tokenization is additive across message boundaries.
- Reserve `max_new_tokens=256` initially inside the total context budget; choose the total length from train/dev lengths and GPU measurements before test. Use greedy decoding (`do_sample=False`), fixed EOS/padding settings and identical precision/quantization for baseline and adapter. Truncated generations count as failures. Freeze any dev-driven change before final scoring.
- The output may be plain SQL or one enclosing SQL Markdown fence. Strip only that enclosing fence and surrounding whitespace; reject extra prose, extra fences or multiple statements. Never recover a first query by discarding the remainder. Parse in SQLite dialect and store raw output as well as extracted SQL.
- Save predictions incrementally by stable example ID, with SQL, outcome, token counts, latency and run identity. Define latency as synchronized model generation time excluding SQL execution, with warmup and batching recorded.

**Phase 0: evaluator first (Mac).** Build schema, prompt and evaluation code with pytest on tiny SQLite fixtures. Cover the comparison, ordering, extraction and resource rules above, including empty-result arity, ambiguous ties, rounding boundaries, huge cells, write attempts and runaway queries. Gold self-comparison is only a plumbing check; independently authored positive/negative fixtures establish comparator behavior. This needs no model or GPU.

**Phase 1: data and baseline (Colab).**
- Verify the database bundle and dataset revisions/checksums, match IDs, confirm split disjointness, and save database/example manifests.
- Run the frozen gold integrity job and freeze exclusions. Self-comparison should pass for every eligible gold query; investigate comparator defects rather than excluding them as data errors.
- Use train/dev token lengths and the real tokenizer to validate the contract and context budget. Save representative decoded inputs and label masks.
- Obtain hardness labels from pinned official Spider logic and compatible parsed SQL, checking ID alignment. If unavailable, omit the breakdown with an explanation rather than inventing labels.
- Score the base model on the fixed dev selection. Benchmark a complete dev evaluation, including generation, SQL execution and peak memory, before setting the run schedule.

**Phase 2: smoke test.** Start with a short run on about 200 examples, extending only within the recorded compute budget.
- Confirm falling loss and substantial training-sample improvement; record a small memorization check using actual generated SQL, not loss alone.
- Interrupt and resume in a fresh runtime with optimizer/scheduler/RNG/progress restored; compare against a short uninterrupted control within documented numerical tolerance.
- Measure peak GPU memory, time per step, checkpoint upload time and full dev evaluation time to finalize the runtime budget.

**Phase 3: full training.** Starting values, to be changed only on dev evidence:
- **LoRA:** `r=16`, `alpha=32`, dropout 0.05, targeting `q_proj,k_proj,v_proj,o_proj`
- **Training:** learning rate 2e-4 with cosine decay, batch size 4 with 4 accumulation steps, up to 3 epochs, gradient checkpointing, fixed seed
- **Checkpoint choice:** highest dev execution accuracy, not lowest loss
- **Tuning:** at most 3 configurations total, logged to a results table; include aborted attempts in the compute log.
- **Evaluation budget:** use a fixed seeded dev selection of up to 300 examples, retaining database coverage, for all checkpoint decisions. Evaluate at epoch boundaries, at most 3 times per configuration. After selection, score the chosen adapter and baseline on full dev once for reporting; this does not trigger reselection.
- **Compute ceiling:** initially 6 GPU-hours per session and 12 GPU-hours total across baseline, smoke tests, training and evaluation. Reserve the measured final-test cost before allocating tuning time. Reduce configurations/epochs using dev timing if necessary; record any revised ceiling before spending beyond it. If the measured minimum experiment cannot fit, stop and revise the compute plan before full training. Never silently overrun the budget.
- **OOM fallback:** reduce microbatch size 4 → 2 → 1 and increase accumulation to preserve effective batch size 16. If that fails, revise context length using train/dev evidence, rebuild training eligibility and rerun a comparable baseline; freeze settings before test.

Uses Transformers, PEFT and bitsandbytes with the plain `Trainer`, so each step stays visible for learning.

**Checkpoint contract:** save every 100 optimizer steps and at epoch boundaries. A resumable checkpoint includes adapter weights/config, optimizer and scheduler state, RNG states, scaler state where used, global step, epoch/data position and exact run/dependency identity. Inference exports additionally include tokenizer/chat-template and base-model revision. Write locally, copy to a unique Drive path, verify a checksum manifest, then write a completion marker; resume only completed compatible checkpoints. Keep the latest two complete resume checkpoints and the best-dev export, deleting older ones only after a verified replacement. Never overwrite the only valid checkpoint.

**Phase 4: final result.**
- Lock code, eligibility, configurations and adapter artifact hashes, then run baseline and adapter once on the same test IDs, with cluster-bootstrap confidence intervals. Interrupted jobs may continue only missing IDs under the identical frozen run identity.
- Review failures by category.
- Once supplied, recreate `docs/mysql-schema.sql` in SQLite, document dialect conversions and seed realistic fixture data. Prepare 10–15 questions with verified reference SQL/answers before viewing model outputs. This is a demo, not part of the score; omit it explicitly if the source schema remains unavailable.
- Write the README: method, results table, exclusions/denominators, compute log, limitations, reproduction steps and dataset attribution/license notices. Label this a custom execution metric; do not present it as directly comparable to published Spider scores. Record possible pretraining exposure to the public benchmark as a limitation.

## Risks
| Risk | Response |
|---|---|
| Spider database download is moved or rate-limited | Phase 1 tests it first; stop and reassess before any training |
| Colab disconnects or GPU quota runs out | Frequent checkpoints to Drive and resume support; split training across sessions |
| Evaluation is too slow | Benchmark the entire path, cache gold results, cap evaluations and reserve final-test compute; 300 predictions at a 30-second SQL timeout alone can take 2.5 hours |
| Some correct answers match by coincidence on small databases | Reported as a known limit, with hardness breakdown and manual review |
| Test results influence tuning | Test runs once, after all choices are locked |

Git excludes model weights, databases and tokens. Publishing to GitHub or Hugging Face only happens with your approval.

## Verification
1. `pytest` passes on the Mac without a GPU or model.
2. All eligible gold queries match themselves; independent comparator fixtures pass, and data exclusions/denominators are frozen and listed.
3. Baseline dev score is recorded before any training.
4. Real-tokenizer label masks and generated SQL pass smoke checks; interrupted training restores full state in a fresh session.
5. Reloading the saved adapter in a fresh Colab session reproduces its dev score, confirming the saved files are complete.
6. Final table reports baseline, adapter, paired difference, cluster-bootstrap interval and database counts, traceable to immutable manifests and saved predictions.

## What you need to do
- Run the notebook in your own Google account, with a GPU runtime and Drive access.
- Choose an approved remote repository or upload a source archive to Colab; creating/publishing a remote is a separate action.
