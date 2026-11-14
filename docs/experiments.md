# Experiment log

Keep code, configuration, and compact result summaries in GitHub. Keep datasets,
model weights, checkpoints, and complete prediction logs in Google Drive. The
numbers below were transcribed from user-provided Colab logs and evaluation
summaries; the original artifact files have not yet been downloaded locally.

## 2026-11-10 — T4 smoke experiment

Goal: validate the QLoRA training, checkpoint export, inference, and execution
accuracy pipeline, then compare changes on seen training examples and validation
questions from unseen databases.

### Run configuration

| Setting | Value |
|---|---|
| Run | `qlora-r16-lr2e-4-t4-b2-smoke` |
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| GPU | Tesla T4 |
| Method | QLoRA; 4-bit NF4, double quantization, fp16 compute |
| LoRA | r=16, alpha=32, dropout=0.05; q/k/v/o projections |
| Trainable parameters | 4,358,144 of 1,548,072,448 total |
| Seed | 42 |
| Learning rate | 2e-4, cosine schedule, 0.03 warmup ratio |
| Batch | microbatch 2 × gradient accumulation 8 = effective batch 16 |
| Epochs / optimizer steps | 3 / 39 |
| Training sequence limit | 2,048 tokens |
| Training data | Seeded selection of 200 examples; 5 overlength examples dropped; 195 retained |
| Longest retained sequence | 1,837 tokens |
| Training configuration hash | `0dcabc4a728d8c7c`; matched against the saved adapter during reconstruction |
| Training time | About 35 minutes (`session_seconds=2107`) |
| Peak training GPU memory | 8.86 GiB |
| Mean training loss | 0.2196 |
| Export state | `complete`; final export directory present |

The actual run used `train-smoke-t4.yaml` on Colab/Drive. The repository's
`configs/train.yaml` is a starting template, not a complete copy of that run's
configuration. The exact base-model commit and dependency versions are recorded
in the final export's `identity.json`; they have not yet been transcribed here.

Dataset: `xlangai/spider`, revision
`0c350918f3f29ec754f1181c65cdce76cd6c133c`.
Database bundle SHA256:
`00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b`.

### Execution accuracy

Both conditions use the same base-model revision, 4-bit quantization, prompt, and
tokenizer. The baseline disables the adapter; the trained condition enables it.
Decoding is greedy with up to 256 generated tokens inside a total 2,048-token
budget. Correctness means matching the reference SQL's execution result, not its
text. Generation failures, execution errors, and context overflows count as wrong.

| Evaluation | Questions / DBs | Baseline | Adapter | Difference |
|---|---:|---:|---:|---:|
| Preliminary validation check | 20 / 1 | 17/20 (85.00%) | 17/20 (85.00%) | 0.00 pp |
| Eligible examples in the fixed validation subset | 278 / 14 | 168/278 (60.43%) | 169/278 (60.79%) | +0.36 pp |
| Eligible reconstructed training examples | 184 / 85 | 101/184 (54.89%) | 118/184 (64.13%) | +9.24 pp |

- Validation paired database-bootstrap 95% interval: **−4.63 to +5.93 percentage
  points**. An improvement is not established.
- Training-example interval: **+1.06 to +17.39 percentage points**. This is a
  diagnostic on seen examples, not evidence of generalization or final test performance.
- The preliminary 20 questions overlap the larger validation run. They are not an
  independent experiment and must not be pooled with it. With only one database,
  that run's bootstrap interval is not useful for uncertainty assessment.
- Training examples were reconstructed with the same configuration, seed, and
  saved tokenizer, and the filtering report matched the saved run. The original
  training run did not save per-example IDs or input hashes, so this is not
  independent proof that every training input was unchanged.
- Of 195 training examples, 11 were excluded from execution scoring under the
  reference-validity rules, including ambiguous ordering.
- The automated reference audit left 26 adjudications unresolved and excluded.
  These results remain provisional; finalize the data-validity policy before
  official scoring.
- No model generation or performance evaluation on final test questions has run.

### Validation changes

| Outcome | Questions |
|---|---:|
| Both correct | 146 |
| Baseline wrong → adapter correct | 23 |
| Baseline correct → adapter wrong | 22 |
| Both wrong | 87 |

Observed examples:

- Improved schema grounding: using the actual `apt_number` column instead of the
  nonexistent `apartment_number`.
- A regression where an unnecessary JOIN changed a simple aggregate's result.
- An incorrect table alias for a name column, causing an execution error.
- Failure to map the question's `LA` to the database value `Los Angeles`.

These are observations from selected examples, not a frequency analysis of all failures.

### Failures and fixes

1. `TrainingArguments` rejected `warmup_ratio` before training started. The code
   now checks the supported arguments and passes the ratio through `warmup_steps`
   when required by the installed Transformers API.
2. The initial microbatch 4 × accumulation 4 run hit CUDA OOM after one optimizer
   step. A new run using microbatch 2 × accumulation 8 completed.
3. The local display stopped showing progress at 85%, but remote training
   completed. The saved log and completion status confirmed this.
4. Validation stopped after 74 completed questions when an unclosed SQL quote
   raised `TokenError`. The parser now rejects tokenization failures as invalid
   SQL. The evaluation resumed, reusing and rescoring those 74 prediction pairs,
   and completed all 278 questions. Invalid SQL remains in the denominator.

The resumed validation invocation took 1,320.32 seconds, excluding the earlier
invocation. Training-example comparison took 1,181.99 seconds. Total GPU usage,
including failed attempts, interruptions, and preparation, has not yet been
compiled; these measurements are not the complete compute cost.

### Artifact locations

Paths below assume Google Drive is mounted in Colab:

```text
/content/drive/MyDrive/text-to-sql/
  configs/train-smoke-t4.yaml
  artifacts/phase1/
  artifacts/smoke-dev20-bf500e3c/
  artifacts/smoke-dev-subset-675bcab1/
  artifacts/smoke-train-check-a1dc122b/
  runs/qlora-r16-lr2e-4-t4-b2-smoke/
    identity.json
    data_report.json
    status.json
    final/
```

Evaluation directories contain `summary.json`, `run.json`, `predictions.jsonl`,
and `exclusions.jsonl`. The inference loader verifies the final export's completion
marker and checksum manifest. Model weights and full databases are not committed.

### Interpretation and next experiment

Execution accuracy improved on reconstructed training examples, but improvement
on validation databases was not established. This alone does not diagnose
overfitting or isolate a training-configuration problem. The two evaluation sets
have different difficulty, so their absolute accuracies are not directly comparable.

The next candidate experiment is a first full-data training run, evaluated against
the same validation policy. Before running it, preserve the actual Colab configuration
and environment versions locally, account for the remaining compute budget, and
finalize data-validity decisions. Improved accuracy from full-data training remains
an untested hypothesis. Keep the test evaluation for the final selected setup.

## 2026-11-10 — First full-data run (launch record)

Run: `qlora-r16-lr2e-4-full-e1`, configured in
`configs/train-full-t4-e1.yaml` (the local template now uses a 20-step saving
interval; the saved launch configuration used 25). Started at 14:48:50 UTC (23:48:50 KST) on the
existing T4 runtime. This is a fresh adapter initialized from the pinned base
model, not continued training of the smoke adapter.

- Full training preflight: 6,116 candidates; 3 invalid references and 82
  overlength examples dropped; **6,031 retained**, longest sequence 1,887 tokens.
- Target: **one epoch**, approximately 377 optimizer steps. Keep the established
  2e-4 learning rate, LoRA settings, 2,048-token context, and effective batch 16
  (microbatch 2 × accumulation 8).
- Base-model revision: `2e1fd397ee46e1388853d2af2c993145b0f1098a`, matching the
  smoke export. Baseline/validation settings remain unchanged.
- Extrapolating the smoke run gives about **5.66 hours per epoch** or roughly
  17 hours for three epochs. This is a rough estimate, not a measured full-run
  duration. The existing runtime was already 2.96 hours old at launch.
- Compute plan: reduce the initial target to one epoch and run this invocation
  with `--session-hours 2`, preserving the existing six-hour session ceiling and
  twelve-hour experiment ceiling. Reassess measured speed and remaining compute
  before another invocation; reserve time for validation and final evaluation.
- At the session deadline, finish the current optimizer step, request a full
  checkpoint, verify its Drive copy, and report `state=paused`. A paused run does
  not export or claim a completed final model. The next invocation with the same
  configuration resumes its optimizer, scheduler, RNG, and data position. Saving
  and finishing the current step can extend the invocation beyond two hours.
- Save regular checkpoints every 25 steps and log every 5 steps. Run detached
  from the local terminal, writing `training.log` directly to Drive. Checkpoint
  copies survive a VM disconnect; the local terminal need not remain connected.

The run directory on Drive contains the actual configuration, a training-source
snapshot, launch metadata, and a preflight per-example supervised-feature hash
manifest. This preflight snapshot is separate from the original smoke run's
reconstruction. No final accuracy result is available for the full-data run yet.

Observed environment at launch: PyTorch `2.14.1+cu130`, Transformers `5.22.1`,
PEFT `0.21.2`, bitsandbytes `0.50.2`, Accelerate `1.15.0`.

Artifacts:
`/content/drive/MyDrive/text-to-sql/runs/qlora-r16-lr2e-4-full-e1/`.

Local verification for the session-budget change: **160 tests passed**. The
deadline callback requests saving and stopping only for incomplete training;
finished training retains the normal completion path. Actual budget-triggered
checkpoint/resume behavior in this GPU run remains to be observed.


## 2026-11-14 — Full-data training completed

The same `qlora-r16-lr2e-4-full-e1` run completed **377/377 optimizer steps** and
one epoch on **6,031 retained examples**. The final status reported `complete`
at `2026-11-14T13:56:13+00:00` (22:56:13 KST), with process exit code 0.
A subsequent read confirmed `final/COMPLETE.json` and resume completion markers
for checkpoints 360 and 377. The export pipeline verifies checksums before
publishing the completion marker; a fresh-runtime inference check remains pending.

Operational changes and observations:

- Original interrupted attempts left checkpoints at 25, 50, and later steps;
  25/50 were saving intervals, not training step limits. The backend termination
  cause was not established. Keeping an actual training cell executing did not
  guarantee session survival.
- After repeated interruptions, saving was changed from 25 to **20 optimizer
  steps** through a callback that sets both Trainer arguments and restored state.
  The saved training configuration and identity were preserved. Periodic saves
  follow global-step multiples; epoch-end and budget saves are additional.
- A supervised invocation resumed from checkpoint-220 and paused normally at
  **checkpoint-360** on its two-hour budget. It reported `state=paused`,
  `reason=session_time_budget`, exit code 0, and `session_seconds=7224`.
- The final invocation resumed checkpoint-360 with a one-hour budget and completed
  at 377. It reported `session_seconds=907` and peak allocated GPU memory of
  **8.90 GiB for that invocation**. These timings are not the total training cost.
- CUDA allocation warnings were followed by progress in multiple attempts. The
  logs do not establish that those warnings caused VM losses.

The first full-data adapter has **not been evaluated on dev or final test**.
Smoke scores above do not describe this adapter. Unresolved reference-validity
adjudications and total compute accounting remain outstanding before official
scoring and further experiment allocation. Total GPU time must include failed
attempts, initialization, evaluation, and interrupted sessions; it is not inferred
from the last invocation's runtime.
