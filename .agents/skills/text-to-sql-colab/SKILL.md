---
name: text-to-sql-colab
description: Inspect, recover, and resume this text-to-sql repository's QLoRA training on Colab using its local controller. Use for training status, saved checkpoints, session loss, or resuming the recorded full-data run; not for unrelated Colab projects or new experiment design.
---

# Text-to-SQL Colab operations

Operate from this repository's root on the Mac. Prefer the existing controller
instead of reconstructing long inline commands. Resolve the root relative to
this skill (`../../..`), then verify `scripts/colab_training.py` and
`scripts/colab_remote.py` exist. Do not silently substitute a global checkout.

Read [the automation guide](../../../docs/colab-automation.md) for operational
limits or recovery details. Read [the experiment log](../../../docs/experiments.md)
when allocating compute or interpreting scores. These are repository resources,
not bundled copies; keep commands and logic in their existing locations.

## Select the requested action

A bare invocation means inspect status and explain the next action. It does not
by itself authorize a new GPU allocation or training. If the user has already
requested resume or authorized a specific session in this conversation, carry
out that work without asking for the same permission again. Creating this skill
or shipping its files does not start training.

Default identifiers for the recorded full-data experiment:

- Run: `qlora-r16-lr2e-4-full-e1`.
- Drive root: `/content/drive/MyDrive/text-to-sql/runs/`.
- GPU session: `sql-resume`; CPU inspection session: `sql-check`.
- Default invocation budget: two hours, including initialization. Optimizer-step
  completion and checkpoint saving can extend this deadline.

Prefer current user-selected names and live artifacts over these defaults. The
controller restores the recorded one-epoch configuration; a new model, epoch
count, batch size, or experiment is a separate task.

## Inspect

```bash
colab sessions
python scripts/colab_training.py status --session sql-resume
```

If the GPU session is absent, inspect saved Drive artifacts through an available,
mounted CPU session:

```bash
python scripts/colab_training.py status --session sql-check
```

If no session is available, explain the missing access. Creating even a CPU
inspection session requires the appropriate task authorization; Google Drive
consent remains interactive. Let the user complete the CLI's browser prompt;
do not save consent URLs, credentials, or tokens in repository files.

Distinguish these observations:

- `pids`: training processes in the inspected VM. CPU-session process information
  does not describe another VM.
- `saved_status`: last recorded heartbeat and step; `training` may be stale.
- `checkpoint_markers`: saved completion markers, not a fresh checksum audit.
- `final_marker`: an export marker; verify the export before inference.
- `log_tail`: recent progress and errors. Match the latest launch, not old
  appended logs. GPU utilization or a started PID alone does not prove progress.

Report the newest logged step separately from the latest saved checkpoint.
Do not infer completion from session disappearance or a low loss.

## Resume

Use the existing T4 session when available:

```bash
python scripts/colab_training.py resume --session sql-resume
```

When a replacement T4 session is authorized and needed, create exactly one:

```bash
python scripts/colab_training.py resume --session sql-resume --create
```

For a shorter authorized invocation, add `--session-hours 1`. The controller
accepts a positive budget of at most two hours. Check the experiment's remaining
compute allowance; the existing plan includes validation and final evaluation
in the total budget. Do not treat a per-invocation cap as a total experiment cap.

The controller mounts Drive, uploads source, restores saved training source and
configuration, installs recorded packages, checks run identity, verifies a
compatible checkpoint, and supervises training inside an executing Colab kernel cell. It skips an active same-VM run or
an existing final export. Checkpoints are selected dynamically: never hardcode
step 25 or 50 as the latest checkpoint.

After the `TEXT_TO_SQL_STARTED` event, inspect the streamed progress for a
matching resume message and increasing steps. The resume command remains
connected until the training child exits; do not launch a second resume command.
A status request against the same busy kernel may queue behind training. Report the selected checkpoint, PID, budget, and verification
limits. Do not claim launch acknowledgement means training completed.

## Failures and stopping conditions

- On `404/401` or missing acknowledgement, inspect sessions and Drive state.
  Never automatically create replacement GPU sessions or retry an uncertain
  launch. Stop when the requested invocation ends or the session is lost.
- On identity mismatch, restore the recorded environment/source or explain the
  exact discrepancy. Never edit identity files or remove checks to force resume.
- Without a verified compatible checkpoint, do not silently start fresh.
- If a newer checkpoint fails checksums and the loader selects an older one,
  disclose the fallback and repeated work.
- If an allocation warning is followed by increasing steps, describe that
  evidence without claiming every later warning recovered. A fatal OOM or dead
  process needs diagnosis; changing batch settings requires a compatible new
  plan, not an unrecorded alteration to the current run.
- `paused` means unfinished training with a budget checkpoint; `complete` means
  the configured target finished. Validate the final artifact before evaluation.

Keep the Mac awake and the CLI connected during validation of supervised
execution. Keeping the actual training cell executing addresses the suspected
idle-kernel problem but does not guarantee Colab VM lifetime. The controller
does not send artificial keep-alive pings or release the VM on pause. It cannot
guarantee execution or monitoring after the Mac disconnects.
Each attempt records launch metadata under `invocations/` and records its exit
code if the VM remains available to observe process termination. Missing exit
metadata does not identify a specific backend failure.

## Changes and validation

For controller fixes, run the offline tests without allocating a GPU:

```bash
uv run --frozen pytest -q tests/unit/test_colab_training.py
```

The notebook at `notebooks/colab-training-runbook.ipynb` is a historical command
walkthrough; its external-operation switches default to off. The controller is
the maintained command entrypoint. A local test pass does not establish GPU
correctness. Do not run final-test evaluation, alter the experiment protocol,
commit, push, or merge merely because this operations skill was invoked.
