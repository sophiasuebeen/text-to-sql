# Colab training automation

Run this controller from the Mac checkout using the authenticated Colab CLI.
It automates the recovery procedure for `qlora-r16-lr2e-4-full-e1`: Drive mounting,
project upload, saved training source/configuration restoration, pinned package
installation, identity/checksum verification, and supervised training execution inside an active Colab kernel cell.

For an existing T4 session:

```bash
python scripts/colab_training.py resume --session sql-resume
```

If that session has disappeared, explicitly create one replacement:

```bash
python scripts/colab_training.py resume --session sql-resume --create
```

Drive authorization may require opening the CLI-provided URL in a browser.
Creation is attempted once; session loss never triggers automatic allocation.
Reassess the remaining experiment budget before creating or resuming a session.
The controller defaults to a two-hour invocation budget, including initialization;
finishing an optimizer step and saving may extend the deadline. To shorten it:

```bash
python scripts/colab_training.py resume --session sql-resume --session-hours 1
```

Inspect the live process list, saved state, checkpoint completion markers, and
last 20 log lines without starting or installing anything:

```bash
python scripts/colab_training.py status --session sql-resume
```

If the GPU session disappears, inspect Drive from the mounted CPU session:

```bash
python scripts/colab_training.py status --session sql-check
```

The controller skips setup and launch when training is already active in that VM
or a final export exists. A same-VM file lock protects setup and is inherited by
the supervised training child. This does not prevent concurrent runs from another
VM or an unrelated launcher. A failed remote acknowledgement is treated as an
error even when the CLI itself exits successfully. The controller never retries
an uncertain launch automatically.

A `TEXT_TO_SQL_STARTED` event confirms process launch, not successful optimizer
steps. The command remains attached, streams progress every 30 seconds, and keeps
the Colab cell executing until the training child exits. Inspect its resume
message and increasing steps. The remote timeout includes an additional hour
for setup and checkpoint saving beyond the training invocation budget.
A second status command targeting the same busy kernel can queue behind training;
read the progress stream instead. A saved `training` state may
be stale after a crash. Status only lists checkpoint completion markers; resume
verifies the newest compatible checkpoint's checksums before launching. If the
newest checkpoint is corrupt, the existing loader can fall back to an older one;
the selected path is printed. It refuses to start fresh without a valid checkpoint.

The script restores the saved `train_source.py`, not the complete historical
source tree; other modules come from the current checkout. Accelerate 1.15.0
comes from the recorded launch environment; other model dependency versions come
from the saved identity. The identity check does not cover every transitive
package or independently prove unchanged training inputs. Preserve a complete
source snapshot and dependency lock for future runs.

Colab still controls VM lifetime. Keep the Mac awake and the CLI connected during
validation of this supervised path. It does not guarantee execution after the
Mac disconnects. The controller does not send artificial keep-alive pings,
automatically shut down VMs, or automatically evaluate final test data.
The former detached launcher returned immediately and left the Jupyter kernel
idle despite GPU work; this is a plausible explanation for the observed session
losses, not a confirmed backend termination reason. Google documents that kernel
activity affects VM liveness: [CLI session management](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md). Release idle GPU resources
separately after confirming a completed checkpoint.

Validation is offline: CLI argument parsing, remote result handling, prevention
of duplicate launches, status reporting, and lost-session handling. No GPU
session, package installation, or real training is launched by the local tests.

Each supervised attempt writes `invocations/<timestamp>-<pid>/launch.json` and,
if the VM survives long enough to observe the child exit, `exit.json` with the
return code and last recorded training state. A missing exit record is evidence
of an unobserved end, not proof of a particular failure cause. Normal exit must
report `paused` or `complete`; a child error is reported as a failed command.

## Save every 20 optimizer steps

The full-run local template now uses `save_steps: 20`. Existing runs retain their
saved `train_config.yaml` and identity, including the original 25-step interval.
The controller launches `scripts/train_with_checkpoint_policy.py --save-steps 20`
to apply an operational override without changing those historical artifacts.
Its callback updates both `TrainingArguments.save_steps` and the restored
`TrainerState.save_steps` at training start. Original checkpoint publishing,
checksum verification, retention, epoch-end saving, and budget callbacks remain
in place. Invocation metadata records `checkpoint_save_steps: 20`.

Regular saves use global-step multiples: resuming checkpoint-50 next saves at
60, then 80, 100, and so on. Saving more often reduces repeated work after a
session loss; it does not prevent Colab from reclaiming the VM. This policy is
applied on the next controller resume; editing local files does not change a
running or terminated VM.

If Drive was mounted manually and `status` successfully reads the saved run,
add `--skip-mount` to resume without repeating the CLI Drive authorization path:

```bash
python scripts/colab_training.py resume --session sql-resume --skip-mount
```

This option still checks the run directory, existing processes, and final export
before setup. It does not bypass identity or checkpoint verification and fails
if Drive is unavailable.
