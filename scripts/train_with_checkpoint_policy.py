"""Apply an operational checkpoint interval to the recorded training source.

Keep the saved config/identity intact. Trainer restores save_steps from its state,
so set both arguments and state in on_train_begin, after checkpoint restoration.
"""
from __future__ import annotations

import argparse
import sys


def checkpoint_callback(save_steps: int):
    from transformers import TrainerCallback

    class CheckpointPolicy(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            args.save_steps = save_steps
            state.save_steps = save_steps
            print(f"Runtime checkpoint policy: every {save_steps} optimizer steps; "
                  f"next periodic save at {(state.global_step // save_steps + 1) * save_steps}",
                  flush=True)
            return control

    return CheckpointPolicy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--save-steps", type=int, default=20)
    policy, training_args = parser.parse_known_args()
    if policy.save_steps <= 0:
        parser.error("--save-steps must be positive")

    from text_to_sql.model import train

    original_callbacks = train._make_callbacks
    original_argv = sys.argv

    def callbacks(*args, **kwargs):
        return [*original_callbacks(*args, **kwargs), checkpoint_callback(policy.save_steps)]

    train._make_callbacks = callbacks
    sys.argv = [original_argv[0], *training_args]
    try:
        train.main()
    finally:
        train._make_callbacks = original_callbacks
        sys.argv = original_argv


if __name__ == "__main__":
    main()
