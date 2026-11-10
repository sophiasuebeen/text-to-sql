# Data card

## Source

**Spider** (Yu et al., 2018): English questions paired with SQL over real SQLite
databases. License: CC BY-SA 4.0. Derived artifacts must keep attribution and the
same license.

The Hugging Face copy (`xlangai/spider`) contains questions and SQL only. The
SQLite databases come from the separate Spider database bundle. Both are pinned
(revision and checksum) in `configs/data.yaml` before any real run;
`scripts/validate_reference_queries.py` refuses to run while they are unpinned.

## Splits

| Split | Source | Use |
|---|---|---|
| train | Spider train, minus held-out databases | QLoRA training |
| dev | ~10% of Spider train *databases* (seeded) | checkpoint choice, settings |
| test | Spider validation | final comparison, run once |

No database appears in more than one split (`data/splits.check_disjoint`).
Example IDs are stable positions in the pinned source files.

## Filtering

- Training drops examples whose reference SQL does not parse or execute, and examples
  longer than the context budget (dropped and counted, never truncated).
- Evaluation exclusions follow the reference query validation rules in `docs/evaluation.md`.

## Not stored in Git

Databases, model weights, checkpoints and tokens are ignored by Git and kept on
Google Drive.

## Known limitations

- Spider is public, so the base model may have seen it during pretraining.
- Small databases can make a wrong query return the right rows by coincidence.
