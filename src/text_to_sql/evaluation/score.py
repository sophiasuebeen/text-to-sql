"""Score one model prediction against an audited gold record."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..sql.executor import OK, Limits, execute_sql
from ..sql.policy import parse_single_read
from .compare import rows_match
from .gold import GoldRecord


@dataclass(frozen=True)
class Score:
    correct: bool
    executed: bool
    pred_status: str
    error: str | None = None


def score_prediction(db_path: str | Path, pred_sql: str | None, gold: GoldRecord, limits: Limits = Limits()) -> Score:
    """Score one already-extracted prediction against an eligible gold record.

    ``pred_sql=None`` means extraction failed or the prompt overflowed: a failure
    that stays in the denominator.
    """
    if not gold.eligible:
        raise ValueError("ineligible gold examples are excluded before scoring")
    if pred_sql is None:
        return Score(False, False, "no_sql")
    tree, err = parse_single_read(pred_sql)
    if err:
        return Score(False, False, "parse_error", err)
    out = execute_sql(db_path, pred_sql, limits)
    if out.status != OK:
        return Score(False, False, out.status, out.error)
    g = gold.outcome
    return Score(rows_match(out.rows, out.ncols, g.rows, g.ncols, ordered=gold.ordered), True, OK)
