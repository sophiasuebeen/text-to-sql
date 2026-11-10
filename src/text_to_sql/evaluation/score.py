"""Score one model prediction against an audited reference record."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..sql.executor import OK, Limits, execute_sql
from ..sql.policy import parse_single_read
from .compare import rows_match
from .reference import ReferenceRecord


@dataclass(frozen=True)
class Score:
    correct: bool
    executed: bool
    pred_status: str
    error: str | None = None


def score_prediction(db_path: str | Path, pred_sql: str | None, reference: ReferenceRecord, limits: Limits = Limits()) -> Score:
    """Score one already-extracted prediction against an eligible reference record.

    ``pred_sql=None`` means extraction failed or the prompt overflowed: a failure
    that stays in the denominator.
    """
    if not reference.eligible:
        raise ValueError("ineligible reference examples are excluded before scoring")
    if pred_sql is None:
        return Score(False, False, "no_sql")
    tree, err = parse_single_read(pred_sql)
    if err:
        return Score(False, False, "parse_error", err)
    out = execute_sql(db_path, pred_sql, limits)
    if out.status != OK:
        return Score(False, False, out.status, out.error)
    expected = reference.outcome
    return Score(rows_match(out.rows, out.ncols, expected.rows, expected.ncols, ordered=reference.ordered), True, OK)
