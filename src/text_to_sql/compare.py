"""Canonical values and result-set comparison for execution accuracy.

Each SQLite value becomes a hashable, typed 2-tuple so mixed-type rows can be
compared and counted without ever sorting incompatible Python values:

    NULL            -> ("null", None)
    INTEGER / REAL  -> ("num",  "<decimal quantized to 6 places>")
    TEXT            -> ("text", <str, case-sensitive>)
    BLOB            -> ("blob", <hex>)

Comparison: column arity must match (even for empty results); columns compare by
position (names ignored); unordered results compare as multisets (duplicates
kept); ordered results (audited gold top-level ORDER BY) compare as sequences.
"""

from __future__ import annotations

import decimal
from collections import Counter
from decimal import Decimal
from typing import Any, Sequence

CanonicalValue = tuple[str, Any]
CanonicalRow = tuple[CanonicalValue, ...]

QUANTUM = Decimal("0.000001")
# Wide enough for any 64-bit integer plus 6 fractional digits, and for the exact
# binary expansion of any finite double before rounding.
_CTX = decimal.Context(prec=400, rounding=decimal.ROUND_HALF_EVEN, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN)


class UnsupportedValue(ValueError):
    """A result value the metric does not define (NaN/inf, unknown type)."""


def canonicalize_number(v: int | float) -> str:
    """SQLite INTEGER/REAL -> decimal string quantized to 6 places, round-half-even.

    Must hold (see tests/unit/test_compare.py::test_canonicalize_number and friends):
      1 and 1.0 -> "1.000000";  2.0000025 -> "2.000002";  -0.0 -> "0.000000"
      2**63 - 1 keeps every digit;  NaN / +-inf raise UnsupportedValue
    """
    d = Decimal(v)
    if not d.is_finite():
        raise UnsupportedValue(f"non-finite numeric result: {v!r}")
    q = d.quantize(QUANTUM, context=_CTX)
    if q.is_zero():
        return "0.000000"
    return format(q, "f")



def canonicalize_value(v: Any) -> CanonicalValue:
    if v is None:
        return ("null", None)
    if isinstance(v, (int, float)):
        return ("num", canonicalize_number(v))
    if isinstance(v, str):
        return ("text", v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return ("blob", bytes(v).hex())
    raise UnsupportedValue(f"unsupported value type {type(v).__name__}")


def canonicalize_row(row: Sequence[Any]) -> CanonicalRow:
    return tuple(canonicalize_value(v) for v in row)


def rows_match(
    pred_rows: Sequence[CanonicalRow],
    pred_ncols: int,
    gold_rows: Sequence[CanonicalRow],
    gold_ncols: int,
    *,
    ordered: bool,
) -> bool:
    if pred_ncols != gold_ncols:
        return False
    if ordered:
        return list(pred_rows) == list(gold_rows)
    return Counter(pred_rows) == Counter(gold_rows)
