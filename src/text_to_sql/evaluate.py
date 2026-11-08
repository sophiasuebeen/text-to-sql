"""Gold integrity audit, gold-outcome cache and per-prediction scoring.

Gold ordering audit (frozen policy, AUDIT_METHOD below):
- Top-level ORDER BY (detected by sqlglot in the SQLite dialect) => sequence
  comparison, but only if ties cannot change the returned sequence. The audit
  re-runs the gold with its ORDER BY keys projected as extra columns and
  LIMIT/OFFSET removed, groups adjacent rows with equal keys, and checks every
  tie group that overlaps the returned window: if it holds differing output rows,
  a different tie order would change the answer -> ``ambiguous_order``.
- Keys that cannot be resolved/projected safely (DISTINCT with non-output keys,
  set operations ordered by non-output terms, non-literal LIMIT, an audit query
  that disagrees with the gold) -> ``needs_adjudication``: a documented manual
  decision frozen before any model score exists.
- LIMIT/OFFSET without top-level ORDER BY -> ``unordered_limit`` unless removing
  the LIMIT/OFFSET returns the same multiset (the limit cuts nothing).
- Known limit: key equality uses canonical values, so ties under a non-binary
  collation (e.g. NOCASE) are not detected; numeric keys equal at 6 places are
  conservatively treated as ties.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlglot import exp

from . import EVALUATOR_VERSION
from .compare import rows_match
from .extract import parse_single_read
from .sandbox import OK, ExecOutcome, Limits, execute_sql

AUDIT_METHOD = "projected_order_keys+limit_removal/v1"

# Exclusion reasons (data validity only; never used to hide comparator defects)
GOLD_PARSE_ERROR = "gold_parse_error"
AMBIGUOUS_ORDER = "ambiguous_order"
UNORDERED_LIMIT = "unordered_limit"
NEEDS_ADJUDICATION = "needs_adjudication"
GOLD_AUDIT_FAILED = "gold_audit_failed"
SELF_COMPARE_FAILED = "self_compare_failed"  # comparator defect: investigate, do not exclude


def gold_exec_reason(status: str) -> str:
    return f"gold_{status}"


@dataclass(frozen=True)
class OrderInfo:
    ordered: bool
    has_limit: bool


def order_info(tree: exp.Expression) -> OrderInfo:
    """Top-level ORDER BY / LIMIT / OFFSET of the outermost query only."""
    return OrderInfo(
        ordered=bool(tree.args.get("order")),
        has_limit=bool(tree.args.get("limit") or tree.args.get("offset")),
    )


def without_limit(tree: exp.Expression) -> str:
    t = tree.copy()
    t.set("limit", None)
    t.set("offset", None)
    return t.sql(dialect="sqlite")


def _int_literal(node: exp.Expression | None) -> int | None:
    if node is None:
        return None
    e = node.args.get("expression") if isinstance(node, (exp.Limit, exp.Offset)) else node
    if isinstance(e, exp.Literal) and not e.is_string and e.this.isdigit():
        return int(e.this)
    raise ValueError("non-literal LIMIT/OFFSET")


def _leftmost_select(tree: exp.Expression) -> exp.Select:
    node = tree
    while not isinstance(node, exp.Select):
        node = node.this
    return node


def _is_star(p: exp.Expression) -> bool:
    return isinstance(p, exp.Star) or (isinstance(p, exp.Column) and isinstance(p.this, exp.Star))


def _resolve_order_keys(tree: exp.Expression) -> list[tuple[str, Any]] | None:
    """Each key is ("col", output_index) or ("expr", expression); None if unresolvable."""
    projections = _leftmost_select(tree).expressions
    star = any(_is_star(p) for p in projections)
    keys: list[tuple[str, Any]] = []
    for ordered in tree.args["order"].expressions:
        e = ordered.this
        if isinstance(e, exp.Literal) and not e.is_string:
            keys.append(("col", int(e.this) - 1))  # ordinal = output column, star or not
            continue
        match = None
        for j, p in enumerate(projections):
            aliased = isinstance(p, exp.Alias)
            if (aliased and isinstance(e, exp.Column) and not e.table and e.name.lower() == p.alias.lower()) or (
                (p.this if aliased else p) == e
            ):
                match = j
                break
        if match is not None and not star:
            keys.append(("col", match))
        elif isinstance(tree, exp.Select):
            keys.append(("expr", e))
        else:
            return None  # compound select ordered by something we cannot map to an output column
    if isinstance(tree, exp.Select) and tree.args.get("distinct") and any(k[0] == "expr" for k in keys):
        return None  # extra projections would change DISTINCT semantics
    return keys


def audit_order(db_path: str | Path, tree: exp.Expression, gold: ExecOutcome, limits: Limits) -> str | None:
    """None if the gold sequence is deterministic, else an exclusion reason."""
    try:
        offset = _int_literal(tree.args.get("offset")) or 0
        limit = _int_literal(tree.args.get("limit"))
    except ValueError:
        return NEEDS_ADJUDICATION
    keys = _resolve_order_keys(tree)
    if keys is None:
        return NEEDS_ADJUDICATION

    t = tree.copy()
    t.set("limit", None)
    t.set("offset", None)
    extras = [e for kind, e in keys if kind == "expr"]
    if extras:
        t.set("expressions", t.expressions + [exp.alias_(e.copy(), f"__tts_k{i}") for i, e in enumerate(extras)])
    full = execute_sql(db_path, t.sql(dialect="sqlite"), limits)
    if full.status != OK:
        return NEEDS_ADJUDICATION

    n = gold.ncols
    outputs = [row[:n] for row in full.rows]
    end = len(outputs) if limit is None else offset + limit
    if outputs[offset:end] != list(gold.rows):
        return NEEDS_ADJUDICATION  # audit query does not reproduce the gold

    def key_of(row):
        k, extra_i = [], 0
        for kind, val in keys:
            if kind == "col":
                k.append(row[val])
            else:
                k.append(row[n + extra_i])
                extra_i += 1
        return tuple(k)

    start = 0
    for i in range(1, len(full.rows) + 1):
        if i == len(full.rows) or key_of(full.rows[i]) != key_of(full.rows[start]):
            overlaps_window = start < end and i > offset
            if overlaps_window and len(set(outputs[start:i])) > 1:
                return AMBIGUOUS_ORDER
            start = i
    return None


@dataclass(frozen=True)
class GoldRecord:
    eligible: bool
    ordered: bool
    reason: str | None
    outcome: ExecOutcome
    audit_method: str = AUDIT_METHOD

    def to_json(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible, "ordered": self.ordered, "reason": self.reason,
            "outcome": self.outcome.to_json(), "audit_method": self.audit_method,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "GoldRecord":
        return cls(d["eligible"], d["ordered"], d["reason"], ExecOutcome.from_json(d["outcome"]), d["audit_method"])


def audit_gold(db_path: str | Path, sql: str, limits: Limits = Limits()) -> GoldRecord:
    tree, err = parse_single_read(sql)
    if err:
        return GoldRecord(False, False, GOLD_PARSE_ERROR, ExecOutcome("parse_error", error=err))
    info = order_info(tree)

    out = execute_sql(db_path, sql, limits)
    if out.status != OK:
        return GoldRecord(False, info.ordered, gold_exec_reason(out.status), out)

    # Plumbing check: the comparator must accept gold against itself.
    if not rows_match(out.rows, out.ncols, out.rows, out.ncols, ordered=info.ordered):
        return GoldRecord(False, info.ordered, SELF_COMPARE_FAILED, out)

    if info.ordered:
        reason = audit_order(db_path, tree, out, limits)
        if reason:
            return GoldRecord(False, True, reason, out)
    elif info.has_limit:
        full = execute_sql(db_path, without_limit(tree), limits)
        if full.status != OK:
            return GoldRecord(False, False, GOLD_AUDIT_FAILED, out)
        if not rows_match(full.rows, full.ncols, out.rows, out.ncols, ordered=False):
            return GoldRecord(False, False, UNORDERED_LIMIT, out)

    return GoldRecord(True, info.ordered, None, out)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def gold_cache_key(db_sha256: str, sql: str, limits: Limits) -> str:
    ident = [db_sha256, sql, sqlite3.sqlite_version, EVALUATOR_VERSION, AUDIT_METHOD, asdict(limits)]
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()


class GoldCache:
    """Append-only JSONL cache of gold audit records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mem: dict[str, GoldRecord] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._mem[d["key"]] = GoldRecord.from_json(d["record"])

    def get(self, key: str) -> GoldRecord | None:
        return self._mem.get(key)

    def put(self, key: str, rec: GoldRecord) -> None:
        self._mem[key] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "record": rec.to_json()}) + "\n")

    def get_or_audit(self, db_path: str | Path, db_sha256: str, sql: str, limits: Limits = Limits()) -> GoldRecord:
        key = gold_cache_key(db_sha256, sql, limits)
        rec = self.get(key)
        if rec is None:
            rec = audit_gold(db_path, sql, limits)
            self.put(key, rec)
        return rec


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
