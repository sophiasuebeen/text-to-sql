"""Reference query validation and execution-result cache.

Reference query ordering audit (frozen policy, AUDIT_METHOD below):
- Top-level ORDER BY (detected by sqlglot in the SQLite dialect) => sequence
  comparison, but only if ties cannot change the returned sequence. The audit
  re-runs the reference query with its ORDER BY keys projected as extra columns and
  LIMIT/OFFSET removed, groups adjacent rows with equal keys, and checks every
  tie group that overlaps the returned window: if it holds differing output rows,
  a different tie order would change the answer -> ``ambiguous_order``.
- Keys that cannot be resolved/projected safely (DISTINCT with non-output keys,
  set operations ordered by non-output terms, non-literal LIMIT, an audit query
  that disagrees with the reference result) -> ``needs_adjudication``: a documented manual
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

from .. import EVALUATOR_VERSION
from ..sql.executor import OK, ExecOutcome, Limits, execute_sql
from ..sql.policy import parse_single_read
from .compare import rows_match

AUDIT_METHOD = "projected_order_keys+limit_removal/v1"

# Exclusion reasons (data validity only; never used to hide comparator defects)
REFERENCE_PARSE_ERROR = "reference_parse_error"
AMBIGUOUS_ORDER = "ambiguous_order"
UNORDERED_LIMIT = "unordered_limit"
NEEDS_ADJUDICATION = "needs_adjudication"
REFERENCE_AUDIT_FAILED = "reference_audit_failed"
SELF_COMPARE_FAILED = "self_compare_failed"  # comparator defect: investigate, do not exclude


def normalize_reference_reason(reason: str | None) -> str | None:
    """Read exclusion reasons written before the reference terminology rename."""
    if reason is not None and reason.startswith("gold_"):
        return "reference_" + reason.removeprefix("gold_")
    return reason


def reference_exec_reason(status: str) -> str:
    return f"reference_{status}"


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


def audit_order(db_path: str | Path, tree: exp.Expression, reference: ExecOutcome, limits: Limits) -> str | None:
    """None if the reference result sequence is deterministic, else an exclusion reason."""
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

    n = reference.ncols
    outputs = [row[:n] for row in full.rows]
    end = len(outputs) if limit is None else offset + limit
    if outputs[offset:end] != list(reference.rows):
        return NEEDS_ADJUDICATION  # audit query does not reproduce the reference result

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
class ReferenceRecord:
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
    def from_json(cls, d: dict[str, Any]) -> "ReferenceRecord":
        return cls(d["eligible"], d["ordered"], normalize_reference_reason(d["reason"]),
                   ExecOutcome.from_json(d["outcome"]), d["audit_method"])


def audit_reference(db_path: str | Path, sql: str, limits: Limits = Limits()) -> ReferenceRecord:
    tree, err = parse_single_read(sql)
    if err:
        return ReferenceRecord(False, False, REFERENCE_PARSE_ERROR, ExecOutcome("parse_error", error=err))
    info = order_info(tree)

    out = execute_sql(db_path, sql, limits)
    if out.status != OK:
        return ReferenceRecord(False, info.ordered, reference_exec_reason(out.status), out)

    # Plumbing check: each reference result must compare equal to itself.
    if not rows_match(out.rows, out.ncols, out.rows, out.ncols, ordered=info.ordered):
        return ReferenceRecord(False, info.ordered, SELF_COMPARE_FAILED, out)

    if info.ordered:
        reason = audit_order(db_path, tree, out, limits)
        if reason:
            return ReferenceRecord(False, True, reason, out)
    elif info.has_limit:
        full = execute_sql(db_path, without_limit(tree), limits)
        if full.status != OK:
            return ReferenceRecord(False, False, REFERENCE_AUDIT_FAILED, out)
        if not rows_match(full.rows, full.ncols, out.rows, out.ncols, ordered=False):
            return ReferenceRecord(False, False, UNORDERED_LIMIT, out)

    return ReferenceRecord(True, info.ordered, None, out)


def reference_cache_key(db_sha256: str, sql: str, limits: Limits) -> str:
    ident = [db_sha256, sql, sqlite3.sqlite_version, EVALUATOR_VERSION, AUDIT_METHOD, asdict(limits)]
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()


class ReferenceCache:
    """Append-only JSONL cache of reference audit records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mem: dict[str, ReferenceRecord] = {}
        # Reuse earlier audits on Drive; current records take precedence.
        paths = [self.path]
        if self.path.name == "reference_cache.jsonl":
            paths.insert(0, self.path.with_name("gold_cache.jsonl"))
        for source in paths:
            if not source.exists():
                continue
            for line in source.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._mem[d["key"]] = ReferenceRecord.from_json(d["record"])

    def get(self, key: str) -> ReferenceRecord | None:
        return self._mem.get(key)

    def put(self, key: str, rec: ReferenceRecord) -> None:
        self._mem[key] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "record": rec.to_json()}) + "\n")

    def get_or_audit(self, db_path: str | Path, db_sha256: str, sql: str, limits: Limits = Limits()) -> ReferenceRecord:
        key = reference_cache_key(db_sha256, sql, limits)
        rec = self.get(key)
        if rec is None:
            rec = audit_reference(db_path, sql, limits)
            self.put(key, rec)
        return rec
