"""What SQL is allowed: output extraction, static parsing, and the runtime authorizer.

Two layers, checked independently:
- before execution: exactly one SQLite read statement, parsed with sqlglot;
- during execution: a SQLite authorizer allowing only reads and approved functions.

Extraction turns raw model output into exactly one SQL read query, or a structured rejection.

Accepted forms (after stripping surrounding whitespace):
- plain SQL, or
- one enclosing Markdown fence: ```` ```sql ```` / ```` ```sqlite ```` / bare ```` ``` ````.

Anything else (prose, extra fences, a second statement, trailing text) is
rejected. We never "recover" a first query by discarding the remainder.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

_FENCE_RE = re.compile(r"\A```([A-Za-z]*)[ \t]*\n(.*)\n```\Z", re.DOTALL)
_ALLOWED_FENCE_TAGS = {"", "sql", "sqlite"}
READ_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)


@dataclass(frozen=True)
class Extraction:
    ok: bool
    sql: str | None
    error: str | None = None  # "empty" | "bad_fence" | "parse_error" | "multiple_statements" | "not_read_query"


def _reject(error: str) -> Extraction:
    return Extraction(ok=False, sql=None, error=error)


def strip_fence(text: str) -> str | None:
    """Return the body if text is plain or one enclosing fence; None if malformed."""
    s = text.strip()
    if "```" not in s:
        return s
    m = _FENCE_RE.match(s)
    if not m or m.group(1).lower() not in _ALLOWED_FENCE_TAGS:
        return None
    body = m.group(2)
    if "```" in body:
        return None
    return body.strip()


def parse_single_read(sql: str) -> tuple[exp.Expression | None, str | None]:
    """Parse in the SQLite dialect; require exactly one read (SELECT-family) statement."""
    try:
        trees = [t for t in sqlglot.parse(sql, read="sqlite") if t is not None]
    except ParseError:
        return None, "parse_error"
    if not trees:
        return None, "empty"
    if len(trees) > 1:
        return None, "multiple_statements"
    tree = trees[0]
    if not isinstance(tree, READ_ROOTS):
        return None, "not_read_query"
    return tree, None


def extract_sql(raw: str) -> Extraction:
    body = strip_fence(raw)
    if body is None:
        return _reject("bad_fence")
    if not body:
        return _reject("empty")
    _tree, err = parse_single_read(body)
    if err:
        return _reject(err)
    return Extraction(ok=True, sql=body)


# ---------------------------------------------------------------- runtime authorizer

APPROVED_FUNCTIONS = frozenset(
    {
        # aggregates
        "avg", "count", "group_concat", "max", "min", "sum", "total", "string_agg",
        # scalar
        "abs", "char", "coalesce", "format", "glob", "hex", "ifnull", "iif", "instr",
        "length", "like", "likelihood", "likely", "lower", "ltrim", "nullif", "octet_length",
        "printf", "quote", "replace", "round", "rtrim", "sign", "substr", "substring",
        "trim", "typeof", "unicode", "unlikely", "upper", "concat", "concat_ws",
        # date/time
        "date", "datetime", "julianday", "strftime", "time", "unixepoch", "timediff",
        # math
        "acos", "asin", "atan", "atan2", "ceil", "ceiling", "cos", "exp", "floor", "ln",
        "log", "log10", "log2", "mod", "pi", "pow", "power", "sin", "sqrt", "tan", "trunc",
        # window
        "row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile",
        "lag", "lead", "first_value", "last_value", "nth_value",
    }
)
# Deliberately absent: random/randomblob (nondeterministic), load_extension,
# sqlite_* introspection, changes/last_insert_rowid, zeroblob (memory amplifier).

_SQLITE_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)
_ALLOWED_ACTIONS = {sqlite3.SQLITE_READ, sqlite3.SQLITE_SELECT, _SQLITE_RECURSIVE}


def authorizer(action: int, arg1: Any, arg2: Any, _db: Any, _src: Any) -> int:
    """sqlite3 authorizer: allow reads, SELECT, recursive CTEs and approved functions; deny the rest."""
    if action in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and isinstance(arg2, str) and arg2.lower() in APPROVED_FUNCTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY
