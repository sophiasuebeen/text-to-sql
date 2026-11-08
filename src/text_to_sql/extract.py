"""Turn raw model output into exactly one SQL read query, or a structured rejection.

Accepted forms (after stripping surrounding whitespace):
- plain SQL, or
- one enclosing Markdown fence: ```` ```sql ```` / ```` ```sqlite ```` / bare ```` ``` ````.

Anything else (prose, extra fences, a second statement, trailing text) is
rejected. We never "recover" a first query by discarding the remainder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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
