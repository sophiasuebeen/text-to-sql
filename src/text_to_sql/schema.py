"""Deterministic SQLite schema -> CREATE TABLE text.

Frozen contract (baseline and adapter see byte-identical schema text):
- Tables sorted by (casefolded name, name); internal ``sqlite_%`` tables skipped.
- Columns keep their original declaration order and declared type (type omitted if none).
- Every identifier is double-quoted with embedded quotes doubled, so spaces,
  reserved words and quotes are always unambiguous.
- Primary key as a table-level constraint (composite keys in key order).
- Foreign keys grouped by constraint, columns in ``seq`` order, constraints
  sorted by (source columns, referenced table). SQLite's internal FK ids run in
  reverse declaration order, so they are not used for ordering. A reference with no explicit target columns is rendered without
  a column list, exactly as SQLite stores it.

PRAGMAs are used here on a trusted, read-only connection; this is separate from
the sandboxed evaluator, which denies PRAGMAs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SERIALIZER_VERSION = "1"


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _connect_ro(db_path: str | Path) -> sqlite3.Connection:
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'"
    ).fetchall()
    return sorted((r[0] for r in rows), key=lambda n: (n.casefold(), n))


def serialize_table(conn: sqlite3.Connection, table: str) -> str:
    q = quote_ident(table)
    # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
    cols = conn.execute(f"PRAGMA table_info({q})").fetchall()
    lines = []
    for _cid, name, ctype, *_rest in sorted(cols, key=lambda c: c[0]):
        lines.append(f"  {quote_ident(name)} {ctype}".rstrip())

    pk_cols = [c[1] for c in sorted((c for c in cols if c[5] > 0), key=lambda c: c[5])]
    if pk_cols:
        lines.append(f"  PRIMARY KEY ({', '.join(quote_ident(c) for c in pk_cols)})")

    # PRAGMA foreign_key_list: (id, seq, table, from, to, on_update, on_delete, match)
    fks: dict[int, list[tuple]] = {}
    for row in conn.execute(f"PRAGMA foreign_key_list({q})").fetchall():
        fks.setdefault(row[0], []).append(row)
    fk_lines = []
    for parts in fks.values():
        parts = sorted(parts, key=lambda r: r[1])
        ref_table = parts[0][2]
        src_cols = tuple(r[3] for r in parts)
        targets = [r[4] for r in parts]
        ref = f"REFERENCES {quote_ident(ref_table)}"
        if all(t is not None for t in targets):
            ref += f" ({', '.join(quote_ident(t) for t in targets)})"
        line = f"  FOREIGN KEY ({', '.join(quote_ident(c) for c in src_cols)}) {ref}"
        fk_lines.append(((src_cols, ref_table), line))
    lines.extend(line for _key, line in sorted(fk_lines))

    return f"CREATE TABLE {q} (\n" + ",\n".join(lines) + "\n);"


def serialize_schema(db_path: str | Path) -> str:
    conn = _connect_ro(db_path)
    try:
        return "\n\n".join(serialize_table(conn, t) for t in _table_names(conn))
    finally:
        conn.close()
