"""Execute one untrusted SQL read query in an isolated, resource-bounded worker.

Layers, from outermost in:
1. A fresh spawned process per query; the parent terminates (then kills) it at
   the wall-clock deadline and always reaps it.
2. Read-only URI connection, extension loading disabled, autocommit mode.
3. SQLite ``hard_heap_limit`` (process-wide heap cap) and ``SQLITE_LIMIT_LENGTH``
   (max size of any single string/blob, i.e. the per-cell bound); best-effort
   ``RLIMIT_AS`` on Linux (macOS does not enforce it).
4. The authorizer allowlist from ``policy.py`` (reads and approved functions only).
5. Progress handler aborting the statement after the internal deadline.
6. Incremental ``fetchmany`` with row-count and serialized-byte budgets.

Results are canonicalized inside the worker so only small, hashable, typed
values cross the process boundary.
"""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..evaluation.compare import CanonicalRow, UnsupportedValue, canonicalize_row
from .policy import authorizer

# Structured outcome statuses
OK = "ok"
SQL_ERROR = "sql_error"
DISALLOWED = "disallowed"
TIMEOUT = "timeout"
RESOURCE_LIMIT = "resource_limit"
UNSUPPORTED_VALUE = "unsupported_value"
WORKER_ERROR = "worker_error"

@dataclass(frozen=True)
class Limits:
    wall_seconds: float = 30.0
    memory_bytes: int = 512 * 1024 * 1024
    max_rows: int = 100_000
    max_output_bytes: int = 16 * 1024 * 1024
    max_cell_bytes: int = 1024 * 1024
    progress_opcodes: int = 10_000
    fetch_batch: int = 1_000


@dataclass(frozen=True)
class ExecOutcome:
    status: str
    ncols: int | None = None
    rows: tuple[CanonicalRow, ...] = ()
    error: str | None = None
    elapsed_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == OK

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["rows"] = [[list(v) for v in row] for row in self.rows]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ExecOutcome":
        rows = tuple(tuple(tuple(v) for v in row) for row in d.get("rows", []))
        return cls(
            status=d["status"], ncols=d.get("ncols"), rows=rows, error=d.get("error"),
            elapsed_s=d.get("elapsed_s", 0.0), extra=d.get("extra", {}),
        )


def _value_bytes(v: Any) -> int:
    if v is None:
        return 1
    if isinstance(v, str):
        return len(v.encode("utf-8", "surrogatepass"))
    if isinstance(v, (bytes, bytearray, memoryview)):
        return len(v)
    return 8


def _classify_db_error(e: sqlite3.Error, timed_out: bool) -> tuple[str, str]:
    msg = str(e)
    low = msg.lower()
    if timed_out or "interrupted" in low:
        return TIMEOUT, msg
    if "not authorized" in low or "prohibited" in low:
        return DISALLOWED, msg
    if "out of memory" in low or "too big" in low or isinstance(e, sqlite3.DataError):
        return RESOURCE_LIMIT, msg
    if "one statement at a time" in low:
        return DISALLOWED, msg
    return SQL_ERROR, msg


def _set_address_space_limit(nbytes: int) -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
    except (ImportError, ValueError, OSError):
        pass


def _run_in_worker(db_path: str, sql: str, limits: Limits) -> tuple:
    """Body of the worker. Returns a picklable message tuple."""
    _set_address_space_limit(limits.memory_bytes + 256 * 1024 * 1024)  # interpreter headroom
    deadline = time.monotonic() + max(0.1, limits.wall_seconds - 0.5)
    timed_out = False

    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=True)
    try:
        if hasattr(conn, "enable_load_extension"):
            conn.enable_load_extension(False)
        # Trusted configuration, applied *before* the authorizer is installed.
        conn.execute(f"PRAGMA hard_heap_limit = {int(limits.memory_bytes)}")
        conn.execute("PRAGMA query_only = ON")
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, limits.max_cell_bytes)
        conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)

        def progress() -> int:
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1
            return 0

        conn.set_progress_handler(progress, limits.progress_opcodes)
        conn.set_authorizer(authorizer)

        try:
            cur = conn.execute(sql)
            ncols = len(cur.description) if cur.description else 0
            rows: list[CanonicalRow] = []
            used = 0
            while True:
                batch = cur.fetchmany(limits.fetch_batch)
                if not batch:
                    break
                for raw in batch:
                    if len(rows) >= limits.max_rows:
                        return (RESOURCE_LIMIT, None, None, f"more than {limits.max_rows} rows")
                    for v in raw:
                        nb = _value_bytes(v)
                        if nb > limits.max_cell_bytes:
                            return (RESOURCE_LIMIT, None, None, f"cell over {limits.max_cell_bytes} bytes")
                        used += nb
                    if used > limits.max_output_bytes:
                        return (RESOURCE_LIMIT, None, None, f"output over {limits.max_output_bytes} bytes")
                    rows.append(canonicalize_row(raw))
            if cur.description is None:
                return (DISALLOWED, None, None, "statement returned no result set")
            return (OK, ncols, tuple(rows), None)
        except UnsupportedValue as e:
            return (UNSUPPORTED_VALUE, None, None, str(e))
        except sqlite3.Warning as e:  # e.g. multiple statements
            return (DISALLOWED, None, None, str(e))
        except sqlite3.Error as e:
            status, msg = _classify_db_error(e, timed_out)
            return (status, None, None, msg)
        except MemoryError:
            return (RESOURCE_LIMIT, None, None, "worker MemoryError")
    finally:
        conn.close()


def _worker_entry(db_path: str, sql: str, limits: Limits, send) -> None:
    try:
        msg = _run_in_worker(db_path, sql, limits)
    except MemoryError:
        msg = (RESOURCE_LIMIT, None, None, "worker MemoryError")
    except BaseException as e:  # never let the worker die silently
        msg = (WORKER_ERROR, None, None, f"{type(e).__name__}: {e}")
    try:
        send.send(msg)
    finally:
        send.close()


def execute_sql(
    db_path: str | Path,
    sql: str,
    limits: Limits = Limits(),
    *,
    start_method: str = "spawn",
) -> ExecOutcome:
    """Run ``sql`` against ``db_path`` in a fresh worker; never raises for query failures."""
    if not Path(db_path).is_file():
        raise FileNotFoundError(db_path)
    ctx = mp.get_context(start_method)
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_worker_entry, args=(str(db_path), sql, limits, send), daemon=True
    )
    start = time.monotonic()
    proc.start()
    send.close()
    msg = None
    try:
        if recv.poll(limits.wall_seconds):
            try:
                msg = recv.recv()
            except EOFError:
                msg = None
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
        proc.join()
        recv.close()
    elapsed = time.monotonic() - start

    if msg is None:
        if elapsed >= limits.wall_seconds:
            return ExecOutcome(TIMEOUT, error="wall-clock deadline; worker terminated", elapsed_s=elapsed)
        return ExecOutcome(
            RESOURCE_LIMIT if proc.exitcode and proc.exitcode < 0 else WORKER_ERROR,
            error=f"worker exited without result (exitcode={proc.exitcode})",
            elapsed_s=elapsed,
        )
    status, ncols, rows, err = msg
    return ExecOutcome(status, ncols=ncols, rows=rows or (), error=err, elapsed_s=elapsed)
