import multiprocessing as mp

import pytest

from text_to_sql.sql.executor import (
    DISALLOWED,
    OK,
    RESOURCE_LIMIT,
    TIMEOUT,
    UNSUPPORTED_VALUE,
    SQL_ERROR,
    Limits,
    execute_sql,
)

RUNAWAY = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"


def test_success_returns_canonical_rows(people_db, fast_limits):
    out = execute_sql(people_db, "SELECT name, score FROM people WHERE id <= 2 ORDER BY id", fast_limits)
    assert out.status == OK and out.ncols == 2
    assert out.rows == ((("text", "Ann"), ("num", "1.500000")), (("text", "bob"), ("num", "2.000000")))


def test_empty_result_keeps_arity(people_db, fast_limits):
    out = execute_sql(people_db, "SELECT name, age FROM people WHERE 0", fast_limits)
    assert out.status == OK and out.ncols == 2 and out.rows == ()


def test_sql_error(people_db, fast_limits):
    assert execute_sql(people_db, "SELECT nope FROM people", fast_limits).status == SQL_ERROR


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM people",
        "INSERT INTO people (id) VALUES (99)",
        "UPDATE people SET age = 0",
        "CREATE TABLE x (a)",
        "DROP TABLE people",
        "ATTACH DATABASE ':memory:' AS other",
        "DETACH DATABASE main",
        "PRAGMA table_info(people)",
        "PRAGMA query_only = OFF",
        "BEGIN",
        "VACUUM",
        "SELECT random()",
        "SELECT load_extension('x')",
        "SELECT 1; DELETE FROM people",
    ],
)
def test_denied(people_db, fast_limits, sql):
    out = execute_sql(people_db, sql, fast_limits)
    assert out.status in {DISALLOWED, SQL_ERROR}, out
    assert out.status != OK
    # database untouched
    assert execute_sql(people_db, "SELECT count(*) FROM people", fast_limits).rows == ((("num", "4.000000"),),)


def test_recursive_cte_and_functions_allowed(people_db, fast_limits):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 5) SELECT sum(x), upper('a') FROM c"
    out = execute_sql(people_db, sql, fast_limits)
    assert out.status == OK and out.rows == ((("num", "15.000000"), ("text", "A")),)


def test_runaway_query_times_out_via_progress_handler(people_db):
    out = execute_sql(people_db, RUNAWAY, Limits(wall_seconds=2.0))
    assert out.status == TIMEOUT
    assert out.elapsed_s < 4.0
    assert mp.active_children() == []


def test_parent_terminates_worker_when_handler_cannot_fire(people_db):
    out = execute_sql(people_db, RUNAWAY, Limits(wall_seconds=1.5, progress_opcodes=2_000_000_000))
    assert out.status == TIMEOUT and "terminated" in out.error
    assert mp.active_children() == []


def test_row_limit(people_db, fast_limits):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 50) SELECT x FROM c"
    out = execute_sql(people_db, sql, Limits(wall_seconds=3.0, max_rows=10))
    assert out.status == RESOURCE_LIMIT


def test_output_byte_budget(people_db):
    out = execute_sql(people_db, "SELECT name FROM people", Limits(wall_seconds=3.0, max_output_bytes=5))
    assert out.status == RESOURCE_LIMIT


def test_huge_cell_bounded(people_db, fast_limits):
    # printf silently returns NULL past SQLITE_LIMIT_LENGTH; replace() raises "too big".
    out = execute_sql(people_db, "SELECT replace(printf('%.*c', 600000, 'x'), 'x', 'xyz')", fast_limits)
    assert out.status == RESOURCE_LIMIT
    ok = execute_sql(people_db, "SELECT length(printf('%.*c', 600000, 'x'))", fast_limits)
    assert ok.rows == ((("num", "600000.000000"),),)


def test_heap_limit(people_db):
    sql = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 200000) "
        "SELECT length(group_concat(printf('%040d', x))) FROM c"
    )
    limits = Limits(wall_seconds=10.0, memory_bytes=4 * 1024 * 1024, max_cell_bytes=64 * 1024 * 1024)
    assert execute_sql(people_db, sql, limits).status == RESOURCE_LIMIT


def test_non_finite_is_unsupported(people_db, fast_limits):
    assert execute_sql(people_db, "SELECT 1e999", fast_limits).status == UNSUPPORTED_VALUE


def test_missing_db_raises(tmp_path, fast_limits):
    with pytest.raises(FileNotFoundError):
        execute_sql(tmp_path / "nope.sqlite", "SELECT 1", fast_limits)
