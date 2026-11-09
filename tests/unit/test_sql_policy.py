import pytest

from text_to_sql.sql.policy import extract_sql


@pytest.mark.parametrize(
    "raw, sql",
    [
        ("SELECT 1", "SELECT 1"),
        ("  SELECT 1;\n", "SELECT 1;"),
        ("```sql\nSELECT a FROM t\n```", "SELECT a FROM t"),
        ("```SQLite\nSELECT a FROM t\n```", "SELECT a FROM t"),
        ("```\nSELECT a\nFROM t\n```", "SELECT a\nFROM t"),
        ("WITH x AS (SELECT 1 AS a) SELECT a FROM x", "WITH x AS (SELECT 1 AS a) SELECT a FROM x"),
        ("SELECT a FROM t UNION SELECT b FROM u", "SELECT a FROM t UNION SELECT b FROM u"),
    ],
)
def test_accepts(raw, sql):
    r = extract_sql(raw)
    assert r.ok and r.sql == sql


@pytest.mark.parametrize(
    "raw, error",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("Here is the query: SELECT 1", "parse_error"),
        ("SELECT 1\nThis returns one.", "parse_error"),
        ("SELECT 1; SELECT 2", "multiple_statements"),
        ("SELECT 1; DROP TABLE t", "multiple_statements"),
        ("DELETE FROM t", "not_read_query"),
        ("PRAGMA table_info(t)", "parse_error"),
        ("Sure!\n```sql\nSELECT 1\n```", "bad_fence"),
        ("```sql\nSELECT 1\n```\nHope this helps", "bad_fence"),
        ("```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```", "bad_fence"),
        ("```python\nSELECT 1\n```", "bad_fence"),
        ("```sql\nSELECT 1", "bad_fence"),
    ],
)
def test_rejects(raw, error):
    r = extract_sql(raw)
    assert not r.ok and r.sql is None
    if error == "parse_error":
        # sqlglot may classify unknown statements as parse errors or non-read roots
        assert r.error in {"parse_error", "not_read_query"}
    else:
        assert r.error == error
