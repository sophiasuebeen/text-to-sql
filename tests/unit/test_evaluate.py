import pytest

import text_to_sql.evaluate as ev
from text_to_sql.evaluate import (
    AMBIGUOUS_ORDER,
    GOLD_PARSE_ERROR,
    NEEDS_ADJUDICATION,
    UNORDERED_LIMIT,
    GoldCache,
    audit_gold,
    score_prediction,
    sha256_file,
)


@pytest.mark.parametrize(
    "sql, eligible, ordered, reason",
    [
        ("SELECT name FROM people", True, False, None),
        ("SELECT name FROM people ORDER BY id", True, True, None),
        # Ann and Cid tie on age=30 and differ in name -> ambiguous
        ("SELECT name FROM people ORDER BY age", False, True, AMBIGUOUS_ORDER),
        # ties exist but tied rows are identical -> permutation cannot change the sequence
        ("SELECT age FROM people ORDER BY age", True, True, None),
        ("SELECT name FROM people ORDER BY age DESC LIMIT 1", True, True, None),
        ("SELECT name FROM people ORDER BY age DESC LIMIT 2", False, True, AMBIGUOUS_ORDER),
        ("SELECT name FROM people LIMIT 2", False, False, UNORDERED_LIMIT),
        ("SELECT name FROM people LIMIT 10", True, False, None),  # limit cuts nothing
        ("SELECT name FROM (SELECT name FROM people ORDER BY age)", True, False, None),  # not top-level
        ("SELECT name FROM people UNION SELECT 'z' ORDER BY 1", True, True, None),
        # tie group (Ann/Cid) lies outside the returned window -> fine
        ("SELECT name FROM people ORDER BY age LIMIT 1", True, True, None),
        ("SELECT name FROM people ORDER BY age DESC LIMIT 1 OFFSET 2", False, True, AMBIGUOUS_ORDER),
        # aggregate keys, aliases, ordinals, star
        ("SELECT age, count(*) FROM people GROUP BY age ORDER BY count(*) DESC LIMIT 1", True, True, None),
        ("SELECT age, count(*) AS c FROM people GROUP BY age ORDER BY c DESC LIMIT 1", True, True, None),
        ("SELECT name FROM people GROUP BY name ORDER BY count(*) DESC LIMIT 1", False, True, AMBIGUOUS_ORDER),
        ("SELECT name, age FROM people ORDER BY 2", False, True, AMBIGUOUS_ORDER),
        ("SELECT * FROM people ORDER BY id", True, True, None),
        ("SELECT * FROM people ORDER BY age", False, True, AMBIGUOUS_ORDER),
        ("SELECT DISTINCT name FROM people ORDER BY age", False, True, NEEDS_ADJUDICATION),
        ("SELECT nope FROM people", False, False, "gold_sql_error"),
        ("SELEC name", False, False, GOLD_PARSE_ERROR),
    ],
)
def test_audit_gold(people_db, fast_limits, sql, eligible, ordered, reason):
    rec = audit_gold(people_db, sql, fast_limits)
    assert (rec.eligible, rec.ordered, rec.reason) == (eligible, ordered, reason)


@pytest.mark.parametrize(
    "gold, pred, correct, status",
    [
        ("SELECT count(*) FROM people", "SELECT count(id) FROM people", True, "ok"),
        ("SELECT name FROM people WHERE age = 30", "SELECT name FROM people WHERE age > 29 AND age < 31", True, "ok"),
        ("SELECT name FROM people", "SELECT name FROM people ORDER BY name DESC", True, "ok"),
        ("SELECT name FROM people ORDER BY id", "SELECT name FROM people ORDER BY id DESC", False, "ok"),
        ("SELECT name FROM people", "SELECT name, age FROM people", False, "ok"),
        ("SELECT name FROM people", "SELECT nme FROM people", False, "sql_error"),
        ("SELECT name FROM people", "DELETE FROM people", False, "parse_error"),
        ("SELECT name FROM people", "SELECT name FROM people; DELETE FROM people", False, "parse_error"),
    ],
)
def test_score_prediction(people_db, fast_limits, gold, pred, correct, status):
    rec = audit_gold(people_db, gold, fast_limits)
    s = score_prediction(people_db, pred, rec, fast_limits)
    assert (s.correct, s.pred_status) == (correct, status)


def test_missing_prediction_counts_as_failure(people_db, fast_limits):
    rec = audit_gold(people_db, "SELECT 1", fast_limits)
    s = score_prediction(people_db, None, rec, fast_limits)
    assert not s.correct and s.pred_status == "no_sql"


def test_ineligible_gold_cannot_be_scored(people_db, fast_limits):
    rec = audit_gold(people_db, "SELECT name FROM people LIMIT 1", fast_limits)
    with pytest.raises(ValueError):
        score_prediction(people_db, "SELECT 1", rec, fast_limits)


def test_gold_cache_hits_and_reloads(people_db, fast_limits, tmp_path, monkeypatch):
    calls = []
    real = ev.audit_gold
    monkeypatch.setattr(ev, "audit_gold", lambda *a, **k: calls.append(1) or real(*a, **k))
    path = tmp_path / "gold.jsonl"
    digest = sha256_file(people_db)

    cache = GoldCache(path)
    r1 = cache.get_or_audit(people_db, digest, "SELECT name FROM people ORDER BY id", fast_limits)
    r2 = cache.get_or_audit(people_db, digest, "SELECT name FROM people ORDER BY id", fast_limits)
    assert len(calls) == 1 and r1 == r2

    r3 = GoldCache(path).get_or_audit(people_db, digest, "SELECT name FROM people ORDER BY id", fast_limits)
    assert len(calls) == 1 and r3 == r1
    # different db checksum is a different key
    cache.get_or_audit(people_db, "0" * 64, "SELECT name FROM people ORDER BY id", fast_limits)
    assert len(calls) == 2
