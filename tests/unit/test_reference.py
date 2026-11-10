import json

import pytest

import text_to_sql.evaluation.reference as ev
from text_to_sql.evaluation.reference import (
    AMBIGUOUS_ORDER,
    REFERENCE_PARSE_ERROR,
    NEEDS_ADJUDICATION,
    UNORDERED_LIMIT,
    ReferenceCache,
    audit_reference,
)
from text_to_sql.evaluation.score import score_prediction
from text_to_sql.io import sha256_file


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
        ("SELECT nope FROM people", False, False, "reference_sql_error"),
        ("SELEC name", False, False, REFERENCE_PARSE_ERROR),
    ],
)
def test_audit_reference(people_db, fast_limits, sql, eligible, ordered, reason):
    rec = audit_reference(people_db, sql, fast_limits)
    assert (rec.eligible, rec.ordered, rec.reason) == (eligible, ordered, reason)


@pytest.mark.parametrize(
    "reference, pred, correct, status",
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
def test_score_prediction(people_db, fast_limits, reference, pred, correct, status):
    rec = audit_reference(people_db, reference, fast_limits)
    s = score_prediction(people_db, pred, rec, fast_limits)
    assert (s.correct, s.pred_status) == (correct, status)


def test_missing_prediction_counts_as_failure(people_db, fast_limits):
    rec = audit_reference(people_db, "SELECT 1", fast_limits)
    s = score_prediction(people_db, None, rec, fast_limits)
    assert not s.correct and s.pred_status == "no_sql"


def test_ineligible_reference_cannot_be_scored(people_db, fast_limits):
    rec = audit_reference(people_db, "SELECT name FROM people LIMIT 1", fast_limits)
    with pytest.raises(ValueError):
        score_prediction(people_db, "SELECT 1", rec, fast_limits)


def test_reference_cache_hits_and_reloads(people_db, fast_limits, tmp_path, monkeypatch):
    calls = []
    real = ev.audit_reference
    monkeypatch.setattr(ev, "audit_reference", lambda *a, **k: calls.append(1) or real(*a, **k))
    path = tmp_path / "reference.jsonl"
    digest = sha256_file(people_db)

    cache = ReferenceCache(path)
    r1 = cache.get_or_audit(people_db, digest, "SELECT name FROM people ORDER BY id", fast_limits)
    r2 = cache.get_or_audit(people_db, digest, "SELECT name FROM people ORDER BY id", fast_limits)
    assert len(calls) == 1 and r1 == r2

    r3 = ReferenceCache(path).get_or_audit(people_db, digest, "SELECT name FROM people ORDER BY id", fast_limits)
    assert len(calls) == 1 and r3 == r1
    # different db checksum is a different key
    cache.get_or_audit(people_db, "0" * 64, "SELECT name FROM people ORDER BY id", fast_limits)
    assert len(calls) == 2


def test_legacy_cache_reused_and_current_cache_takes_precedence(people_db, fast_limits, tmp_path, monkeypatch):
    sql = "SELECT nope FROM people"
    key = ev.reference_cache_key(sha256_file(people_db), sql, fast_limits)
    record = audit_reference(people_db, sql, fast_limits)
    legacy_record = record.to_json()
    legacy_record["reason"] = "gold_sql_error"
    legacy_path = tmp_path / "gold_cache.jsonl"
    legacy_path.write_text(json.dumps({"key": key, "record": legacy_record}) + "\n")
    original = legacy_path.read_bytes()

    def unexpected_audit(*args, **kwargs):
        pytest.fail("the existing audit should be reused")

    monkeypatch.setattr(ev, "audit_reference", unexpected_audit)
    path = tmp_path / "reference_cache.jsonl"
    cache = ReferenceCache(path)
    assert cache.get_or_audit(people_db, sha256_file(people_db), sql, fast_limits) == record

    updated = ev.ReferenceRecord(False, False, "reference_timeout", record.outcome)
    cache.put(key, updated)
    assert ReferenceCache(path).get(key) == updated
    assert legacy_path.read_bytes() == original
