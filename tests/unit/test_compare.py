"""Independently authored comparator fixtures (not gold self-comparison)."""

import pytest

from text_to_sql.compare import (
    UnsupportedValue,
    canonicalize_number,
    canonicalize_row,
    canonicalize_value,
    rows_match,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, "1.000000"),
        (1.0, "1.000000"),
        (-3, "-3.000000"),
        (0.1 + 0.2, "0.300000"),
        # exact binary expansion decides, not the decimal literal:
        (1.0000005, "1.000001"),  # 1.00000050000000006988... rounds up
        (1.0000015, "1.000001"),  # 1.00000149999999998762... rounds down
        (2.0000025, "2.000002"),  # half-up on the literal would wrongly give 2.000003
        (-0.0, "0.000000"),
        (-0.0000001, "0.000000"),  # rounds to -0, normalized to 0
        (0.0000004, "0.000000"),
        (123456.7890123, "123456.789012"),
    ],
)
def test_canonicalize_number(value, expected):
    assert canonicalize_number(value) == expected


def test_integer_precision_is_preserved():
    big = 2**63 - 1
    assert canonicalize_number(big) == "9223372036854775807.000000"
    assert canonicalize_number(big) != canonicalize_number(float(big))
    assert canonicalize_number(2**53 + 1) != canonicalize_number(2**53)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rejected(bad):
    with pytest.raises(UnsupportedValue):
        canonicalize_number(bad)


def test_value_types_are_distinct():
    assert canonicalize_value(None) == canonicalize_value(None)
    assert canonicalize_value(None) != canonicalize_value(0)
    assert canonicalize_value(None) != canonicalize_value("")
    assert canonicalize_value("1") != canonicalize_value(1)
    assert canonicalize_value("Ann") != canonicalize_value("ann")
    assert canonicalize_value(b"\x00\x01") == ("blob", "0001")
    with pytest.raises(UnsupportedValue):
        canonicalize_value(object())


def R(*rows):
    return [canonicalize_row(r) for r in rows]


def test_empty_results_compare_arity():
    assert rows_match([], 2, [], 2, ordered=False)
    assert not rows_match([], 1, [], 2, ordered=False)
    assert not rows_match([], 1, [], 2, ordered=True)


def test_columns_by_position_names_ignored():
    assert rows_match(R((1, "a")), 2, R((1.0, "a")), 2, ordered=False)
    assert not rows_match(R((1, "a")), 2, R(("a", 1)), 2, ordered=False)


def test_multiset_keeps_duplicates():
    assert rows_match(R((1,), (2,), (1,)), 1, R((1,), (1,), (2,)), 1, ordered=False)
    assert not rows_match(R((1,), (2,)), 1, R((1,), (1,), (2,)), 1, ordered=False)
    assert not rows_match(R((1,), (1,), (2,)), 1, R((1,), (2,), (2,)), 1, ordered=False)


def test_ordered_requires_sequence():
    assert rows_match(R((1,), (2,)), 1, R((1,), (2,)), 1, ordered=True)
    assert not rows_match(R((2,), (1,)), 1, R((1,), (2,)), 1, ordered=True)
    assert rows_match(R((2,), (1,)), 1, R((1,), (2,)), 1, ordered=False)


def test_mixed_types_in_one_column_do_not_crash():
    pred = R((1,), ("1",), (None,), (b"x",))
    gold = R((None,), (b"x",), ("1",), (1.0,))
    assert rows_match(pred, 1, gold, 1, ordered=False)
    assert not rows_match(pred, 1, R((None,), (b"x",), (1,), (1.0,)), 1, ordered=False)
