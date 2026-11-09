import math

import pytest

from text_to_sql.evaluation.metrics import (
    accuracy,
    grouped_accuracy,
    macro_accuracy,
    paired_cluster_bootstrap,
    percentile,
)


def test_accuracy_and_groups():
    assert accuracy([True, False, True, True]) == 0.75
    assert math.isnan(accuracy([]))
    assert grouped_accuracy(["a", "b", "a"], [True, False, False]) == {"a": (0.5, 2), "b": (0.0, 1)}


def test_macro_weights_databases_equally():
    # db a: 1/1, db b: 0/3 -> question-weighted 0.25, macro 0.5
    dbs, flags = ["a", "b", "b", "b"], [True, False, False, False]
    assert accuracy(flags) == 0.25
    assert macro_accuracy(dbs, flags) == 0.5


def test_percentile_matches_linear_interpolation():
    vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert percentile(vals, 0.0) == 0.0
    assert percentile(vals, 1.0) == 4.0
    assert percentile(vals, 0.5) == 2.0
    assert percentile(vals, 0.1) == pytest.approx(0.4)


def test_bootstrap_identical_models_gives_zero_interval():
    dbs = ["a"] * 5 + ["b"] * 5
    flags = [True, False] * 5
    r = paired_cluster_bootstrap(dbs, flags, flags, n_resamples=500, seed=1)
    assert (r.diff, r.ci_low, r.ci_high) == (0.0, 0.0, 0.0)
    assert not r.excludes_zero
    assert (r.n_questions, r.n_databases) == (10, 2)


def test_bootstrap_uniformly_better_adapter_excludes_zero():
    dbs = [f"d{i % 8}" for i in range(80)]
    base = [i % 4 == 0 for i in range(80)]
    adapter = [True] * 80
    r = paired_cluster_bootstrap(dbs, base, adapter, n_resamples=2000, seed=7)
    assert r.diff == pytest.approx(0.75)
    assert r.ci_low > 0 and r.excludes_zero


def test_bootstrap_is_seeded_and_cluster_based():
    # One database carries the whole improvement: cluster resampling must show wide uncertainty.
    dbs = ["big"] * 20 + [f"s{i}" for i in range(20)]
    base = [False] * 40
    adapter = [True] * 20 + [False] * 20
    r1 = paired_cluster_bootstrap(dbs, base, adapter, n_resamples=2000, seed=3)
    r2 = paired_cluster_bootstrap(dbs, base, adapter, n_resamples=2000, seed=3)
    assert r1 == r2
    assert r1.ci_low == 0.0  # resamples that miss "big" show no improvement


def test_bootstrap_rejects_misaligned():
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(["a"], [True], [True, False], seed=0)
