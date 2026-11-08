"""Accuracy summaries and the paired database-cluster bootstrap."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence


def accuracy(flags: Sequence[bool]) -> float:
    return sum(flags) / len(flags) if flags else float("nan")


def grouped_accuracy(groups: Sequence[Hashable], flags: Sequence[bool]) -> dict[Hashable, tuple[float, int]]:
    acc: dict[Hashable, list[bool]] = defaultdict(list)
    for g, f in zip(groups, flags, strict=True):
        acc[g].append(f)
    return {g: (accuracy(v), len(v)) for g, v in sorted(acc.items(), key=lambda kv: str(kv[0]))}


def macro_accuracy(db_ids: Sequence[str], flags: Sequence[bool]) -> float:
    per_db = grouped_accuracy(db_ids, flags)
    return sum(a for a, _n in per_db.values()) / len(per_db)


def percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default method)."""
    if not sorted_vals:
        raise ValueError("empty")
    pos = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


@dataclass(frozen=True)
class BootstrapResult:
    baseline_acc: float
    adapter_acc: float
    diff: float
    ci_low: float
    ci_high: float
    n_questions: int
    n_databases: int
    n_resamples: int
    seed: int

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def paired_cluster_bootstrap(
    db_ids: Sequence[str],
    baseline: Sequence[bool],
    adapter: Sequence[bool],
    *,
    n_resamples: int = 10_000,
    seed: int,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Resample databases with replacement; each draw brings all its questions.

    The statistic is the question-weighted paired difference adapter - baseline.
    """
    if not (len(db_ids) == len(baseline) == len(adapter)) or not db_ids:
        raise ValueError("inputs must be non-empty and aligned")
    clusters: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # n, base_correct, adapter_correct
    for d, b, a in zip(db_ids, baseline, adapter):
        c = clusters[d]
        c[0] += 1
        c[1] += int(b)
        c[2] += int(a)
    keys = sorted(clusters)
    stats = [clusters[k] for k in keys]

    rng = random.Random(seed)
    diffs = []
    k = len(stats)
    for _ in range(n_resamples):
        n = base = adp = 0
        for idx in rng.choices(range(k), k=k):
            s = stats[idx]
            n += s[0]
            base += s[1]
            adp += s[2]
        diffs.append((adp - base) / n)
    diffs.sort()

    n_total = len(db_ids)
    b_acc = sum(map(int, baseline)) / n_total
    a_acc = sum(map(int, adapter)) / n_total
    return BootstrapResult(
        baseline_acc=b_acc, adapter_acc=a_acc, diff=a_acc - b_acc,
        ci_low=percentile(diffs, alpha / 2), ci_high=percentile(diffs, 1 - alpha / 2),
        n_questions=n_total, n_databases=k, n_resamples=n_resamples, seed=seed,
    )
