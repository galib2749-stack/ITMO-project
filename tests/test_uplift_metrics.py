"""Known-answer / sanity tests for src/uplift_metrics.py.

Run with: pytest tests/test_uplift_metrics.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uplift_metrics import (  # noqa: E402
    auuc,
    bootstrap_ci,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
    uplift_by_decile,
)


def make_perfect_uplift_data(n_per_group=1000, seed=0):
    """Construct data where treatment truly helps exactly the first half of
    each group's population and the score perfectly identifies who those are.
    This gives a known-sign, strictly-positive-uplift ground truth."""
    rng = np.random.RandomState(seed)
    n = n_per_group * 2
    treatment = np.array([1] * n_per_group + [0] * n_per_group)
    rng.shuffle(treatment)

    # "true_responder" = clients who would only convert if treated (persuadables)
    true_responder = np.zeros(n, dtype=int)
    true_responder[: n // 2] = 1
    rng.shuffle(true_responder)

    baseline_conversion = 0.1
    y = (rng.rand(n) < baseline_conversion).astype(int)
    # persuadables convert when treated, on top of baseline
    persuaded = (true_responder == 1) & (treatment == 1)
    y[persuaded] = 1

    # perfect score = true_responder (ties broken by noise)
    score = true_responder.astype(float) + rng.rand(n) * 1e-6
    return y, treatment, score, true_responder


def test_uplift_at_k_matches_reference_formula():
    rng = np.random.RandomState(1)
    n = 500
    y = (rng.rand(n) < 0.3).astype(int)
    t = (rng.rand(n) < 0.5).astype(int)
    score = rng.rand(n)

    result = uplift_at_k(y, t, score, k=0.3)

    order = np.argsort(-score)
    y_sorted, t_sorted = y[order], t[order]
    treatment_n = int((t == 1).sum() * 0.3)
    treatment_p = y_sorted[t_sorted == 1][:treatment_n].mean()
    control_n = int((t == 0).sum() * 0.3)
    control_p = y_sorted[t_sorted == 0][:control_n].mean()
    expected = treatment_p - control_p

    assert result == pytest.approx(expected)


def test_perfect_model_has_positive_qini_and_auuc():
    y, t, score, _ = make_perfect_uplift_data()
    qc = qini_coefficient(y, t, score)
    a = auuc(y, t, score)
    assert qc > 0
    assert a > 0


def test_perfect_model_beats_random_model():
    y, t, score, _ = make_perfect_uplift_data(seed=2)
    rng = np.random.RandomState(3)
    random_score = rng.rand(len(y))

    qc_perfect = qini_coefficient(y, t, score)
    qc_random = qini_coefficient(y, t, random_score)
    assert qc_perfect > qc_random


def test_random_model_qini_is_centered_near_zero_on_average():
    """A single random permutation can be positive or negative (sampling
    noise around the theoretical random-targeting line), but the *average*
    Qini coefficient over many independent random scorings must be close to
    zero. This directly documents/validates the "why can random Qini be
    negative" behaviour requested in the project spec."""
    rng = np.random.RandomState(4)
    n = 2000
    y = (rng.rand(n) < 0.2).astype(int)
    t = (rng.rand(n) < 0.5).astype(int)

    values = []
    for i in range(200):
        score = np.random.RandomState(100 + i).rand(n)
        values.append(qini_coefficient(y, t, score))
    values = np.array(values)

    assert values.mean() == pytest.approx(0.0, abs=0.01)
    # confirm some individual draws ARE negative -- this is expected, not a bug
    assert (values < 0).sum() > 0
    assert (values > 0).sum() > 0


def test_inverted_model_has_negative_qini():
    y, t, score, _ = make_perfect_uplift_data(seed=5)
    inverted_score = -score
    qc = qini_coefficient(y, t, inverted_score)
    assert qc < 0


def test_uplift_by_decile_shape_and_monotonic_for_perfect_model():
    y, t, score, _ = make_perfect_uplift_data(seed=6)
    rows = uplift_by_decile(y, t, score)
    assert len(rows) == 10
    assert sum(r["n"] for r in rows) == len(y)
    uplifts = [r["uplift"] for r in rows]
    # top decile (persuadables ranked first) should show higher uplift than bottom decile
    assert uplifts[0] > uplifts[-1]


def test_bootstrap_ci_contains_point_estimate_and_has_positive_width():
    y, t, score, _ = make_perfect_uplift_data(seed=7)
    point, lo, hi = bootstrap_ci(qini_coefficient, y, t, score, n_boot=100, random_state=42)
    assert lo <= point <= hi
    assert hi > lo


def test_qini_curve_endpoints():
    y, t, score, _ = make_perfect_uplift_data(seed=8)
    fractions, qini_values, random_values = qini_curve(y, t, score, n_points=10)
    assert fractions[-1] == pytest.approx(1.0)
    # at full population, qini curve value equals the random reference (both are qini(N))
    assert qini_values[-1] == pytest.approx(random_values[-1])


def test_uplift_at_k_zero_for_identical_treatment_and_control_distributions():
    """If treatment has no effect at all (target independent of treatment),
    uplift@k should be close to zero regardless of score."""
    rng = np.random.RandomState(9)
    n = 5000
    t = (rng.rand(n) < 0.5).astype(int)
    y = (rng.rand(n) < 0.25).astype(int)  # independent of t
    score = rng.rand(n)
    result = uplift_at_k(y, t, score, k=0.3)
    assert result == pytest.approx(0.0, abs=0.05)
