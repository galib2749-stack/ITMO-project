"""Uplift evaluation metrics: Qini curve/coefficient, AUUC, uplift@k, bootstrap CIs.

All functions take three parallel 1-D arrays for a scored population:
  y_true         -- observed binary outcome (0/1)
  treatment      -- binary treatment flag (1 = treated/communicated, 0 = control)
  uplift_score   -- model's predicted uplift (higher = target more)

Conventions (documented explicitly because sign/normalization conventions vary
across the uplift-modeling literature and libraries):

Qini curve
----------
Individuals are ranked by descending uplift_score. At each prefix of size k
(k = 1..N, evaluated at the population fractions in `qini_curve`'s `fractions`
grid), let n_t/n_c be the number of treated/control units in that prefix and
y_t/y_c the summed observed outcome for each group in that prefix. Then

    qini(k) = y_t(k) - y_c(k) * n_t(k) / n_c(k)      (n_c(k) > 0)
    qini(k) = y_t(k)                                  (n_c(k) == 0)

This is the standard Radcliffe (2007) Qini curve: it rescales the control
group's cumulative response to the size of the treatment group in the same
prefix, so it is a fair estimate of "incremental responders in the top k"
even when treatment/control are imbalanced.

The random-targeting reference curve is the straight line from (0, 0) to
(N, qini(N)) -- i.e. what you'd expect from a model with no ranking ability
(uplift accrues linearly with population fraction).

Qini coefficient = area under qini(k) - area under the random reference line,
both by the trapezoidal rule over k = 0..N, normalized by N so the number is
a per-capita quantity independent of population size.

AUUC (area under the uplift curve) here is defined as the raw (non
baseline-subtracted) trapezoidal area under qini(k)/N over k/N in [0, 1] --
i.e. the average cumulative incremental-outcome curve height. It is reported
alongside the Qini coefficient because the two answer slightly different
questions: Qini coefficient measures ranking quality *relative to random*,
AUUC measures the absolute average incremental yield of the ranking.

Why a random model's Qini coefficient is not exactly zero
----------------------------------------------------------
Qini coefficient is defined relative to the *theoretical* random-targeting
line. Any *actual* random permutation of a finite sample is a single noisy
draw around that line, so its empirical Qini coefficient fluctuates around 0
and can be slightly negative or positive purely from sampling variance. It
converges to 0 in expectation as N -> inf. This is verified in
tests/test_uplift_metrics.py by checking the empirical distribution over many
random permutations is centered near zero, not that any single draw is
exactly zero.
"""
from __future__ import annotations

import numpy as np


def _as_arrays(y_true, treatment, uplift_score):
    y_true = np.asarray(y_true, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    uplift_score = np.asarray(uplift_score, dtype=float)
    if not (len(y_true) == len(treatment) == len(uplift_score)):
        raise ValueError("y_true, treatment, uplift_score must be the same length")
    return y_true, treatment, uplift_score


def qini_curve(y_true, treatment, uplift_score, n_points=100):
    """Return (fractions, qini_values, random_reference_values).

    fractions: array of population fractions in (0, 1], length n_points.
    qini_values: qini(k) evaluated at those fractions (see module docstring).
    random_reference_values: the straight-line random-targeting reference at
    the same fractions.
    """
    y_true, treatment, uplift_score = _as_arrays(y_true, treatment, uplift_score)
    n = len(y_true)
    order = np.argsort(-uplift_score, kind="mergesort")
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1 - t_sorted)
    cum_yt = np.cumsum(y_sorted * t_sorted)
    cum_yc = np.cumsum(y_sorted * (1 - t_sorted))

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(cum_c > 0, cum_t / np.where(cum_c > 0, cum_c, 1), 0.0)
    qini_all_k = cum_yt - cum_yc * ratio

    fractions = np.linspace(1.0 / n_points, 1.0, n_points)
    idx = np.clip(np.round(fractions * n).astype(int) - 1, 0, n - 1)
    qini_values = qini_all_k[idx]
    qini_at_n = qini_all_k[-1]
    random_reference_values = fractions * qini_at_n
    return fractions, qini_values, random_reference_values


def qini_coefficient(y_true, treatment, uplift_score, n_points=1000):
    """Qini coefficient: area under qini curve minus area under the random
    reference line, trapezoidal rule, normalized per-capita (divided by N)."""
    y_true, treatment, uplift_score = _as_arrays(y_true, treatment, uplift_score)
    n = len(y_true)
    fractions, qini_values, random_values = qini_curve(y_true, treatment, uplift_score, n_points=n_points)
    x = np.concatenate([[0.0], fractions])
    qini_y = np.concatenate([[0.0], qini_values])
    rand_y = np.concatenate([[0.0], random_values])
    area_qini = np.trapezoid(qini_y, x)
    area_random = np.trapezoid(rand_y, x)
    return float((area_qini - area_random) / n)


def auuc(y_true, treatment, uplift_score, n_points=1000):
    """Area under the (raw, non-baseline-subtracted) qini curve, per-capita."""
    y_true, treatment, uplift_score = _as_arrays(y_true, treatment, uplift_score)
    n = len(y_true)
    fractions, qini_values, _ = qini_curve(y_true, treatment, uplift_score, n_points=n_points)
    x = np.concatenate([[0.0], fractions])
    qini_y = np.concatenate([[0.0], qini_values])
    area = np.trapezoid(qini_y, x)
    return float(area / n)


def uplift_at_k(y_true, treatment, uplift_score, k=0.3):
    """Official X5 RetailHero contest metric (see uplift_solution.py):

    Take the top `k` fraction of the TREATED units by score and the top `k`
    fraction of the CONTROL units by score (each ranked/sliced within its own
    group), and return mean(target | top-k treated) - mean(target | top-k control).
    """
    y_true, treatment, uplift_score = _as_arrays(y_true, treatment, uplift_score)
    order = np.argsort(-uplift_score, kind="mergesort")
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    treatment_mask = t_sorted == 1
    control_mask = t_sorted == 0
    n_t = int(treatment_mask.sum() * k)
    n_c = int(control_mask.sum() * k)

    y_treat_topk = y_sorted[treatment_mask][:n_t]
    y_ctrl_topk = y_sorted[control_mask][:n_c]

    treat_rate = float(y_treat_topk.mean()) if n_t > 0 else float("nan")
    ctrl_rate = float(y_ctrl_topk.mean()) if n_c > 0 else float("nan")
    return treat_rate - ctrl_rate


def uplift_by_decile(y_true, treatment, uplift_score):
    """Return a DataFrame-like list of dicts: for each decile of population
    (ranked by descending score), the treatment rate, control rate, and their
    difference (observed uplift) within that decile only (not cumulative)."""
    y_true, treatment, uplift_score = _as_arrays(y_true, treatment, uplift_score)
    n = len(y_true)
    order = np.argsort(-uplift_score, kind="mergesort")
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    bounds = np.linspace(0, n, 11).astype(int)
    rows = []
    for d in range(10):
        lo, hi = bounds[d], bounds[d + 1]
        y_slice = y_sorted[lo:hi]
        t_slice = t_sorted[lo:hi]
        t_rate = float(y_slice[t_slice == 1].mean()) if (t_slice == 1).any() else float("nan")
        c_rate = float(y_slice[t_slice == 0].mean()) if (t_slice == 0).any() else float("nan")
        rows.append({
            "decile": d + 1,
            "n": int(hi - lo),
            "n_treatment": int((t_slice == 1).sum()),
            "n_control": int((t_slice == 0).sum()),
            "treatment_rate": t_rate,
            "control_rate": c_rate,
            "uplift": t_rate - c_rate,
        })
    return rows


def bootstrap_ci(metric_fn, y_true, treatment, uplift_score, n_boot=200, alpha=0.05, random_state=0):
    """Percentile bootstrap CI for any metric_fn(y_true, treatment, uplift_score).

    Resamples individuals with replacement (keeping the (y, t, score) triple
    together) n_boot times and returns (point_estimate, ci_low, ci_high).
    """
    y_true, treatment, uplift_score = _as_arrays(y_true, treatment, uplift_score)
    n = len(y_true)
    rng = np.random.RandomState(random_state)
    point = metric_fn(y_true, treatment, uplift_score)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        stats[b] = metric_fn(y_true[idx], treatment[idx], uplift_score[idx])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return point, lo, hi
