"""Train / validation / final-holdout split and validation sanity checks.

All labeled data comes from uplift_train.csv (the only table with ground-truth
`target`). uplift_test.csv has no labels and is therefore never used for any
metric in this project -- only for an optional bonus submission-style
prediction file. Every model in the comparison table is fit on the SAME
train split and evaluated on the SAME validation/holdout splits, with the
SAME treatment/target definitions and the SAME metric implementations
(src/uplift_metrics.py), per the project's split-integrity requirements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def make_splits(df, train_frac=0.6, val_frac=0.2, seed=RANDOM_SEED):
    """df must have a 'client_id' column. Returns df with an added 'split'
    column in {'train', 'val', 'holdout'}, stratified by treatment_flg x
    target so all three splits have comparable treatment/target balance."""
    assert "treatment_flg" in df.columns and "target" in df.columns
    rng = np.random.RandomState(seed)
    strata = df["treatment_flg"].astype(str) + "_" + df["target"].astype(str)
    split = pd.Series(index=df.index, dtype="object")

    for stratum_value in strata.unique():
        idx = df.index[strata == stratum_value].to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        split.loc[idx[:n_train]] = "train"
        split.loc[idx[n_train:n_train + n_val]] = "val"
        split.loc[idx[n_train + n_val:]] = "holdout"

    out = df.copy()
    out["split"] = split.values
    return out


def empirical_ate(treatment, target):
    treatment = np.asarray(treatment)
    target = np.asarray(target)
    return float(target[treatment == 1].mean() - target[treatment == 0].mean())


def sanity_checks(y_true, treatment, uplift_score, metric_fns, n_perm=50, seed=RANDOM_SEED):
    """Run the required validation sanity checks and return a dict of results:

    - shuffled_treatment: treatment labels randomly permuted (breaks the true
      treatment/outcome relationship) -> real models' metrics should collapse
      towards the "random model" range.
    - shuffled_target: target labels randomly permuted -> same expectation.
    - constant_score: every unit gets the same uplift score (no ranking
      information at all) -> Qini/AUUC should be ~0 by construction, uplift@k
      should be ~0 in expectation (ties broken arbitrarily).
    - random_score: iid random score, repeated n_perm times -> reports the
      empirical distribution (mean/std), used to confirm real models clear
      this noise band.
    - empirical_ate: the actual average treatment effect in the data, used as
      an anchor / plausibility check for the overall scale of uplift produced
      by every model.
    """
    y_true = np.asarray(y_true, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    uplift_score = np.asarray(uplift_score, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(y_true)

    results = {"empirical_ate": empirical_ate(treatment, y_true)}

    shuffled_t = treatment.copy()
    rng.shuffle(shuffled_t)
    results["shuffled_treatment"] = {name: fn(y_true, shuffled_t, uplift_score) for name, fn in metric_fns.items()}

    shuffled_y = y_true.copy()
    rng.shuffle(shuffled_y)
    results["shuffled_target"] = {name: fn(shuffled_y, treatment, uplift_score) for name, fn in metric_fns.items()}

    const_score = np.zeros(n)
    results["constant_score"] = {name: fn(y_true, treatment, const_score) for name, fn in metric_fns.items()}

    random_runs = {name: [] for name in metric_fns}
    for i in range(n_perm):
        rs = np.random.RandomState(seed + 1000 + i).rand(n)
        for name, fn in metric_fns.items():
            random_runs[name].append(fn(y_true, treatment, rs))
    results["random_score"] = {
        name: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_perm": n_perm}
        for name, v in random_runs.items()
    }

    return results
