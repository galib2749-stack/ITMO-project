# Experiment Report — X5 RetailHero Uplift Modeling

All numbers below are read directly from `artifacts/metrics.csv` /
`artifacts/validation_sanity_checks.json`, produced by
`src/run_full_pipeline.py` on the real dataset and reproduced by executing
`notebooks/x5_uplift_full_pipeline.ipynb` end to end. Holdout size: 40,011
clients (20% of `uplift_train.csv`, stratified by treatment × target,
disjoint from the train/val splits used to fit every model).

## Final metrics table

| model | AUUC | Qini | 95% Qini CI | uplift@10% | uplift@20% | uplift@30% |
|---|---|---|---|---|---|---|
| x_learner_catboost | 0.01247 | **0.00418** | [0.00307, 0.00531] | 0.1130 | 0.0814 | 0.0755 |
| t_learner_catboost | 0.01200 | 0.00370 | [0.00256, 0.00508] | 0.1020 | 0.0704 | 0.0639 |
| transformer_two_head | 0.00997 | 0.00167 | [0.00034, 0.00290] | 0.0929 | 0.0653 | 0.0481 |
| random_targeting | 0.00903 | 0.00073 | [-0.00079, 0.00211] | 0.0536 | 0.0460 | 0.0374 |
| response_catboost | 0.00681 | **-0.00149** | [-0.00255, -0.00030] | -0.0025 | 0.0109 | 0.0219 |

Ranking is **consistent across all three metrics** (AUUC, Qini, uplift@30%):
X-Learner > T-Learner > Transformer > Random > Response CatBoost.

## Key findings

**1. X-Learner is the best-performing approach** on this holdout, with a
Qini 95% CI ([0.00307, 0.00531]) that excludes zero and does not overlap
zero even at its lower bound — a statistically real ranking signal, not
noise. T-Learner is a close second with a materially overlapping but
slightly lower CI.

**2. Response CatBoost is significantly *worse* than random targeting.**
Its Qini CI ([-0.00255, -0.00030]) is entirely negative. This is the
concrete, measured demonstration (not merely asserted) of the project's
core methodological point: ranking clients by `P(Y=1|X)` targets people who
were going to convert anyway (or who are simply generally high-propensity),
which in this holdout is *anti-correlated* with who actually benefits from
the communication. A retailer using a response model to target
communications would, per this holdout, do worse than randomly assigning
who gets contacted.

**3. The Transformer two-head model clears the random baseline with a
statistically real (CI excludes zero: [0.00034, 0.00290]) but smaller
effect than either tabular causal learner.** Training stopped early at
epoch 7 of a maximum 15 (validation loss plateaued from epoch 4 onward —
see `artifacts/transformer_training_history.csv`), on a deliberately modest
CPU-trainable configuration (`hidden_size=64`, 2 encoder layers, 4 heads).
This is reported as a genuine, unfavorable-to-the-headline-model result: on
this dataset, at this compute budget, sequence modeling does not beat
well-tuned tabular meta-learners on RFM-style aggregates. See Limitations
(Section 19 of the notebook) for what could plausibly change this (larger
hidden size, longer training, learned rather than sinusoidal-style position
handling, longer max sequence length).

**4. Validation sanity checks all behave as expected**
(`artifacts/validation_sanity_checks.json`):
- Empirical ATE (target rate, treatment − control) = **+0.0332**, the
  anchor every model's overall predicted effect is compared against.
- Shuffled treatment, shuffled target, and constant-score all produce Qini
  values within noise of zero (range ±0.0007), as they must if the metric
  implementation is correct.
- The random-score Monte Carlo distribution (50 draws) has mean Qini
  8.4e-05 with std 7.2e-04 — centered on zero, confirming
  `test_random_model_qini_is_centered_near_zero_on_average` (already unit
  tested in `tests/test_uplift_metrics.py`) also holds on the real holdout
  population, not just synthetic toy data.

## What this does and does not prove

These are offline metrics on a historical randomized-experiment holdout.
They demonstrate that X-Learner (and, to a lesser extent, T-Learner and the
Transformer) can *rank* clients by estimated treatment effect meaningfully
better than chance, and that a naive response model actively *should not*
be used for this purpose. They do **not** by themselves prove a prospective
business outcome — see Section 16 (Business evaluation, explicit
scenario-based) and Section 17 (Online A/B-test design) of the notebook for
what would be required before deploying any of this.
