# Leakage Audit — X5 RetailHero Uplift Modeling

## Reference cutoff established

`purchases.csv` is documented by the official README as "история покупок
клиентов до смс кампании" (purchase history *before* the SMS campaign). Its
transactions span **2018-11-21 21:02:33** to **2019-03-18 23:40:03**, with a
sharp spike of clients' *last* purchase landing exactly on 2019-03-18
(70,273 of 400,162 clients — far more than any neighboring day). This is
treated as the operational **pre-campaign cutoff: 2019-03-18**. Any
feature that can encode information dated after this point is a leakage risk
for predicting `target` (which is measured in the post-communication window).

## Finding: `first_redeem_date` (clients.csv) is contaminated by post-cutoff events

- `first_redeem_date` range in `clients.csv`: **2017-04-11 → 2019-11-20** —
  i.e. it extends **8 months past** the purchase-history cutoff.
- **44,953 of 400,162 clients (11.23%)** have `first_redeem_date` strictly
  after the 2019-03-18 cutoff.
- Within `uplift_train.csv`, target rate splits sharply on whether
  `first_redeem_date` is populated at all:
  - `first_redeem_date` present: target rate **0.644**
  - `first_redeem_date` null (never redeemed as of data extraction): target
    rate **0.364**
  - This 28-percentage-point gap is far larger than the entire treatment
    effect (~3.3pp naive ATE, see below), and the field's timestamp range
    proves at least part of it is populated by events *after* the campaign
    window — i.e. `first_redeem_date` (and any date-derived quantity built
    from it, such as the official example solution's `first_redeem_unixtime`
    / `issue_redeem_delay`) can be set by the same redemption event that also
    contributes to `target`. Using it raw is direct post-outcome leakage.

**Decision: `first_redeem_date` is used only in censored form.** We derive
`has_redeemed_before_cutoff = 1{first_redeem_date <= 2019-03-18}` and
`redeem_delay_days_before_cutoff` (days between `first_issue_date` and
`first_redeem_date`, computed **only** when `first_redeem_date <= cutoff`,
else set to missing/0 with a separate "unknown" indicator). This keeps the
genuinely pre-treatment signal (an engaged client who redeemed early) while
removing the post-cutoff contamination. This directly implements the audit
requirement to check `redeem_delay_days` / `has_redeem` for leakage before
using them.

The official `uplift_solution.py` baseline shipped with the dataset uses
`first_redeem_unixtime` and `issue_redeem_delay` **without this censoring** —
this project intentionally does not copy that baseline's feature engineering
as-is for this reason.

## Fields checked and cleared

| Field | Risk considered | Verdict |
|---|---|---|
| `first_issue_date` | Could postdate cutoff | Max value 2019-03-15, before cutoff. Safe. |
| `age` | N/A for leakage, but data quality issue found (see below) | Not leakage; needs cleaning |
| `gender` | Static demographic | Safe |
| All `purchases.csv` fields | By construction, pre-campaign only (see cutoff derivation above) | Safe |
| `treatment_flg`, `target` | These ARE the experiment design / outcome, not features | N/A — used only as labels, never as model inputs |

## Data quality issue (not leakage, but must be handled before modeling)

`age` contains implausible values: min **-7491**, max **1901**, with 138
values ≤ 0 and 1,049 values > 100. These are almost certainly data entry /
encoding errors (not leakage). Handling: clip `age` to a plausible range
(0–100) and treat out-of-range values as missing, imputed with the
population median, with a `age_was_imputed` flag feature.

## Balance / integrity checks

- Treatment/control split in `uplift_train.csv`: 49.98% / 50.02% — balanced
  by design (randomized experiment).
- Naive ATE (target rate treatment − target rate control): 0.636511 −
  0.603280 = **+0.0332** (3.3 percentage points), consistent with the
  publicly known scale of effect for this contest and used as a sanity
  anchor for all uplift models' overall predicted effect.
- No client_id overlap between `uplift_train.csv` and `uplift_test.csv`
  (0 shared ids) — confirms train/test are disjoint client populations, so a
  held-out validation split constructed from `uplift_train.csv` is
  appropriate and does not need to worry about `uplift_test.csv` contamination.
- 100% of `uplift_train.csv` and `uplift_test.csv` client_ids are present in
  `clients.csv` (no orphan ids in the label tables).
- Purchases coverage: of 400,162 clients in `clients.csv`, all are present in
  the purchases-derived aggregate table (`data/interim/client_purchase_features.csv`);
  clients with zero purchase history are represented as all-zero /
  missing-flagged purchase features rather than dropped.

## Conclusion

With `first_redeem_date` censored at the pre-campaign cutoff and `age`
cleaned, no remaining feature in the safe feature set (documented in
`docs/features.md`) is derived from information dated after 2019-03-18. All
purchase-history-derived features (RFM aggregates and the receipt sequences
fed to the Transformer) are computed exclusively from `purchases.csv`, which
is pre-cutoff by construction.
