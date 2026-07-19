# Feature Documentation

All features are computed exclusively from information available **before**
the pre-campaign cutoff (2019-03-18, see `reports/leakage_audit.md`). Source
tables: `clients.csv` (static/demographic) and `purchases.csv` (pre-campaign
transaction history, aggregated by `src/build_client_aggregates.py` into
`data/interim/client_purchase_features.csv` and
`data/interim/receipts_sequences.parquet`).

## Tabular features (CatBoost models: response, T-Learner, X-Learner)

| Feature | Source | Definition | Leakage note |
|---|---|---|---|
| `age_clean` | clients.csv | `age` clipped to [0, 100], else imputed with population median | Cleaned for a data-quality issue (raw min -7491, max 1901), not leakage |
| `age_was_imputed` | clients.csv | 1 if `age` was out of [0,100] | — |
| `gender` | clients.csv | categorical M/F/U, passed as a CatBoost categorical feature | Safe, static |
| `first_issue_days_ago` | clients.csv | days between `first_issue_date` and cutoff (2019-03-18) | Safe — max date is 2019-03-15, before cutoff |
| `has_redeemed_before_cutoff` | clients.csv | 1 if `first_redeem_date <= cutoff` | Censored — see leakage audit |
| `redeem_delay_days_censored` | clients.csv | days between `first_issue_date` and `first_redeem_date`, only if `first_redeem_date <= cutoff`, else 0 | Censored — see leakage audit |
| `n_receipts` | purchases.csv (aggregated) | count of distinct transactions before cutoff | Safe |
| `n_items` | purchases.csv | count of product-lines before cutoff | Safe |
| `n_active_days` | purchases.csv | count of distinct calendar days with a purchase | Safe |
| `history_span_days` | purchases.csv | days between first and last purchase in history | Safe |
| `total_spend` | purchases.csv | sum of `purchase_sum` across receipts | Safe |
| `avg_receipt_value` | purchases.csv | mean `purchase_sum` per receipt | Safe |
| `total_quantity` | purchases.csv | sum of `product_quantity` | Safe |
| `n_unique_stores` | purchases.csv | distinct `store_id` visited | Safe |
| `n_unique_products` | purchases.csv | distinct `product_id` purchased | Safe |
| `n_unique_categories` | purchases.csv | distinct `level_2` product category purchased | Safe |
| `total_regular_points_received` / `total_express_points_received` | purchases.csv | sum of loyalty points earned | Safe (pre-cutoff only) |
| `total_regular_points_spent` / `total_express_points_spent` | purchases.csv | sum of loyalty points redeemed | Safe (pre-cutoff only) |
| `total_discount` | purchases.csv | sum of `trn_sum_from_red` | Safe |
| `receipts_per_week` | purchases.csv | `n_receipts / max(history_span_days/7, 1)` | Safe |
| `has_purchase_history` | purchases.csv | 0/1 flag for clients with zero pre-cutoff purchases | Safe |

Missing purchase-history features (client has zero pre-cutoff transactions)
are filled with 0 and flagged via `has_purchase_history`.

## Excluded features (considered and rejected)

| Feature | Reason excluded |
|---|---|
| Raw `first_redeem_date` / `first_redeem_unixtime` (uncensored) | Post-cutoff contamination, see leakage audit |
| Raw `issue_redeem_delay` (uncensored, as used in official baseline) | Same — depends on uncensored `first_redeem_date` |
| Any feature computed from `uplift_test.csv` or from other clients in `uplift_train.csv` (population-level target leakage via target encoding) | Not computed at all in this project — no target-encoded features are used |

## Sequence features (Transformer)

Built from `data/interim/receipts_sequences.parquet`, one row per client
receipt (transaction), ordered chronologically, truncated to the most recent
`MAX_SEQ_LEN` receipts per client (see notebook section "Sequence
construction" for the length-distribution justification of the receipt-level
granularity and the chosen cap).

Per-receipt (per-event) fields fed to the Transformer:

- `store_id` (categorical embedding)
- `n_items`, `n_unique_products`, `n_unique_categories`, `purchase_sum`,
  `total_quantity`, `regular_points_received`, `express_points_received`,
  `regular_points_spent`, `express_points_spent`, `discount_sum`, `iss_sum`
  (numerical, projected)
- `weekday` (categorical embedding)
- `gap_days` (time-gap embedding input)
- position within sequence (positional embedding)

All of the above are computed strictly from `purchases.csv` rows with
`transaction_datetime <= cutoff`, which is guaranteed by construction since
the entire file is pre-cutoff.
