# Data Download Report — X5 RetailHero Uplift Modeling

## Source

- Competition / data description page: https://ods.ai/competitions/x5-retailhero-uplift-modeling/data
- Original contest platform referenced inside the data package itself: https://retailhero.ai/c/uplift_modeling/
- The dataset archive was placed by the user directly into the working directory
  (`retailhero-uplift.zip`, 625,190,754 bytes) prior to this session, per the project's
  Critical Rule #4 fallback path (manual placement when automated download requires
  authentication). This report documents verification of that archive, not an
  automated fetch — ODS.ai gates the direct data-file download behind an authenticated
  account, so no unauthenticated automated download was possible from this environment.

## Download / placement date

- File timestamps on disk: 2026-07-19 (see `ls -la` timestamps captured during verification).
- Verified and catalogued in this session: 2026-07-19.

## Archive contents

Extracted from `retailhero-uplift.zip` into `data/raw/x5_retailhero/`:

```
data/raw/x5_retailhero/
├── README                         (1,035 bytes)
├── requirements.txt               (49 bytes)
├── uplift_solution.py             (4,233 bytes)  — official example solution script
├── retailhero-uplift.zip          (625,190,754 bytes) — original archive, kept for provenance
└── data/
    ├── clients.csv                (21,736,219 bytes / 400,163 lines incl. header)
    ├── products.csv               (3,890,339 bytes / 43,039 lines incl. header)
    ├── purchases.csv              (4,463,775,504 bytes / ~45.8M lines, transaction-level log)
    ├── uplift_train.csv           (3,000,616 bytes / 200,040 lines incl. header)
    ├── uplift_test.csv            (2,201,363 bytes / 200,124 lines incl. header)
    └── uplift_sample_submission.csv (6,057,851 bytes / 200,124 lines incl. header)
```

## Checksums (SHA-256)

```
866ace4ff511aa3ad548efba204b61c48048f020ba327616fafc2939c1e00362  retailhero-uplift.zip
0135e7a3dfb2174c0fff4033cc895c070794e3c19bf5a6b2a685f4acde32e7ba  data/clients.csv
c174d8394d528c463e0fedfa5ba6de20e27f47fb3249a91b3dffda5c98f30f9e  data/products.csv
a9e132e3dde95655a0074622bb93af0616b7c4f047e71c4ce7322e2fe6d262c1  data/purchases.csv
ae3dc4c17e98426fa4233fa03421328d259b8720e56ad0380b260ca2edd19bd4  data/uplift_train.csv
55c4bdc210ed71c94fddc4472e8b1ac37dc17d8602ebd48dbc03aaf462e7cf60  data/uplift_test.csv
c9e16b3ee3de1807647b1e9220998218bf93a6ca57a9727de6676b0220d4bc85  data/uplift_sample_submission.csv
```

Full checksum log: `data/raw/x5_retailhero/checksums.sha256.txt`.

## Structural validation

The README bundled inside the archive is in Russian and self-identifies the dataset:

> RetailHero Uplift Modeling — Задача на uplift-моделирование. Необходимо
> отранжировать клиентов по убыванию эффективности коммуникации.
> Страница соревнования: https://retailhero.ai/c/uplift_modeling/

This matches the known X5 RetailHero uplift contest exactly (client-level SMS
communication uplift task, X5 Retail Group loyalty data).

Column headers confirmed by direct inspection (not assumed):

- `clients.csv`: `client_id, first_issue_date, first_redeem_date, age, gender`
- `products.csv`: `product_id, level_1, level_2, level_3, level_4, segment_id, brand_id, vendor_id, netto, is_own_trademark, is_alcohol`
- `purchases.csv`: `client_id, transaction_id, transaction_datetime, regular_points_received, express_points_received, regular_points_spent, express_points_spent, purchase_sum, store_id, product_id, product_quantity, trn_sum_from_iss, trn_sum_from_red`
- `uplift_train.csv`: `client_id, treatment_flg, target`
- `uplift_test.csv`: `client_id`
- `uplift_sample_submission.csv`: `client_id, uplift`

Presence check against Critical Rule requirements:

| Required element | Present | Where |
|---|---|---|
| Client identifier | Yes | `client_id` in all tables |
| Purchase history | Yes | `purchases.csv`, one row per (transaction, product) line, pre-campaign history |
| Treatment / control | Yes | `uplift_train.csv.treatment_flg` (binary) |
| Target | Yes | `uplift_train.csv.target` (binary, post-campaign purchase indicator) |

File-type check: all files opened as plain UTF-8 CSV text with the expected
comma-delimited header row on the first line — none are HTML, none are error
pages, none are empty placeholders.

## Status

**VALIDATED.** Real X5 RetailHero data is present, checksummed, and structurally
verified. Final training and reporting may proceed on this data per Critical Rule #2.
