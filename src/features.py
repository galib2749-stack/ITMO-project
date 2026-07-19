"""Assemble the leakage-safe tabular feature matrix.

See reports/leakage_audit.md and docs/features.md for the rationale behind
every inclusion/exclusion/censoring decision made here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2019-03-18 23:59:59")

CATEGORICAL_FEATURES = ["gender"]

NUMERIC_FEATURES = [
    "age_clean", "age_was_imputed",
    "first_issue_days_ago",
    "has_redeemed_before_cutoff", "redeem_delay_days_censored",
    "n_receipts", "n_items", "n_active_days", "history_span_days",
    "total_spend", "avg_receipt_value", "total_quantity",
    "n_unique_stores", "n_unique_products", "n_unique_categories",
    "total_regular_points_received", "total_express_points_received",
    "total_regular_points_spent", "total_express_points_spent",
    "total_discount", "receipts_per_week", "has_purchase_history",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_client_features(clients_csv_path, client_purchase_features_csv_path, cutoff=CUTOFF):
    clients = pd.read_csv(
        clients_csv_path,
        parse_dates=["first_issue_date", "first_redeem_date"],
    )

    age = clients["age"].astype(float)
    valid_age_mask = (age >= 0) & (age <= 100)
    median_age = age[valid_age_mask].median()
    clients["age_clean"] = np.where(valid_age_mask, age, median_age)
    clients["age_was_imputed"] = (~valid_age_mask).astype(int)

    clients["gender"] = clients["gender"].fillna("U")

    clients["first_issue_days_ago"] = (cutoff - clients["first_issue_date"]).dt.total_seconds() / 86400.0

    redeem_before_cutoff = clients["first_redeem_date"].notna() & (clients["first_redeem_date"] <= cutoff)
    clients["has_redeemed_before_cutoff"] = redeem_before_cutoff.astype(int)
    delay = (clients["first_redeem_date"] - clients["first_issue_date"]).dt.total_seconds() / 86400.0
    clients["redeem_delay_days_censored"] = np.where(redeem_before_cutoff, delay, 0.0)

    purchase_features = pd.read_csv(
        client_purchase_features_csv_path,
        parse_dates=["first_purchase_date", "last_purchase_date"],
    )
    purchase_cols = [c for c in purchase_features.columns if c not in ("client_id", "first_purchase_date", "last_purchase_date")]

    merged = clients.merge(purchase_features, on="client_id", how="left")
    merged["has_purchase_history"] = merged["n_receipts"].notna().astype(int)
    for c in purchase_cols:
        merged[c] = merged[c].fillna(0.0)

    keep_cols = ["client_id"] + ALL_FEATURES
    return merged[keep_cols].copy()


def to_model_matrix(features_df, feature_names=ALL_FEATURES):
    """Return (X, cat_feature_indices) ready for CatBoost, preserving column
    order. Categorical columns are cast to string (CatBoost requirement)."""
    X = features_df[feature_names].copy()
    cat_idx = [feature_names.index(c) for c in CATEGORICAL_FEATURES if c in feature_names]
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype(str)
    return X, cat_idx
