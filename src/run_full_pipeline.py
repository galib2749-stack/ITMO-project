"""End-to-end pipeline: load real X5 data -> leakage-safe features -> splits
-> fit all 5 required approaches -> evaluate on the SAME holdout with the
SAME metrics -> save artifacts/metrics.csv, metrics.json, per-model predictions,
qini curve data, and uplift-by-decile tables.

This is the canonical pipeline; notebooks/x5_uplift_full_pipeline.ipynb calls
into this same src/ code (not a re-implementation) so the notebook narrative
and this script can never silently diverge.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from features import build_client_features, to_model_matrix, ALL_FEATURES  # noqa: E402
from splits import make_splits, sanity_checks, RANDOM_SEED  # noqa: E402
from models_classical import random_targeting_score, ResponseModel, TLearner, XLearner  # noqa: E402
from sequences import (  # noqa: E402
    load_receipts, fit_numeric_scaler, build_client_sequences, StoreVocab,
    NUMERIC_EVENT_FIELDS, MAX_SEQ_LEN,
)
from model_transformer import (  # noqa: E402
    ReceiptSequenceDataset, TwoHeadTransformer, train_transformer, predict_uplift,
)
from uplift_metrics import qini_coefficient, auuc, uplift_at_k, uplift_by_decile, bootstrap_ci  # noqa: E402

DATA_DIR = "data/raw/x5_retailhero/data"
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts"
FIGURES_DIR = "reports/figures"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(transformer_epochs=15, transformer_batch_size=512, seed=RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    log("Loading labels...")
    train_labels = pd.read_csv(f"{DATA_DIR}/uplift_train.csv")
    split_df = make_splits(train_labels, seed=seed)
    split_df.to_csv(f"{PROCESSED_DIR}/train_val_holdout_split.csv", index=False)
    log(f"Split sizes: {split_df['split'].value_counts().to_dict()}")

    log("Building tabular features...")
    feats = build_client_features(f"{DATA_DIR}/clients.csv", f"{INTERIM_DIR}/client_purchase_features.csv")
    df = split_df.merge(feats, on="client_id", how="left")
    X_all, cat_idx = to_model_matrix(df, ALL_FEATURES)

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    holdout_mask = (df["split"] == "holdout").to_numpy()

    treatment = df["treatment_flg"].to_numpy()
    target = df["target"].to_numpy()

    X_train, t_train, y_train = X_all[train_mask], treatment[train_mask], target[train_mask]
    X_val, t_val, y_val = X_all[val_mask], treatment[val_mask], target[val_mask]
    X_holdout, t_holdout, y_holdout = X_all[holdout_mask], treatment[holdout_mask], target[holdout_mask]

    cat_features = [ALL_FEATURES[i] for i in cat_idx]
    log(f"Tabular feature matrix ready: train={X_train.shape} val={X_val.shape} holdout={X_holdout.shape}")

    predictions = {}

    log("Model 0: random_targeting")
    predictions["random_targeting"] = random_targeting_score(len(y_holdout), random_state=seed)

    log("Model 1: response_catboost")
    resp = ResponseModel(random_state=seed, cat_features=cat_features)
    resp.fit(X_train, y_train)
    predictions["response_catboost"] = resp.predict_uplift(X_holdout)

    log("Model 2: t_learner_catboost")
    t_learner = TLearner(random_state=seed, cat_features=cat_features)
    t_learner.fit(X_train, t_train, y_train)
    predictions["t_learner_catboost"] = t_learner.predict_uplift(X_holdout)

    log("Model 3: x_learner_catboost")
    x_learner = XLearner(random_state=seed, cat_features=cat_features, propensity=0.5)
    x_learner.fit(X_train, t_train, y_train)
    predictions["x_learner_catboost"] = x_learner.predict_uplift(X_holdout)

    log("Model 4: transformer_two_head -- building sequences...")
    client_ids = df["client_id"].tolist()
    receipts_df = load_receipts(f"{INTERIM_DIR}/receipts_sequences.parquet", client_ids=client_ids)

    train_client_ids = df.loc[train_mask, "client_id"].tolist()
    train_receipts_only = receipts_df[receipts_df["client_id"].isin(set(train_client_ids))]
    vocab = StoreVocab(receipts_df["store_id"])
    numeric_stats = fit_numeric_scaler(train_receipts_only)

    log("Building padded tensors for train/val/holdout...")
    tr_tensors = build_client_sequences(receipts_df, train_client_ids, vocab, numeric_stats)
    val_client_ids = df.loc[val_mask, "client_id"].tolist()
    ho_client_ids = df.loc[holdout_mask, "client_id"].tolist()
    va_tensors = build_client_sequences(receipts_df, val_client_ids, vocab, numeric_stats)
    ho_tensors = build_client_sequences(receipts_df, ho_client_ids, vocab, numeric_stats)

    tr_ds = ReceiptSequenceDataset(tr_tensors, t_train, y_train)
    va_ds = ReceiptSequenceDataset(va_tensors, t_val, y_val)
    ho_ds = ReceiptSequenceDataset(ho_tensors, t_holdout, y_holdout)

    tr_loader = DataLoader(tr_ds, batch_size=transformer_batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=1024, shuffle=False, num_workers=0)
    ho_loader = DataLoader(ho_ds, batch_size=1024, shuffle=False, num_workers=0)

    model = TwoHeadTransformer(n_stores=vocab.size, n_numeric_fields=len(NUMERIC_EVENT_FIELDS),
                                max_seq_len=MAX_SEQ_LEN, hidden_size=64, n_heads=4, n_layers=2, dropout=0.2)

    log(f"Training transformer for up to {transformer_epochs} epochs...")
    model, history = train_transformer(model, tr_loader, va_loader, epochs=transformer_epochs,
                                        patience=3, weight_decay=1e-4, grad_clip=1.0)
    pd.DataFrame(history).to_csv(f"{ARTIFACTS_DIR}/transformer_training_history.csv", index=False)
    torch.save(model.state_dict(), f"{ARTIFACTS_DIR}/transformer_model.pt")
    config = {
        "n_stores": vocab.size, "n_numeric_fields": len(NUMERIC_EVENT_FIELDS),
        "max_seq_len": MAX_SEQ_LEN, "hidden_size": 64, "n_heads": 4, "n_layers": 2,
        "dropout": 0.2, "numeric_event_fields": NUMERIC_EVENT_FIELDS,
        "random_seed": seed, "epochs_trained": len(history),
    }
    with open(f"{ARTIFACTS_DIR}/transformer_config.json", "w") as f:
        json.dump(config, f, indent=2)

    uplift_ho, mu0_ho, mu1_ho = predict_uplift(model, ho_loader)
    predictions["transformer_two_head"] = uplift_ho

    # -------- evaluate all models on the SAME holdout --------
    log("Evaluating all models on the shared holdout...")
    rows = []
    qini_curves = {}
    decile_tables = {}
    for name, score in predictions.items():
        qc, qc_lo, qc_hi = bootstrap_ci(qini_coefficient, y_holdout, t_holdout, score, n_boot=200, random_state=seed)
        a, a_lo, a_hi = bootstrap_ci(auuc, y_holdout, t_holdout, score, n_boot=200, random_state=seed)
        u10 = uplift_at_k(y_holdout, t_holdout, score, 0.10)
        u20 = uplift_at_k(y_holdout, t_holdout, score, 0.20)
        u30 = uplift_at_k(y_holdout, t_holdout, score, 0.30)
        rows.append({
            "model": name, "auuc": a, "auuc_ci_low": a_lo, "auuc_ci_high": a_hi,
            "qini": qc, "qini_ci_low": qc_lo, "qini_ci_high": qc_hi,
            "uplift_at_10": u10, "uplift_at_20": u20, "uplift_at_30": u30,
            "n_holdout": len(y_holdout),
        })
        from uplift_metrics import qini_curve
        fractions, qini_vals, rand_vals = qini_curve(y_holdout, t_holdout, score, n_points=100)
        qini_curves[name] = {"fractions": fractions.tolist(), "qini": qini_vals.tolist(), "random": rand_vals.tolist()}
        decile_tables[name] = uplift_by_decile(y_holdout, t_holdout, score)
        log(f"  {name}: qini={qc:.4f} auuc={a:.4f} uplift@30={u30:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(f"{ARTIFACTS_DIR}/metrics.csv", index=False)
    with open(f"{ARTIFACTS_DIR}/metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(f"{ARTIFACTS_DIR}/qini_curves.json", "w") as f:
        json.dump(qini_curves, f, indent=2)
    with open(f"{ARTIFACTS_DIR}/uplift_by_decile.json", "w") as f:
        json.dump(decile_tables, f, indent=2)

    for name, score in predictions.items():
        np.save(f"{PROCESSED_DIR}/holdout_scores_{name}.npy", score)
    np.save(f"{PROCESSED_DIR}/holdout_treatment.npy", t_holdout)
    np.save(f"{PROCESSED_DIR}/holdout_target.npy", y_holdout)

    # -------- validation sanity checks (on t_learner as representative model) --------
    log("Running validation sanity checks...")
    metric_fns = {
        "qini": qini_coefficient, "auuc": auuc,
        "uplift_at_10": lambda y, t, s: uplift_at_k(y, t, s, 0.10),
        "uplift_at_30": lambda y, t, s: uplift_at_k(y, t, s, 0.30),
    }
    sanity = sanity_checks(y_holdout, t_holdout, predictions["t_learner_catboost"], metric_fns, seed=seed)
    with open(f"{ARTIFACTS_DIR}/validation_sanity_checks.json", "w") as f:
        json.dump(sanity, f, indent=2)

    log("Pipeline complete.")
    return metrics_df


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    df = main(transformer_epochs=epochs)
    print(df.to_string())
