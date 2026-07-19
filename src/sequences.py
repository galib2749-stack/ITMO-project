"""Build padded receipt-sequence tensors for the Transformer model from
data/interim/receipts_sequences.parquet.

Sequence unit: RECEIPT (transaction), chosen over item-level or day-level
after comparing length distributions (see data/interim/sequence_length_summary.json
and the notebook's "Sequence construction" section): item-level histories are
far too long (p95=312, max=2513) for a Transformer with a reasonable context
window, while receipt-level (p95=54, max=320) and day-level (p95=46, max=116)
are both compact. Receipt-level is preferred over day-level because it keeps
one real shopping-trip granularity (store, basket composition, discount)
rather than merging multiple same-day trips into one artificial event.

MAX_SEQ_LEN=50 covers the ~90th percentile of clients' receipt histories
(p90=42); longer histories are truncated to the most recent 50 receipts
(most predictive / most recent behavior kept, oldest dropped).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MAX_SEQ_LEN = 50

NUMERIC_EVENT_FIELDS = [
    "n_items", "n_unique_products", "n_unique_categories", "purchase_sum",
    "total_quantity", "regular_points_received", "express_points_received",
    "regular_points_spent", "express_points_spent", "discount_sum", "iss_sum",
]


class StoreVocab:
    def __init__(self, store_ids):
        uniques = sorted(pd.unique(store_ids))
        self.stoi = {s: i + 1 for i, s in enumerate(uniques)}  # 0 reserved for padding/unknown
        self.size = len(uniques) + 1

    def encode(self, store_ids):
        return np.array([self.stoi.get(s, 0) for s in store_ids], dtype=np.int32)


def load_receipts(parquet_path, client_ids=None):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    if client_ids is not None:
        client_ids = set(client_ids)
        df = df[df["client_id"].isin(client_ids)]
    return df


def fit_numeric_scaler(df, fields=NUMERIC_EVENT_FIELDS):
    """Simple robust scaler: median / (p95 - p5), fit on the given df (should
    be TRAIN split receipts only, to avoid any val/holdout leakage into
    normalization statistics)."""
    stats = {}
    for f in fields:
        vals = df[f].to_numpy(dtype=np.float64)
        median = np.median(vals)
        p95 = np.percentile(vals, 95)
        p5 = np.percentile(vals, 5)
        scale = max(p95 - p5, 1e-6)
        stats[f] = (median, scale)
    return stats


def build_client_sequences(df, client_ids_ordered, store_vocab, numeric_stats,
                            max_seq_len=MAX_SEQ_LEN, fields=NUMERIC_EVENT_FIELDS):
    """Return dict of arrays, each first dim = len(client_ids_ordered):
      store_ids   int32   [N, L]
      weekday     int8    [N, L]
      numeric     float32 [N, L, K]
      gap_days    float32 [N, L]
      mask        bool    [N, L]   True where a real (non-padding) event is present
      length      int32   [N]
    Sequences are the LAST max_seq_len receipts per client (chronological
    order preserved), left-padded with zeros for clients with fewer receipts.
    """
    # Vectorized preparation: encode/scale ALL rows ONCE, then use
    # groupby(...).indices (integer position arrays, not materialized
    # per-group DataFrames -- materializing ~200k individual DataFrame
    # objects via `{cid: g for cid, g in df.groupby(...)}` was measured to be
    # the dominant cost of this function, turning a few-minute job into an
    # hours-long one) to slice pre-built numpy arrays per client.
    df_sorted = df.sort_values(["client_id", "receipt_idx"], kind="stable").reset_index(drop=True)

    store_ids_all = store_vocab.encode(df_sorted["store_id"].to_numpy())
    weekday_all = df_sorted["weekday"].to_numpy(dtype=np.int8)
    gap_days_all = np.clip(df_sorted["gap_days"].to_numpy(dtype=np.float32), 0, 365)

    k = len(fields)
    numeric_all = np.empty((len(df_sorted), k), dtype=np.float32)
    for j, f in enumerate(fields):
        median, scale = numeric_stats[f]
        numeric_all[:, j] = ((df_sorted[f].to_numpy(dtype=np.float64) - median) / scale).astype(np.float32)

    group_positions = df_sorted.groupby("client_id", sort=False, observed=True).indices

    n = len(client_ids_ordered)
    store_ids = np.zeros((n, max_seq_len), dtype=np.int32)
    weekday = np.zeros((n, max_seq_len), dtype=np.int8)
    numeric = np.zeros((n, max_seq_len, k), dtype=np.float32)
    gap_days = np.zeros((n, max_seq_len), dtype=np.float32)
    mask = np.zeros((n, max_seq_len), dtype=bool)
    lengths = np.zeros(n, dtype=np.int32)

    for i, cid in enumerate(client_ids_ordered):
        pos = group_positions.get(cid)
        if pos is None or len(pos) == 0:
            continue
        pos = pos[-max_seq_len:]  # positions are already chronological (sorted above)
        L = len(pos)
        lengths[i] = L
        store_ids[i, -L:] = store_ids_all[pos]
        weekday[i, -L:] = weekday_all[pos]
        gap_days[i, -L:] = gap_days_all[pos]
        numeric[i, -L:, :] = numeric_all[pos]
        mask[i, -L:] = True

    return {
        "store_ids": store_ids,
        "weekday": weekday,
        "numeric": numeric,
        "gap_days": gap_days,
        "mask": mask,
        "length": lengths,
    }
