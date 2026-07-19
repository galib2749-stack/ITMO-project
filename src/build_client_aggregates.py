"""Single streaming pass over purchases.csv producing:

1. data/interim/receipts_sequences.parquet -- one row per client receipt
   (the raw input the Transformer consumes).
2. data/interim/client_purchase_features.csv -- per-client scalar "RFM-style"
   aggregate features for the CatBoost models.
3. data/interim/sequence_length_summary.json -- length-distribution summary
   stats at item / receipt / day granularity, used to justify the
   sequence-unit choice in the notebook.

Everything here is computed ONLY from purchases.csv, which the official
README states is "история покупок клиентов до смс кампании" (purchase history
BEFORE the SMS campaign) -- i.e. it is pre-treatment by construction for every
client. No post-treatment information is read in this module.

Performance note: this processes ~45.8M rows. All aggregation is done with
vectorized pandas groupby calls over whole chunks (tens of thousands of
clients at once), never a Python-level loop per client -- a per-client loop
over 400k clients was measured to be too slow (did not finish a 2M-row subset
in 180s) and was replaced with this chunk-vectorized version.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from purchases_stream import PURCHASES_DTYPES

RECEIPT_SCHEMA = pa.schema([
    ("client_id", pa.string()),
    ("receipt_idx", pa.int32()),
    ("transaction_datetime", pa.timestamp("s")),
    ("gap_days", pa.float32()),
    ("weekday", pa.int8()),
    ("store_id", pa.string()),
    ("n_items", pa.int32()),
    ("n_unique_products", pa.int32()),
    ("n_unique_categories", pa.int32()),
    ("purchase_sum", pa.float32()),
    ("total_quantity", pa.float32()),
    ("regular_points_received", pa.float32()),
    ("express_points_received", pa.float32()),
    ("regular_points_spent", pa.float32()),
    ("express_points_spent", pa.float32()),
    ("discount_sum", pa.float32()),
    ("iss_sum", pa.float32()),
])


def _load_product_category_map(products_path, level="level_2"):
    prod = pd.read_csv(products_path, usecols=["product_id", level], dtype="string")
    return dict(zip(prod["product_id"], prod[level]))


def _split_off_carry(chunk: pd.DataFrame):
    """Return (complete, carry) where carry is the still-possibly-incomplete
    trailing block of the last client_id in this chunk (may itself later be
    merged with more chunks if that client's block spans further)."""
    ids = chunk["client_id"].to_numpy()
    last_id = ids[-1]
    is_last = ids == last_id
    if is_last.all():
        return chunk.iloc[0:0], chunk
    boundary = len(chunk) - is_last[::-1].argmin()
    return chunk.iloc[:boundary], chunk.iloc[boundary:]


def _process_chunk(chunk: pd.DataFrame, product_to_cat: dict, writer_holder: dict,
                    receipts_out_path: str, client_agg_parts: list,
                    len_parts: dict):
    if chunk.empty:
        return

    chunk = chunk.sort_values(["client_id", "transaction_datetime"], kind="stable")
    chunk = chunk.assign(_cat=chunk["product_id"].map(product_to_cat))
    chunk = chunk.assign(_date=chunk["transaction_datetime"].dt.floor("D"))

    # ---- item / day level length stats (vectorized) ----
    item_counts = chunk.groupby("client_id", sort=False, observed=True).size()
    day_counts = chunk.groupby("client_id", sort=False, observed=True)["_date"].nunique()
    len_parts["item"].append(item_counts)
    len_parts["day"].append(day_counts)

    # ---- receipt level aggregation (vectorized across all clients at once) ----
    g = chunk.groupby(["client_id", "transaction_id"], sort=False, observed=True)
    receipts = g.agg(
        transaction_datetime=("transaction_datetime", "first"),
        store_id=("store_id", "first"),
        n_items=("product_id", "size"),
        n_unique_products=("product_id", "nunique"),
        n_unique_categories=("_cat", "nunique"),
        purchase_sum=("purchase_sum", "first"),
        total_quantity=("product_quantity", "sum"),
        regular_points_received=("regular_points_received", "first"),
        express_points_received=("express_points_received", "first"),
        regular_points_spent=("regular_points_spent", "first"),
        express_points_spent=("express_points_spent", "first"),
        discount_sum=("trn_sum_from_red", "sum"),
        iss_sum=("trn_sum_from_iss", "sum"),
    ).reset_index()

    receipts = receipts.sort_values(["client_id", "transaction_datetime"], kind="stable")
    receipts["receipt_idx"] = receipts.groupby("client_id", sort=False, observed=True).cumcount().astype("int32")
    gap = receipts.groupby("client_id", sort=False, observed=True)["transaction_datetime"].diff().dt.total_seconds() / 86400.0
    receipts["gap_days"] = gap.fillna(0.0).astype("float32")
    receipts["weekday"] = receipts["transaction_datetime"].dt.weekday.astype("int8")

    receipt_counts = receipts.groupby("client_id", sort=False, observed=True).size()
    len_parts["receipt"].append(receipt_counts)

    # ---- per-client scalar aggregates for this chunk ----
    rec_client_agg = receipts.groupby("client_id", sort=False, observed=True).agg(
        n_receipts=("receipt_idx", "size"),
        total_spend=("purchase_sum", "sum"),
        avg_receipt_value=("purchase_sum", "mean"),
        total_quantity=("total_quantity", "sum"),
        total_regular_points_received=("regular_points_received", "sum"),
        total_express_points_received=("express_points_received", "sum"),
        total_regular_points_spent=("regular_points_spent", "sum"),
        total_express_points_spent=("express_points_spent", "sum"),
        total_discount=("discount_sum", "sum"),
        first_purchase_date=("transaction_datetime", "min"),
        last_purchase_date=("transaction_datetime", "max"),
    ).reset_index()

    item_client_agg = chunk.groupby("client_id", sort=False, observed=True).agg(
        n_items=("product_id", "size"),
        n_unique_stores=("store_id", "nunique"),
        n_unique_products=("product_id", "nunique"),
        n_unique_categories=("_cat", "nunique"),
        n_active_days=("_date", "nunique"),
    ).reset_index()

    client_chunk_features = rec_client_agg.merge(item_client_agg, on="client_id", how="outer")
    client_agg_parts.append(client_chunk_features)

    # ---- write receipts to parquet ----
    table = pa.Table.from_pandas(
        receipts.assign(client_id=receipts["client_id"].astype("string"))[[f.name for f in RECEIPT_SCHEMA]],
        schema=RECEIPT_SCHEMA,
        preserve_index=False,
    )
    if writer_holder.get("writer") is None:
        writer_holder["writer"] = pq.ParquetWriter(receipts_out_path, RECEIPT_SCHEMA)
    writer_holder["writer"].write_table(table)


def build(purchases_path, products_path, out_receipts_parquet, out_client_features_csv,
          out_length_summary_json, nrows=None, chunksize=2_000_000, log_every=1):
    product_to_cat = _load_product_category_map(products_path)

    dtypes = PURCHASES_DTYPES
    reader = pd.read_csv(
        purchases_path,
        chunksize=chunksize,
        dtype=dtypes,
        parse_dates=["transaction_datetime"],
        nrows=nrows,
    )

    writer_holder: dict = {"writer": None}
    client_agg_parts: list = []
    len_parts = {"item": [], "receipt": [], "day": []}

    t0 = time.time()
    carry = None
    n_chunks = 0
    n_rows_seen = 0
    for chunk in reader:
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        complete, carry = _split_off_carry(chunk)
        _process_chunk(complete, product_to_cat, writer_holder, out_receipts_parquet,
                        client_agg_parts, len_parts)
        n_chunks += 1
        n_rows_seen += len(complete)
        if n_chunks % log_every == 0:
            print(f"  chunk {n_chunks}: rows_seen={n_rows_seen} elapsed={time.time()-t0:.1f}s")

    if carry is not None and not carry.empty:
        _process_chunk(carry, product_to_cat, writer_holder, out_receipts_parquet,
                        client_agg_parts, len_parts)
        n_rows_seen += len(carry)

    if writer_holder["writer"] is not None:
        writer_holder["writer"].close()

    # ---- combine per-chunk client aggregates (a client only ever appears in
    # exactly one processed chunk, thanks to the carry-over boundary logic,
    # so this is a concat, not a re-aggregation) ----
    client_features = pd.concat(client_agg_parts, ignore_index=True)
    client_features["history_span_days"] = (
        (client_features["last_purchase_date"] - client_features["first_purchase_date"])
        .dt.total_seconds() / 86400.0
    ).clip(lower=0.0)
    client_features["receipts_per_week"] = client_features["n_receipts"] / client_features["history_span_days"].clip(lower=1.0) * 7.0
    client_features.to_csv(out_client_features_csv, index=False)

    def _summ(series_list):
        a = pd.concat(series_list).to_numpy()
        return {
            "count": int(len(a)),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": int(a.min()),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": int(a.max()),
        }

    summary = {
        "n_clients": int(len(client_features)),
        "n_rows_processed": n_rows_seen,
        "item_level": _summ(len_parts["item"]),
        "receipt_level": _summ(len_parts["receipt"]),
        "day_level": _summ(len_parts["day"]),
        "elapsed_seconds": time.time() - t0,
    }
    with open(out_length_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import sys
    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    summary = build(
        purchases_path="data/raw/x5_retailhero/data/purchases.csv",
        products_path="data/raw/x5_retailhero/data/products.csv",
        out_receipts_parquet="data/interim/receipts_sequences.parquet",
        out_client_features_csv="data/interim/client_purchase_features.csv",
        out_length_summary_json="data/interim/sequence_length_summary.json",
        nrows=nrows,
    )
    print(json.dumps(summary, indent=2))
