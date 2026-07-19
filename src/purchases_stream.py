"""Memory-safe streaming reader for purchases.csv.

purchases.csv (X5 RetailHero) is ~4.4GB / ~45.8M product-line rows and is
verified (see reports/data_download_report.md and the audit notebook section)
to be grouped contiguously by client_id, with the only "boundary" splits being
where a client's block straddles two read-chunks. This module exploits that to
stream the file once, yielding one client's full block of rows at a time,
without ever holding more than a few chunks in memory.

Do not use this on files where the contiguous-by-client assumption hasn't been
verified first (see notebook section "Data audit").
"""
from __future__ import annotations

import pandas as pd

PURCHASES_DTYPES = {
    "client_id": "string",
    "transaction_id": "string",
    "regular_points_received": "float32",
    "express_points_received": "float32",
    "regular_points_spent": "float32",
    "express_points_spent": "float32",
    "purchase_sum": "float32",
    "store_id": "string",
    "product_id": "string",
    "product_quantity": "float32",
    "trn_sum_from_iss": "float32",
    "trn_sum_from_red": "float32",
}

PURCHASES_PARSE_DATES = ["transaction_datetime"]


def iter_client_blocks(csv_path, chunksize=2_000_000, usecols=None, nrows=None):
    """Yield (client_id, pd.DataFrame) one client at a time, in file order.

    Each yielded DataFrame contains ALL rows for that client_id (never split
    across yields), read directly from the CSV in a single streaming pass.
    """
    dtypes = PURCHASES_DTYPES if usecols is None else {
        k: v for k, v in PURCHASES_DTYPES.items() if k in usecols
    }
    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        dtype=dtypes,
        parse_dates=[c for c in PURCHASES_PARSE_DATES if usecols is None or c in usecols],
        usecols=usecols,
        nrows=nrows,
    )

    carry = None  # DataFrame of the not-yet-finalized last client's rows
    for chunk in reader:
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)

        client_ids = chunk["client_id"].values
        last_id = client_ids[-1]
        is_last_block = client_ids == last_id
        boundary = len(chunk) - is_last_block[::-1].argmin() if not is_last_block.all() else 0

        complete = chunk.iloc[:boundary] if boundary > 0 else chunk.iloc[0:0]
        carry = chunk.iloc[boundary:] if not is_last_block.all() else chunk

        if not complete.empty:
            for cid, block in complete.groupby("client_id", sort=False, observed=True):
                yield cid, block

    if carry is not None and not carry.empty:
        for cid, block in carry.groupby("client_id", sort=False, observed=True):
            yield cid, block


def verify_contiguity(csv_path, chunksize=5_000_000, max_chunks=None):
    """Streaming check that client_id blocks are contiguous (no scattered dupes).

    Returns a dict with cum_rows, cum_unique_clients, transitions,
    boundary_adjacent_expected, unexpected_reappearances.
    """
    reader = pd.read_csv(csv_path, usecols=["client_id"], chunksize=chunksize, dtype="string")
    seen_clients: set[str] = set()
    prev_last = None
    transitions = 0
    total_rows = 0
    unexpected = 0
    boundary_adjacent = 0
    for i, chunk in enumerate(reader):
        ids = chunk["client_id"].to_numpy()
        total_rows += len(ids)
        changed = int((ids[1:] != ids[:-1]).sum())
        transitions += changed
        crosses_boundary = prev_last is not None and ids[0] == prev_last
        if prev_last is not None and ids[0] != prev_last:
            transitions += 1
        prev_last = ids[-1]

        chunk_unique = set(pd.unique(ids))
        reappear = chunk_unique & seen_clients
        if crosses_boundary:
            reappear = reappear - {prev_last if False else None}
        n_reappear = len(reappear)
        if crosses_boundary and ids[0] in reappear:
            n_reappear -= 1
            boundary_adjacent += 1
        unexpected += n_reappear
        seen_clients |= chunk_unique
        if max_chunks is not None and i + 1 >= max_chunks:
            break

    return {
        "total_rows": total_rows,
        "cum_unique_clients": len(seen_clients),
        "transitions": transitions,
        "boundary_adjacent_splits": boundary_adjacent,
        "unexpected_reappearances": unexpected,
    }
