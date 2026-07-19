"""Transformer Encoder with Shared Customer Representation and Two Outcome Heads.

(Not called a "Causal Transformer" -- see project spec naming requirement.)

Architecture:

    purchase (receipt) sequence
            |
    event embedding = proj([store_id_emb, weekday_emb, numeric_proj])
            |
    + positional embedding
    + time-gap embedding (from gap_days, a learned projection)
            |
    Transformer Encoder (shared, `num_layers` layers, `n_heads` heads)
            |
    mean-pool over non-padded positions -> customer representation
            |
       /                    \\
  control head           treatment head
   mu_0(x)                 mu_1(x)
       \\                    /
         uplift = mu_1(x) - mu_0(x)

The encoder and pooling are shared across both treatment and control units
(both flow through the SAME weights); only the two small output heads are
separate. Per-observation training loss is computed on whichever head
matches that observation's OBSERVED treatment status only -- a control unit
never contributes gradient to the treatment head and vice versa.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ReceiptSequenceDataset(Dataset):
    def __init__(self, tensors, treatment=None, target=None):
        self.store_ids = torch.as_tensor(tensors["store_ids"], dtype=torch.long)
        self.weekday = torch.as_tensor(tensors["weekday"], dtype=torch.long)
        self.numeric = torch.as_tensor(tensors["numeric"], dtype=torch.float32)
        self.gap_days = torch.as_tensor(tensors["gap_days"], dtype=torch.float32)
        self.mask = torch.as_tensor(tensors["mask"], dtype=torch.bool)
        self.treatment = None if treatment is None else torch.as_tensor(treatment, dtype=torch.float32)
        self.target = None if target is None else torch.as_tensor(target, dtype=torch.float32)

    def __len__(self):
        return self.store_ids.shape[0]

    def __getitem__(self, idx):
        item = {
            "store_ids": self.store_ids[idx],
            "weekday": self.weekday[idx],
            "numeric": self.numeric[idx],
            "gap_days": self.gap_days[idx],
            "mask": self.mask[idx],
        }
        if self.treatment is not None:
            item["treatment"] = self.treatment[idx]
        if self.target is not None:
            item["target"] = self.target[idx]
        return item


class TwoHeadTransformer(nn.Module):
    def __init__(self, n_stores, n_numeric_fields, max_seq_len=50, hidden_size=64,
                 n_heads=4, n_layers=2, dropout=0.2, weekday_vocab=7):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        store_emb_dim = 16
        weekday_emb_dim = 4
        numeric_proj_dim = 24

        self.store_emb = nn.Embedding(n_stores, store_emb_dim, padding_idx=0)
        self.weekday_emb = nn.Embedding(weekday_vocab, weekday_emb_dim)
        self.numeric_proj = nn.Linear(n_numeric_fields, numeric_proj_dim)

        self.event_proj = nn.Linear(store_emb_dim + weekday_emb_dim + numeric_proj_dim, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.gap_proj = nn.Sequential(nn.Linear(1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))

        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        def make_head():
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )

        self.control_head = make_head()
        self.treatment_head = make_head()

    def encode(self, store_ids, weekday, numeric, gap_days, mask):
        B, L = store_ids.shape
        event = torch.cat([
            self.store_emb(store_ids),
            self.weekday_emb(weekday),
            torch.relu(self.numeric_proj(numeric)),
        ], dim=-1)
        h = self.event_proj(event)

        positions = torch.arange(L, device=store_ids.device).unsqueeze(0).expand(B, L)
        h = h + self.pos_emb(positions)
        h = h + self.gap_proj(gap_days.unsqueeze(-1))
        h = self.input_dropout(self.input_norm(h))

        key_padding_mask = ~mask  # True = ignore, for positions with no real event
        # guard against fully-empty sequences (a client with zero receipts):
        # give the encoder at least one attendable position to avoid NaNs.
        all_padded = key_padding_mask.all(dim=1)
        if all_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded, 0] = False

        encoded = self.encoder(h, src_key_padding_mask=key_padding_mask)

        mask_f = mask.float().unsqueeze(-1)
        summed = (encoded * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        customer_repr = summed / counts
        return customer_repr

    def forward(self, store_ids, weekday, numeric, gap_days, mask):
        customer_repr = self.encode(store_ids, weekday, numeric, gap_days, mask)
        mu0_logit = self.control_head(customer_repr).squeeze(-1)
        mu1_logit = self.treatment_head(customer_repr).squeeze(-1)
        return mu0_logit, mu1_logit


def two_head_loss(mu0_logit, mu1_logit, treatment, target, control_weight=None, treatment_weight=None):
    bce = nn.functional.binary_cross_entropy_with_logits
    control_mask = (treatment == 0)
    treatment_mask = (treatment == 1)

    loss = torch.zeros((), device=mu0_logit.device)
    n_terms = 0
    if control_mask.any():
        w = control_weight if control_weight is not None else 1.0
        loss = loss + w * bce(mu0_logit[control_mask], target[control_mask])
        n_terms += 1
    if treatment_mask.any():
        w = treatment_weight if treatment_weight is not None else 1.0
        loss = loss + w * bce(mu1_logit[treatment_mask], target[treatment_mask])
        n_terms += 1
    return loss / max(n_terms, 1)


def train_transformer(model, train_loader, val_loader, epochs=15, lr=1e-3, weight_decay=1e-4,
                       grad_clip=1.0, patience=3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_train_control = 0
    n_train_treatment = 0
    for batch in train_loader:
        n_train_control += int((batch["treatment"] == 0).sum())
        n_train_treatment += int((batch["treatment"] == 1).sum())
    total = n_train_control + n_train_treatment
    control_weight = total / (2.0 * max(n_train_control, 1))
    treatment_weight = total / (2.0 * max(n_train_treatment, 1))

    history = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            mu0_logit, mu1_logit = model(batch["store_ids"], batch["weekday"], batch["numeric"],
                                          batch["gap_days"], batch["mask"])
            loss = two_head_loss(mu0_logit, mu1_logit, batch["treatment"], batch["target"],
                                  control_weight, treatment_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                mu0_logit, mu1_logit = model(batch["store_ids"], batch["weekday"], batch["numeric"],
                                              batch["gap_days"], batch["mask"])
                loss = two_head_loss(mu0_logit, mu1_logit, batch["treatment"], batch["target"])
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        elapsed = time.time() - t0
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "seconds": elapsed})
        print(f"epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({elapsed:.1f}s)")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_uplift(model, loader, device="cpu"):
    model.eval()
    model.to(device)
    mu0s, mu1s = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        mu0_logit, mu1_logit = model(batch["store_ids"], batch["weekday"], batch["numeric"],
                                      batch["gap_days"], batch["mask"])
        mu0s.append(torch.sigmoid(mu0_logit).cpu().numpy())
        mu1s.append(torch.sigmoid(mu1_logit).cpu().numpy())
    mu0 = np.concatenate(mu0s)
    mu1 = np.concatenate(mu1s)
    return mu1 - mu0, mu0, mu1
