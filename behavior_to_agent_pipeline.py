# -*- coding: utf-8 -*-
"""
Behavior → Decision Agent Pipeline (Standalone, Original)
==========================================================

Purpose
-------
This module converts raw customer behavior logs into
action-oriented signals consumable by the Marketing Agent Layer.

Conceptual Flow
---------------
[Behavior Logs]
 → Sequence Encoding
 → Purchase Likelihood Estimation
 → Expected Time-to-Action Estimation
 → (CSV outputs)
 → MarketingAgentLayer (rule-based decision & LLM explanation)

Notes
-----
- This file is written from scratch for individual use.
- Naming avoids FT1/FT2 terminology intentionally.
- Outputs are compatible with agent_layer.py without semantic coupling.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    user_col: str = "user_id"
    category_col: str = "category_id"
    event_col: str = "event"
    time_col: str = "timestamp"

    max_seq_len: int = 60
    batch_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    output_prob_csv: str = "inference_targets.csv"
    output_time_csv: str = "service_eta_predictions.csv"


# ============================================================================
# Data Preparation
# ============================================================================

EVENT_VOCAB = {
    "view": 1,
    "click": 2,
    "cart": 3,
    "remove": 4,
    "purchase": 5,
}


def build_sequences(
    df: pd.DataFrame,
    cfg: PipelineConfig
) -> Tuple[List[Tuple[str, str]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build (user, category) sequences.

    Returns:
        keys: [(user_id, category_id)]
        seq_tensor: (N, L)
        purchase_label: (N,) 0/1
        approx_time_label: (N,) float days (nan if unavailable)
    """

    df = df.copy()
    df[cfg.user_col] = df[cfg.user_col].astype(str)
    df[cfg.category_col] = df[cfg.category_col].astype(str)
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")

    keys, seqs, y_buy, y_time = [], [], [], []

    for (u, c), g in df.groupby([cfg.user_col, cfg.category_col]):
        g = g.sort_values(cfg.time_col)
        tokens = [EVENT_VOCAB.get(e, 0) for e in g[cfg.event_col]]

        if len(tokens) < 2:
            continue

        tokens = tokens[-cfg.max_seq_len:]
        seq = np.zeros(cfg.max_seq_len, dtype=np.int64)
        seq[-len(tokens):] = tokens

        has_purchase = "purchase" in g[cfg.event_col].values
        y_buy.append(float(has_purchase))

        if has_purchase:
            t_last = g[cfg.time_col].max()
            t_buy = g.loc[g[cfg.event_col] == "purchase", cfg.time_col].max()
            eta = max(0.0, (t_buy - t_last).total_seconds() / 86400.0)
        else:
            eta = np.nan

        keys.append((u, c))
        seqs.append(seq)
        y_time.append(eta)

    return (
        keys,
        torch.tensor(seqs),
        torch.tensor(y_buy),
        torch.tensor(y_time),
    )


# ============================================================================
# Model
# ============================================================================

class BehaviorEncoder(nn.Module):
    def __init__(self, vocab_size: int = 8, dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.emb(x)
        _, h_last = self.gru(h)
        return self.norm(h_last.squeeze(0))


class LikelihoodHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.fc(h)).squeeze(-1)


class TimeHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, h):
        return torch.relu(self.fc(h)).squeeze(-1)


# ============================================================================
# Inference Pipeline
# ============================================================================

@torch.no_grad()
def run_inference(
    keys,
    seqs,
    encoder,
    prob_head,
    time_head,
    cfg: PipelineConfig,
):
    encoder.eval()
    prob_head.eval()
    time_head.eval()

    seqs = seqs.to(cfg.device)
    h = encoder(seqs)

    prob = prob_head(h).cpu().numpy()
    eta_days = time_head(h).cpu().numpy()

    # Save probability output
    prob_df = pd.DataFrame([
        {"user_id": u, "category_id": c, "prob": float(p)}
        for (u, c), p in zip(keys, prob)
    ])
    prob_df.to_csv(cfg.output_prob_csv, index=False)

    # Save time-to-action output
    time_df = pd.DataFrame([
        {
            "user_id": u,
            "category_id": c,
            "eta_days": float(d),
            "eta_sec": float(d * 86400),
            "was_clipped": False,
            "n_events": int((seqs[i] > 0).sum())
        }
        for i, ((u, c), d) in enumerate(zip(keys, eta_days))
    ])
    time_df.to_csv(cfg.output_time_csv, index=False)

    return prob_df, time_df


# ============================================================================
# Entry Point
# ============================================================================

def main(log_csv: str):
    cfg = PipelineConfig()
    df = pd.read_csv(log_csv)

    keys, seqs, y_buy, y_time = build_sequences(df, cfg)

    encoder = BehaviorEncoder().to(cfg.device)
    prob_head = LikelihoodHead(64).to(cfg.device)
    time_head = TimeHead(64).to(cfg.device)

    # (Demo) random init inference
    run_inference(keys, seqs, encoder, prob_head, time_head, cfg)


if __name__ == "__main__":
    main("behavior_log.csv")
