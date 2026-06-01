"""model.py - FALL reproduction. ELECTRA/DATE-style discriminator:
token+position embeddings -> 4-layer transformer encoder -> RTD head (per-token
binary) and RMD head ([CLS]-pooled, K-way, D13).

D3 dims (d_model=256, 4 heads, d_ff=1024, 4 layers) are NOT pinned by the paper
(only the layer count is) -> flag in write-up. RTD is per-token sigmoid/BCE:
the paper's Eq.4 symbol is garbled, this is the correct repair (a typo, not an
ablation). Effective loss is mu*RTD + lam*RMD -- L_MLM is inactive under D1's
random generator.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4,
                 dim_ff=1024, max_len=128, k_patterns=50, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.rtd_head = nn.Linear(d_model, 1)
        self.rmd_head = nn.Linear(d_model, k_patterns)
        self.register_buffer("pos_ids", torch.arange(max_len).unsqueeze(0))

    def forward(self, input_ids, attention_mask):
        """input_ids [B,L], attention_mask [B,L] (1=real, 0=pad).
        Returns rtd_logits [B,L], rmd_logits [B,K]."""
        L = input_ids.size(1)
        x = self.tok_emb(input_ids) + self.pos_emb(self.pos_ids[:, :L])
        x = self.drop(x)
        pad_mask = attention_mask == 0                      # True -> ignored
        h = self.encoder(x, src_key_padding_mask=pad_mask)  # [B,L,d]
        rtd_logits = self.rtd_head(h).squeeze(-1)           # [B,L]
        rmd_logits = self.rmd_head(h[:, 0])                 # [B,K] CLS pool (D13)
        return rtd_logits, rmd_logits


def fall_loss(rtd_logits, rmd_logits, rtd_labels, rmd_targets, rtd_loss_mask,
              mu=50.0, lam=100.0):
    """Effective FALL loss (D1: L_MLM inactive) = mu*RTD + lam*RMD.
    rtd_loss_mask [B,L]: 1 on positions counted for RTD (non-[CLS], non-pad)."""
    bce = F.binary_cross_entropy_with_logits(
        rtd_logits, rtd_labels.float(), reduction="none")   # [B,L]
    m = rtd_loss_mask.float()
    rtd_loss = (bce * m).sum() / m.sum().clamp_min(1.0)
    rmd_loss = F.cross_entropy(rmd_logits, rmd_targets)
    total = mu * rtd_loss + lam * rmd_loss
    return total, rtd_loss.detach(), rmd_loss.detach()
