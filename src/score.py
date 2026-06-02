"""score.py - FALL reproduction, anomaly scoring (spec III-C, D6).

Given a window's pooled RTD replacement-probabilities (one value per real content
token, concatenated across the window's segments, D10), compute the anomaly
score. Provides the FALL score plus baseline/ablation variants and
threshold-agnostic AUC + avg-FP helpers (no sklearn dependency).

FALL score (D6): mean of the top-(1/n) sharpened probabilities.
  sharpen(p, T)_i = p_i^(1/T) / sum_j p_j^(1/T)        (Eq. 6, T=1/2)
  m = floor(L / n)  (n=4) ; score = (n/m) * sum(top-m sharpened) = n * mean(top-m)
The constant n is harmless (global), so ranking == mean of top-(1/n) sharpened.
"""
from __future__ import annotations

import torch


def sharpen(probs, T=0.5):
    p = probs.clamp_min(1e-12)
    w = p ** (1.0 / T)
    return w / w.sum().clamp_min(1e-12)


def fall_score(probs, n=4, T=0.5):                 # primary (D6)
    L = probs.numel()
    if L == 0:
        return 0.0
    s = sharpen(probs, T)
    m = max(1, L // n)
    return (n / m) * torch.topk(s, m).values.sum().item()


def date_score(probs):                              # B.1.1 baseline: mean all tokens
    return probs.mean().item() if probs.numel() else 0.0


def sharpen_only_score(probs, T=0.5):               # B.1.2: sharpen, mean all
    return sharpen(probs, T).mean().item() if probs.numel() else 0.0


def partial_only_score(probs, n=4):                 # B.1.3: top-(1/n) mean, raw probs
    L = probs.numel()
    if L == 0:
        return 0.0
    m = max(1, L // n)
    return torch.topk(probs, m).values.mean().item()


def fall_sum_score(probs, n=4, T=0.5):              # B.2.2 hedge: top-m SUM (no mean)
    L = probs.numel()
    if L == 0:
        return 0.0
    s = sharpen(probs, T)
    m = max(1, L // n)
    return torch.topk(s, m).values.sum().item()


SCORERS = {
    "fall": fall_score,
    "date": date_score,
    "sharpen_only": sharpen_only_score,
    "partial_only": partial_only_score,
    "fall_sum": fall_sum_score,
}


def auc_roc(scores, labels):
    """Rank-based AUC (Mann-Whitney U); higher score = positive. NaN if a class
    is absent. Average ranks on ties."""
    pairs = list(zip(scores, [int(y) for y in labels]))
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0                    # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(pairs)) if pairs[i][1] == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def threshold_at_percentile(normal_scores, pct=99.0):
    xs = sorted(normal_scores)
    if not xs:
        return float("inf")
    k = min(len(xs) - 1, int(round(pct / 100.0 * (len(xs) - 1))))
    return xs[k]


def false_positives(normal_scores, threshold):
    return sum(1 for x in normal_scores if x > threshold)