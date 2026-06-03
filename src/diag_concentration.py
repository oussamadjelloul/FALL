"""diag_concentration.py - per-token RTD probability concentration diagnostic.

Question this answers:
  Selective/partial-token scoring only beats plain averaging when the anomaly
  signal is CONCENTRATED in a few tokens. The paper says selective is its most
  useful component (Table VIII/IX); in our runs partial_only ~= date, i.e. it
  adds nothing. This script checks WHY: are the model's per-token replacement
  probabilities concentrated (a few sharp spikes) or diffuse (flat)?

Per window we pool the raw RTD probs over content tokens and compute:
  mean_all     mean of all token probs              (~ the DATE score)
  mean_top     mean of the top-1/n token probs      (~ the selective score)
  top1         single highest token prob            (does the model spike?)
  mass_top     fraction of total prob mass in top-1/n tokens   (1/n = perfectly flat)
  gini         inequality of the prob vector        (0 = flat, ->1 = concentrated)
  ratio        mean_top / mean_all                  (1.0 = flat, >1 = concentrated)

We then report the class means (normal vs pre-failure) and the AUC of mean_all
and mean_top as class separators (these mirror the date and partial_only AUCs).

If pre-failure windows are NOT more concentrated than normal windows, selective
scoring cannot separate the classes -> reproduction gap is in the model/tokens,
not the scoring.

Usage:
  python src/diag_concentration.py --windows-dir data/processed/bgl --scenario s1 \
      --tokenizer data/processed/bgl/bgl_tok_8000.json \
      --ckpt-dir results/ckpt/bgl --seed 0
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import evaluate as E
import score as S
from model import Discriminator


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * np.sum(x)) - (n + 1.0) / n)


def win_stats(p, n=4):
    p = np.asarray(p, dtype=float)
    L = p.size
    k = max(1, L // n)
    top = np.sort(p)[::-1][:k]
    s = p.sum()
    mean_all = float(p.mean())
    mean_top = float(top.mean())
    return {
        "mean_all": mean_all,
        "mean_top": mean_top,
        "top1": float(p.max()),
        "mass_top": float(top.sum() / s) if s > 0 else 0.0,
        "gini": gini(p),
        "ratio": (mean_top / mean_all) if mean_all > 0 else 1.0,
    }


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["config"]
    m = Discriminator(c["vocab"], d_model=c["d_model"], nhead=c["nhead"],
                      num_layers=c["layers"], dim_ff=c["dim_ff"],
                      max_len=c["max_len"], k_patterns=c["k"],
                      pad_id=c["pad"]).to(device)
    m.load_state_dict(ck["model"])
    return m


def main():
    ap = argparse.ArgumentParser(description="Per-token concentration diagnostic")
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok, cls, sep, pad = E.load_tokenizer(args.tokenizer)
    test_w = E.load_windows(os.path.join(args.windows_dir, args.scenario,
                                         "test.jsonl"))
    model = load_model(os.path.join(args.ckpt_dir,
                                    f"{args.scenario}_seed{args.seed}.pt"),
                       args.device)
    t_ii, t_cl, t_wid, t_lab = E.build_segments(test_w, tok, cls, sep, pad)
    segs = E.window_segment_probs(model, t_ii, t_cl, t_wid, len(test_w),
                                  args.device)
    labels = t_lab.tolist()

    keys = ["mean_all", "mean_top", "top1", "mass_top", "gini", "ratio"]
    acc = {0: {k: [] for k in keys}, 1: {k: [] for k in keys}}
    mean_all_scores, mean_top_scores = [], []
    for w_segs, y in zip(segs, labels):
        pooled = torch.cat(w_segs).numpy()
        st = win_stats(pooled, args.n)
        for k in keys:
            acc[y][k].append(st[k])
        mean_all_scores.append(st["mean_all"])
        mean_top_scores.append(st["mean_top"])

    n_norm = len(acc[0]["mean_all"])
    n_fail = len(acc[1]["mean_all"])
    flat_mass = 1.0 / args.n
    print(f"[{args.scenario}] seed={args.seed}  normal={n_norm}  "
          f"pre-failure={n_fail}  top-fraction=1/{args.n}")
    print(f"  (flat baseline: mass_top={flat_mass:.2f}, gini=0.00, ratio=1.00)")
    print(f"  {'metric':>10} {'normal':>12} {'pre-failure':>14} {'fail/normal':>12}")
    for k in keys:
        mn = float(np.mean(acc[0][k]))
        mf = float(np.mean(acc[1][k]))
        rr = (mf / mn) if mn != 0 else float("nan")
        print(f"  {k:>10} {mn:12.4f} {mf:14.4f} {rr:12.2f}")

    auc_all = S.auc_roc(mean_all_scores, labels)
    auc_top = S.auc_roc(mean_top_scores, labels)
    print(f"  AUC as class-separator:  mean_all (=date) {auc_all:.4f}   "
          f"mean_top (=selective) {auc_top:.4f}   "
          f"lift {auc_top - auc_all:+.4f}")


if __name__ == "__main__":
    main()
