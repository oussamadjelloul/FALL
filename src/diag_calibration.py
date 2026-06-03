"""diag_calibration.py - validation vs test NORMAL score distributions.

Checks whether the D7 threshold (99th pct of VALIDATION-normal scores) is
miscalibrated for the TEST period. If validation-normal scores are systematically
lower than test-normal scores (a chronological-split distribution shift), the
threshold sits too low and inflates test false positives.

Per scenario and mode it reports:
  val p50/p99      percentiles of validation-normal scores
  test p50/p99     percentiles of test-normal scores
  testFPR@valp99   fraction of test-normal exceeding the D7 threshold
                   (~1% = well calibrated; >>1% = miscalibrated / shifted)
  FP               that count (matches evaluate.py avg-FP)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import evaluate as E
from model import Discriminator


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
    ap = argparse.ArgumentParser(description="val vs test normal-score calibration")
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default="date,fall,partial_only,sharpen_only")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    modes = args.modes.split(",")

    tok, cls, sep, pad = E.load_tokenizer(args.tokenizer)
    val_w = E.load_windows(os.path.join(args.windows_dir, args.scenario, "val.jsonl"))
    test_w = E.load_windows(os.path.join(args.windows_dir, args.scenario, "test.jsonl"))
    model = load_model(os.path.join(args.ckpt_dir,
                                    f"{args.scenario}_seed{args.seed}.pt"),
                       args.device)

    v_ii, v_cl, v_wid, v_lab = E.build_segments(val_w, tok, cls, sep, pad)
    t_ii, t_cl, t_wid, t_lab = E.build_segments(test_w, tok, cls, sep, pad)
    v_segs = E.window_segment_probs(model, v_ii, v_cl, v_wid, len(val_w), args.device)
    t_segs = E.window_segment_probs(model, t_ii, t_cl, t_wid, len(test_w), args.device)
    v_lab, t_lab = v_lab.tolist(), t_lab.tolist()

    print(f"[{args.scenario}] seed={args.seed}  "
          f"val_norm={v_lab.count(0)}  test_norm={t_lab.count(0)}")
    print(f"  {'mode':>14} {'val_p50':>9} {'val_p99':>9} {'test_p50':>9} "
          f"{'test_p99':>9} {'testFPR@valp99':>16} {'FP':>7}")
    for mode in modes:
        vs = E.score_windows(v_segs, mode, args.n, args.T, "segment")
        ts = E.score_windows(t_segs, mode, args.n, args.T, "segment")
        vn = np.array([s for s, y in zip(vs, v_lab) if y == 0])
        tn = np.array([s for s, y in zip(ts, t_lab) if y == 0])
        vp50, vp99 = np.percentile(vn, [50, 99])
        tp50, tp99 = np.percentile(tn, [50, 99])
        fp = int((tn >= vp99).sum())
        fpr = fp / len(tn)
        print(f"  {mode:>14} {vp50:9.4f} {vp99:9.4f} {tp50:9.4f} {tp99:9.4f} "
              f"{fpr * 100:14.2f}%  {fp:7d}")


if __name__ == "__main__":
    main()
