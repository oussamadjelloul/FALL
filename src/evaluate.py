"""evaluate.py - FALL reproduction evaluation (Table XI / Table VIII, BGL).

Loads each trained discriminator, scores test windows, and reports AUC-ROC +
avg-FP at the D7 val-normal threshold, per scoring mode, mean+-std over seeds.

Score unit (length-confound fix):
  segment (default) - score each <=128-token segment on its own (sharpen within
                      the segment), then average the segment scores per window.
                      Paper's "split and process sequentially": bounds the
                      sharpening normalization length so the score does not
                      collapse to 1/L (window length).
  pooled            - old behaviour: concatenate all of a window's segment probs
                      into one vector and sharpen over the whole window, which
                      lets window length leak into the score (1/L artifact).

Scoring modes (all from the SAME checkpoints, no retraining):
  fall          primary (D6: sharpen T=1/2 + top-(1/n) mean)
  date          baseline B.1.1 (mean of RTD probs, no sharpen/partial)
  sharpen_only  ablation B.1.2
  partial_only  ablation B.1.3
  fall_sum      hedge B.2.2 (top-m sum)

Usage:
  python src/evaluate.py --windows-dir data/processed/bgl --scenario s1 \
      --tokenizer data/processed/bgl/bgl_tok_8000.json \
      --ckpt-dir results/ckpt/bgl --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
from tokenizers import Tokenizer

from model import Discriminator
import score as S

MAX_LEN = 128


def load_tokenizer(path):
    tok = Tokenizer.from_file(path)
    return (tok, tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]"),
            tok.token_to_id("[PAD]"))


def load_windows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_segments(windows, tok, cls, sep, pad, max_len=MAX_LEN):
    """Returns input_ids[Nseg,L], content_len[Nseg], win_id[Nseg], labels[Nwin].
    Mirrors train.SegmentPool layout (D9/D10) but tracks the source window."""
    budget = max_len - 2
    rows, clens, wid, labels = [], [], [], []
    for w_i, w in enumerate(windows):
        labels.append(int(w["label"]))
        ids = []
        for log in w["logs"]:
            ids.extend(tok.encode(log).ids)
        for off in range(0, max(1, len(ids)), budget):
            chunk = ids[off:off + budget]
            if not chunk:
                continue
            seq = [cls] + chunk + [sep]
            seq += [pad] * (max_len - len(seq))
            rows.append(seq)
            clens.append(len(chunk))
            wid.append(w_i)
    return (torch.tensor(rows, dtype=torch.long),
            torch.tensor(clens, dtype=torch.long),
            torch.tensor(wid, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))


@torch.no_grad()
def window_segment_probs(model, input_ids, content_len, win_id, n_win, device,
                         batch=256):
    """Per window, a LIST of per-segment RTD-probability vectors (NOT pooled).
    Each vector is the sigmoid RTD probs at that segment's content positions."""
    pos = torch.arange(MAX_LEN).unsqueeze(0)
    buckets = [[] for _ in range(n_win)]
    model.eval()
    for s in range(0, input_ids.size(0), batch):
        ii = input_ids[s:s + batch].to(device)
        cl = content_len[s:s + batch]
        attn = (pos <= (cl.unsqueeze(1) + 1)).long().to(device)
        rtd_logits, _ = model(ii, attn)
        probs = torch.sigmoid(rtd_logits).cpu()
        for r in range(ii.size(0)):
            c = int(cl[r])
            buckets[int(win_id[s + r])].append(probs[r, 1:1 + c])
    return [b if b else [torch.zeros(1)] for b in buckets]


def _apply(fn, mode, vec, n, T):
    if mode in ("fall", "fall_sum"):
        return fn(vec, n=n, T=T)
    if mode == "sharpen_only":
        return fn(vec, T=T)
    if mode == "partial_only":
        return fn(vec, n=n)
    return fn(vec)


def score_windows(win_segs, mode, n, T, unit):
    """unit='segment': score each segment, average per window (length-neutral).
       unit='pooled'  : concatenate a window's segments, score once (old)."""
    fn = S.SCORERS[mode]
    out = []
    for segs in win_segs:
        if unit == "pooled":
            out.append(_apply(fn, mode, torch.cat(segs), n, T))
        else:
            seg_scores = [_apply(fn, mode, seg, n, T) for seg in segs]
            out.append(sum(seg_scores) / len(seg_scores))
    return out


def evaluate_seed(ckpt_path, val_w, test_w, tok, cls, sep, pad, modes, n, T,
                  pct, unit, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = Discriminator(cfg["vocab"], d_model=cfg["d_model"], nhead=cfg["nhead"],
                          num_layers=cfg["layers"], dim_ff=cfg["dim_ff"],
                          max_len=cfg["max_len"], k_patterns=cfg["k"],
                          pad_id=cfg["pad"]).to(device)
    model.load_state_dict(ck["model"])

    v_ii, v_cl, v_wid, v_lab = build_segments(val_w, tok, cls, sep, pad)
    t_ii, t_cl, t_wid, t_lab = build_segments(test_w, tok, cls, sep, pad)
    v_probs = window_segment_probs(model, v_ii, v_cl, v_wid, len(val_w), device)
    t_probs = window_segment_probs(model, t_ii, t_cl, t_wid, len(test_w), device)
    v_labels, t_labels = v_lab.tolist(), t_lab.tolist()

    res = {}
    for mode in modes:
        v_scores = score_windows(v_probs, mode, n, T, unit)
        t_scores = score_windows(t_probs, mode, n, T, unit)
        v_normal = [sc for sc, y in zip(v_scores, v_labels) if y == 0]
        thr = S.threshold_at_percentile(v_normal, pct)
        t_normal = [sc for sc, y in zip(t_scores, t_labels) if y == 0]
        auc = S.auc_roc(t_scores, t_labels)
        fp = S.false_positives(t_normal, thr)
        res[mode] = (auc, fp, len(t_normal))
    return res


def main():
    ap = argparse.ArgumentParser(description="FALL evaluation (BGL)")
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--modes", default="fall,date,sharpen_only,partial_only")
    ap.add_argument("--score-unit", dest="unit", default="segment",
                    choices=["segment", "pooled"],
                    help="segment (fix, default) or pooled (old, length-leaking)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--pct", type=float, default=99.0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    modes = args.modes.split(",")

    tok, cls, sep, pad = load_tokenizer(args.tokenizer)
    val_w = load_windows(os.path.join(args.windows_dir, args.scenario, "val.jsonl"))
    test_w = load_windows(os.path.join(args.windows_dir, args.scenario, "test.jsonl"))
    n_pos = sum(int(w["label"]) for w in test_w)
    per = {m: {"auc": [], "fp": []} for m in modes}
    for seed in (int(x) for x in args.seeds.split(",")):
        ckpt = os.path.join(args.ckpt_dir, f"{args.scenario}_seed{seed}.pt")
        res = evaluate_seed(ckpt, val_w, test_w, tok, cls, sep, pad, modes,
                            args.n, args.T, args.pct, args.unit, args.device)
        for m, (auc, fp, _) in res.items():
            per[m]["auc"].append(auc)
            per[m]["fp"].append(fp)

    print(f"[{args.scenario}] test windows={len(test_w)} pos={n_pos} "
          f"neg={len(test_w) - n_pos}  seeds={args.seeds}  score_unit={args.unit}")
    print(f"{'mode':>14} {'AUC-ROC':>18} {'avg-FP':>16}")
    for m in modes:
        a, f = per[m]["auc"], per[m]["fp"]
        am = statistics.mean(a)
        asd = statistics.pstdev(a) if len(a) > 1 else 0.0
        fm = statistics.mean(f)
        fsd = statistics.pstdev(f) if len(f) > 1 else 0.0
        print(f"{m:>14}   {am:.4f} +- {asd:.4f}   {fm:7.1f} +- {fsd:.1f}")


if __name__ == "__main__":
    main()