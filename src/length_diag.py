"""length_diag.py - FALL reproduction diagnostic (BGL length confound).

Question: do BGL pre-failure windows have a different token-length than normal
windows? If so, FALL's score (which reduces to 1/length under normalized
sharpening) detects the LENGTH, not the anomaly. No model needed -- length is
model-independent, so this runs in seconds on a login node.

Reports per scenario: window counts, median token-length of normal vs
pre-failure windows, and the AUC from scoring on inverse length (1/L) alone.
Compare the AUC(1/L) column to your sharpen_only / fall AUCs from evaluate.py:
if they match, FALL's BGL AUC is largely a length artifact.
"""
import argparse
import json
import os
import statistics

from tokenizers import Tokenizer
import score as S


def load_windows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def window_token_length(tok, w):
    return sum(len(tok.encode(log).ids) for log in w["logs"])


def main():
    ap = argparse.ArgumentParser(description="BGL length-vs-label diagnostic")
    ap.add_argument("--windows-dir", required=True, help="e.g. data/processed/bgl")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--scenarios", default="s1,s2,s3,s4,s5")
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)

    print(f"{'scen':>4} {'n_norm':>7} {'n_fail':>7} "
          f"{'medLen_norm':>12} {'medLen_fail':>12} {'AUC(1/L)':>9}")
    for sc in a.scenarios.split(","):
        w = load_windows(os.path.join(a.windows_dir, sc, "test.jsonl"))
        Ls = [window_token_length(tok, win) for win in w]
        ys = [int(win["label"]) for win in w]
        norm = [L for L, y in zip(Ls, ys) if y == 0]
        fail = [L for L, y in zip(Ls, ys) if y == 1]
        scores = [1.0 / max(1, L) for L in Ls]      # shorter window -> higher score
        auc = S.auc_roc(scores, ys)
        mn = statistics.median(norm) if norm else 0
        mf = statistics.median(fail) if fail else 0
        print(f"{sc:>4} {len(norm):>7} {len(fail):>7} "
              f"{mn:>12.0f} {mf:>12.0f} {auc:>9.4f}")


if __name__ == "__main__":
    main()