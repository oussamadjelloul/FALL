"""diag_nodes.py - BGL node-granularity diagnostic (gate-4 unblock).

Reads preprocessed records and reports, for several node-key coarsening levels,
how many groups clear the 30-log window and how many 30/stride-15 windows
result. BGL's location field (e.g. R02-M1-N0-C:J12-U11) is far finer than the
paper's "node" concept; full-field grouping starves per-node history. This picks
the coarsening that maps BGL onto the paper's node level (decision D14).

Read-only: writes nothing. Run:  python src/diag_nodes.py data/processed/bgl/bgl_pre.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter

WINDOW, STRIDE = 30, 15


def key_full(n):      return n                                          # R02-M1-N0-C:J12-U11
def key_drop_chip(n): return n.split(":")[0]                            # R02-M1-N0-C
def key_node_card(n): return "-".join(n.split(":")[0].split("-")[:3])   # R02-M1-N0
def key_first2(n):    return "-".join(n.split("-")[:2])                 # R02-M1

LEVELS = [("full", key_full), ("drop_chip", key_drop_chip),
          ("node_card", key_node_card), ("first2", key_first2)]


def windows_for(count):
    return 0 if count < WINDOW else (count - WINDOW) // STRIDE + 1


def main(path):
    counts = {name: Counter() for name, _ in LEVELS}
    sample, n = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            node = json.loads(line)["node"]
            n += 1
            if len(sample) < 15:
                sample.append(node)
            for name, kf in LEVELS:
                counts[name][kf(node)] += 1

    print(f"records={n}")
    print("sample node strings:")
    for s in sample:
        print("   ", s)
    print(f"\n{'level':>10} {'groups':>8} {'g>=30':>8} {'medRec':>8} "
          f"{'maxRec':>9} {'windows':>9}")
    for name, _ in LEVELS:
        vals = sorted(counts[name].values())
        ge30 = sum(1 for v in vals if v >= WINDOW)
        med = vals[len(vals) // 2] if vals else 0
        wins = sum(windows_for(v) for v in vals)
        print(f"{name:>10} {len(vals):>8} {ge30:>8} {med:>8} "
              f"{(vals[-1] if vals else 0):>9} {wins:>9}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/processed/bgl/bgl_pre.jsonl")
