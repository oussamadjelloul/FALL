"""windowing.py - FALL reproduction (Thunderbird / BGL).

Builds 30-log / stride-15 windows per node, applies temporal failure-prediction
labels for the 5 scenarios, performs a chronological 70/15/15 split with a
within-partition shuffle, and reports positive-window counts per scenario.

Implements spec v3: A.1 (window 30, stride 15), A.3 (windowing/labeling),
D8 (chronological split + within-partition shuffle), D12 (lead-gap -> normal).

Input contract (produced by preprocess.py):
    An iterable of preprocessed records, each a dict:
        {"node": str, "ts": float, "fail": bool, "content": str}
    - content is already lowercased / punctuation-stripped / NUM-substituted and
      consecutive-deduped per node (preprocess.py's responsibility).
    - Order does not matter; this module sorts each node stream by ts.

Note: this draft loads all records in memory. Fine for the loghub BGL/TB samples;
for the full corpora, stream per-node on Narval (TODO when we scale up).
"""
from __future__ import annotations

import json
import os
import random
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass

WINDOW = 30
STRIDE = 15
SPLIT = (0.70, 0.15, 0.15)   # train / val / test  (D8)
SEED = 1337

# (name, lead_time_s, interval_end_s)  -- Table VIII
SCENARIOS = [
    ("s1", 10, 30),
    ("s2", 30, 60),
    ("s3", 60, 180),
    ("s4", 180, 300),
    ("s5", 300, 420),
]


@dataclass
class Window:
    node: str
    t_start: float
    t_end: float
    logs: list          # 30 normalized log-content strings
    label: int = 0      # set per scenario
    split: str = ""     # "train" | "val" | "test"


def group_by_node(records):
    by_node = defaultdict(list)
    for r in records:
        by_node[r["node"]].append(r)
    for recs in by_node.values():
        recs.sort(key=lambda r: r["ts"])
    return by_node


def build_windows(by_node):
    """Label-agnostic windows + per-node sorted failure-timestamp arrays."""
    windows = []
    fail_ts = {}
    for node, recs in by_node.items():
        fail_ts[node] = [r["ts"] for r in recs if r["fail"]]  # sorted (recs sorted)
        n = len(recs)
        if n < WINDOW:
            continue
        for i in range(0, n - WINDOW + 1, STRIDE):
            seg = recs[i:i + WINDOW]
            windows.append(Window(
                node=node,
                t_start=seg[0]["ts"],
                t_end=seg[-1]["ts"],
                logs=[r["content"] for r in seg],
            ))
    return windows, fail_ts


def label_window(w, fail_ts_node, lead, end):
    """Abnormal iff a failure occurs in (t_end+lead, t_end+end] on the same node.
    Failures in the lead gap (t_end, t_end+lead] are ignored -> normal (D12)."""
    lo = bisect_right(fail_ts_node, w.t_end + lead)   # first ts > t_end+lead
    hi = bisect_right(fail_ts_node, w.t_end + end)     # first ts > t_end+end
    return 1 if hi > lo else 0


def chronological_split(windows, split=SPLIT):
    """Cut window indices into train/val/test by t_end. Membership only;
    the within-partition shuffle happens at emit time (fixed seed)."""
    ordered = sorted(range(len(windows)), key=lambda k: windows[k].t_end)
    n = len(ordered)
    n_tr = int(n * split[0])
    n_va = int(n * split[1])
    return {
        "train": ordered[:n_tr],
        "val":   ordered[n_tr:n_tr + n_va],
        "test":  ordered[n_tr + n_va:],
    }


def emit(windows, labels, parts, seed=SEED):
    """Materialize labeled windows per split. Training partition is filtered to
    normal-only (one-class, Table III). Each partition is shuffled in place."""
    rng = random.Random(seed)
    out = {}
    for split_name, idxs in parts.items():
        bucket = []
        for k in idxs:
            lbl = labels[k]
            if split_name == "train" and lbl != 0:
                continue
            w = windows[k]
            bucket.append(Window(w.node, w.t_start, w.t_end, w.logs, lbl, split_name))
        rng.shuffle(bucket)                  # within-partition shuffle only (D8)
        out[split_name] = bucket
    return out


def _save(scenario_out, out_dir, dataset, name):
    base = os.path.join(out_dir, dataset, name)
    os.makedirs(base, exist_ok=True)
    for split_name, ws in scenario_out.items():
        path = os.path.join(base, f"{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for w in ws:
                f.write(json.dumps({
                    "node": w.node, "t_start": w.t_start, "t_end": w.t_end,
                    "label": w.label, "logs": w.logs,
                }) + "\n")


def run(records, dataset, out_dir=None, verbose=True):
    by_node = group_by_node(records)
    windows, fail_ts = build_windows(by_node)
    parts = chronological_split(windows)
    if verbose:
        print(f"[{dataset}] nodes={len(by_node)}  windows={len(windows)}  "
              f"train:{len(parts['train'])} val:{len(parts['val'])} test:{len(parts['test'])}")
        print(f"{'scen':>4} {'lead':>5} {'end':>5} {'allPos':>8} {'prev%':>7} "
              f"{'valTot':>7} {'valPos':>7} {'tstTot':>7} {'tstPos':>7} {'trNorm':>7}")
    results = {}
    for name, lead, end in SCENARIOS:
        labels = [label_window(w, fail_ts[w.node], lead, end) for w in windows]
        emitted = emit(windows, labels, parts)
        results[name] = emitted
        if verbose:
            all_pos = sum(labels)
            prev = (100.0 * all_pos / len(windows)) if windows else 0.0
            vt, vp = len(emitted["val"]),  sum(x.label for x in emitted["val"])
            tt, tp = len(emitted["test"]), sum(x.label for x in emitted["test"])
            trn = len(emitted["train"])
            print(f"{name:>4} {lead:>5} {end:>5} {all_pos:>8} {prev:>7.3f} "
                  f"{vt:>7} {vp:>7} {tt:>7} {tp:>7} {trn:>7}")
        if out_dir:
            _save(emitted, out_dir, dataset, name)
    return results


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FALL windowing + temporal labeling")
    ap.add_argument("--input", required=True, help="preprocessed JSONL from preprocess.py")
    ap.add_argument("--dataset", required=True, choices=["bgl", "thunderbird"])
    ap.add_argument("--out", default=None, help="output dir for window JSONL (optional)")
    a = ap.parse_args()
    run(list(_read_jsonl(a.input)), a.dataset, a.out)
