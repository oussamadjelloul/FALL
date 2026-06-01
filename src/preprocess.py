"""preprocess.py - FALL reproduction (BGL).

Parses raw loghub BGL logs into the record contract consumed by windowing.py:
    {"node": str, "ts": float, "fail": bool, "content": str}

Pipeline (spec v3 A.3): lowercase -> non-alphanumeric to space -> pure-numeric
tokens to NUM -> collapse whitespace. failure = label != '-'. Node coarsening
(D14), then per-node sort + consecutive dedup.

Provenance: the BGL field layout and the label != '-' failure rule are loghub
conventions, not from the FALL paper (which never published a TB/BGL pipeline).

Decision P1 (NUM scope): only pure-numeric tokens become NUM. IPs/addresses
collapse to NUM via the non-alphanumeric split; hex (0x..) and alphanumeric
identifiers (ddr3, core0, ...) are left intact -- the paper's conclusion lists
IP/file-location tokenization as future work, so we stay faithful to "numbers".
Adjust numeric_token() to change this.

Decision P2 (dedup key): consecutive dedup compares (normalized content, fail)
per node. Matching on the pair (not content alone) means a label transition on
otherwise-identical text is preserved, so no failure event is ever dropped.

Decision D14 (node granularity): BGL's location field (R02-M1-N0-C:J12-U11) is
chip-level, far finer than the paper's "node". Default groups at the node-card
level (R0X-MX-NX) -- the faithful mapping of the paper's compute-node concept --
which is applied BEFORE dedup so dedup runs at node granularity (paper orders it
group-by-node then dedup). Use --node-level full for the D14 sensitivity check.
NODE_LEVELS key functions are BGL-location-specific.
"""
from __future__ import annotations

import json
import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_PURE_NUM = re.compile(r"^[0-9]+$")


def _node_full(n):      return n                                          # R02-M1-N0-C:J12-U11
def _node_drop_chip(n): return n.split(":")[0]                            # R02-M1-N0-C
def _node_card(n):      return "-".join(n.split(":")[0].split("-")[:3])   # R02-M1-N0
def _node_first2(n):    return "-".join(n.split("-")[:2])                 # R02-M1

NODE_LEVELS = {"full": _node_full, "drop_chip": _node_drop_chip,
               "node_card": _node_card, "first2": _node_first2}


def numeric_token(tok):
    return bool(_PURE_NUM.match(tok))


def normalize(content):
    """lowercase -> split on non-alphanumeric -> pure-numeric -> NUM -> join."""
    toks = _NON_ALNUM.sub(" ", content.lower()).split()
    return " ".join("NUM" if numeric_token(t) else t for t in toks)


def parse_bgl(line):
    """Raw loghub BGL.log line -> (node, ts, fail, content) or None if malformed.
    Layout: Label Ts Date Node Time NodeRepeat Type Component Level Content..."""
    parts = line.rstrip("\n").split(maxsplit=9)
    if len(parts) < 10:
        return None
    label, ts_s, node, content = parts[0], parts[1], parts[3], parts[9]
    try:
        ts = float(ts_s)
    except ValueError:
        return None
    return node, ts, (label != "-"), content


PARSERS = {"bgl": parse_bgl}


def preprocess(path, dataset, node_level="node_card"):
    """Stream raw log -> normalized records; coarsen node key (D14); per-node
    sort + consecutive dedup."""
    parse = PARSERS.get(dataset)
    if parse is None:
        raise NotImplementedError(
            f"No verified parser for '{dataset}'. Confirm its field layout "
            f"against the raw .log header before adding a parser.")
    nodekey = NODE_LEVELS[node_level]
    by_node = {}
    n_lines = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            n_lines += 1
            parsed = parse(line)
            if parsed is None:
                continue
            node, ts, fail, content = parsed
            node = nodekey(node)                 # D14 coarsening (before dedup)
            norm = normalize(content)
            if not norm:
                continue
            by_node.setdefault(node, []).append((ts, fail, norm))
    records = []
    for node, rows in by_node.items():
        rows.sort(key=lambda r: r[0])
        prev = None
        for ts, fail, norm in rows:
            key = (norm, fail)
            if key == prev:                      # consecutive dedup (P2)
                continue
            records.append({"node": node, "ts": ts, "fail": fail, "content": norm})
            prev = key
    return records, n_lines, len(records)


def _write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FALL preprocess (raw log -> records)")
    ap.add_argument("--input", required=True, help="raw log file, e.g. BGL.log")
    ap.add_argument("--dataset", required=True, choices=["bgl", "thunderbird"])
    ap.add_argument("--out", required=True, help="output preprocessed JSONL")
    ap.add_argument("--node-level", default="node_card",
                    choices=list(NODE_LEVELS), help="node grouping granularity (D14)")
    a = ap.parse_args()
    recs, n_lines, n_kept = preprocess(a.input, a.dataset, a.node_level)
    _write_jsonl(recs, a.out)
    pct = (100.0 * n_kept / n_lines) if n_lines else 0.0
    fails = sum(1 for r in recs if r["fail"])
    nodes = len({r["node"] for r in recs})
    print(f"[{a.dataset}] node_level={a.node_level}  lines={n_lines}  "
          f"kept={n_kept} ({pct:.1f}% after dedup)  nodes={nodes}  failures={fails}")
