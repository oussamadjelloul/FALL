"""preprocess.py - FALL reproduction (BGL).

Parses raw loghub BGL logs into the record contract consumed by windowing.py:
    {"node": str, "ts": float, "fail": bool, "content": str}

Pipeline (spec v3 A.3): lowercase -> non-alphanumeric to space -> pure-numeric
tokens to NUM -> collapse whitespace. failure = label != '-'. Per-node
consecutive dedup.

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
"""
from __future__ import annotations

import json
import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_PURE_NUM = re.compile(r"^[0-9]+$")


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


def preprocess(path, dataset):
    """Stream raw log -> normalized records; per-node sort + consecutive dedup."""
    parse = PARSERS.get(dataset)
    if parse is None:
        raise NotImplementedError(
            f"No verified parser for '{dataset}'. Confirm its field layout "
            f"against the raw .log header before adding a parser.")
    by_node = {}
    n_lines = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            n_lines += 1
            parsed = parse(line)
            if parsed is None:
                continue
            node, ts, fail, content = parsed
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
            if key == prev:        # consecutive dedup (P2)
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
    a = ap.parse_args()
    recs, n_lines, n_kept = preprocess(a.input, a.dataset)
    _write_jsonl(recs, a.out)
    pct = (100.0 * n_kept / n_lines) if n_lines else 0.0
    fails = sum(1 for r in recs if r["fail"])
    nodes = len({r["node"] for r in recs})
    print(f"[{a.dataset}] lines={n_lines}  kept={n_kept} ({pct:.1f}% after dedup)  "
          f"nodes={nodes}  failures={fails}")
