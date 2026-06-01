"""tokenizer_train.py - FALL reproduction (D2 / gate 3).

Trains a WordPiece tokenizer on the chronological TRAINING portion of the
preprocessed records (leak-free: val/test text is never shown to the trainer),
then reports the stats needed to pin the D2 vocab size:
  - natural vocabulary (unique whitespace tokens in the train text)
  - tokens-per-log distribution at the chosen vocab size
  - expected window length (30 x mean tpl) vs the 128 cap -> segmentation rate
  - [UNK] rate on the held-out (val) portion (the D11 OOV signal)

Input: preprocessed JSONL from preprocess.py ({"node","ts","fail","content"}).
Requires the `tokenizers` package (HF). Special tokens: [PAD] [UNK] [CLS] [SEP].

TRAIN_FRAC / VAL_FRAC below MUST match windowing.SPLIT so the tokenizer's train
text lines up with the model's training partition (record-level approximation of
the window-level chronological split).
"""
from __future__ import annotations

import json
import statistics

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15        # must match windowing.SPLIT
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
WINDOW = 30
MAX_LEN = 128
SPECIAL_BUDGET = 2                        # [CLS] + [SEP]  (D9)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def chronological_text_split(records):
    """Sort by ts, cut train/val by TRAIN_FRAC/VAL_FRAC. Record-level
    approximation of the window-level split in windowing.py."""
    recs = sorted(records, key=lambda r: r["ts"])
    n = len(recs)
    n_tr = int(n * TRAIN_FRAC)
    n_va = int(n * VAL_FRAC)
    train = [r["content"] for r in recs[:n_tr]]
    val = [r["content"] for r in recs[n_tr:n_tr + n_va]]
    return train, val


def natural_vocab(lines):
    v = set()
    for ln in lines:
        v.update(ln.split())
    return v


def train_tokenizer(train_lines, vocab_size, out_path):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=SPECIALS,
        continuing_subword_prefix="##")
    tok.train_from_iterator(train_lines, trainer)
    tok.save(out_path)
    return tok


def tokens_per_log(tok, lines):
    return [len(tok.encode(ln).tokens) for ln in lines]


def unk_rate(tok, lines):
    tot = unk = 0
    for ln in lines:
        toks = tok.encode(ln).tokens
        tot += len(toks)
        unk += sum(1 for t in toks if t == "[UNK]")
    return (unk / tot) if tot else 0.0


def run(input_path, vocab_size, out_path):
    records = list(_read_jsonl(input_path))
    train, val = chronological_text_split(records)
    nat = natural_vocab(train)
    print(f"records={len(records)}  train_logs={len(train)}  val_logs={len(val)}")
    print(f"natural vocab (unique whitespace tokens in train) = {len(nat)}")

    tok = train_tokenizer(train, vocab_size, out_path)
    tpl = tokens_per_log(tok, train)
    mean_tpl = statistics.mean(tpl) if tpl else 0.0
    idx = max(0, int(0.95 * len(tpl)) - 1)
    p95 = sorted(tpl)[idx] if tpl else 0
    exp_win = WINDOW * mean_tpl
    cap = MAX_LEN - SPECIAL_BUDGET
    seg = (exp_win / cap) if cap else 0.0
    uk = unk_rate(tok, val)

    print(f"vocab_size requested={vocab_size}  actual={tok.get_vocab_size()}")
    print(f"tokens/log: mean={mean_tpl:.2f}  "
          f"median={statistics.median(tpl):.0f}  p95={p95}" if tpl else "tokens/log: n/a")
    print(f"expected window tokens = 30 x {mean_tpl:.2f} = {exp_win:.0f}  "
          f"(cap {cap}) -> ~{seg:.1f} segments/window")
    print(f"[UNK] rate on val = {100.0 * uk:.3f}%")
    print(f"saved tokenizer -> {out_path}")
    return tok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FALL WordPiece trainer (gate 3 / D2)")
    ap.add_argument("--input", required=True, help="preprocessed JSONL")
    ap.add_argument("--vocab-size", type=int, default=12000)
    ap.add_argument("--out", default="tokenizer.json")
    a = ap.parse_args()
    run(a.input, a.vocab_size, a.out)
