"""train.py - FALL reproduction, phase-2 training.

Reads windowed train.jsonl, tokenizes + lays out [CLS] content [SEP] [PAD] (D9),
segments windows with >126 content tokens into independent examples (D10), and
trains the discriminator self-supervised with random-generator corruption (D1)
for 20k steps x 3 seeds, AdamW + linear warmup (D4). L_MLM is inactive (D1).

Usage:
  python src/train.py --windows-dir data/processed/bgl --scenario s1 \
      --tokenizer data/processed/bgl/bgl_tok_8000.json \
      --out results/ckpt/bgl --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from tokenizers import Tokenizer

from masking import Masker
from model import Discriminator, fall_loss

MAX_LEN = 128
SPECIALS = ("[CLS]", "[SEP]", "[PAD]")


def load_tokenizer(path):
    tok = Tokenizer.from_file(path)
    ids = {s: tok.token_to_id(s) for s in SPECIALS}
    for s, i in ids.items():
        if i is None:
            raise ValueError(f"tokenizer missing special token {s}")
    return tok, ids["[CLS]"], ids["[SEP]"], ids["[PAD]"]


class SegmentPool:
    """Flat pool of <=128-token segments (D9 layout, D10 segmentation)."""

    def __init__(self, windows, tok, cls, sep, pad, max_len=MAX_LEN):
        budget = max_len - 2                       # room for [CLS] + [SEP]
        rows, clens = [], []
        for w in windows:
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
        self.input_ids = torch.tensor(rows, dtype=torch.long)        # [N,L]
        self.content_len = torch.tensor(clens, dtype=torch.long)     # [N]
        self.pos = torch.arange(max_len).unsqueeze(0)                # [1,L]

    def __len__(self):
        return self.input_ids.size(0)

    def batch(self, idx):
        """idx: LongTensor of row indices -> (input_ids, attn, maskable, rtd_mask)."""
        ii = self.input_ids[idx]                          # [b,L]
        cl = self.content_len[idx].unsqueeze(1)           # [b,1]
        attn = (self.pos <= cl + 1).long()                # [CLS]..content..[SEP]
        maskable = (self.pos >= 1) & (self.pos <= cl)     # content only
        rtd_mask = attn.clone()
        rtd_mask[:, 0] = 0                                # RTD loss excludes [CLS]
        return ii, attn, maskable, rtd_mask

    @classmethod
    def from_jsonl(cls, path, tok, c, s, p, max_len=MAX_LEN):
        with open(path, encoding="utf-8") as f:
            windows = [json.loads(line) for line in f if line.strip()]
        return cls(windows, tok, c, s, p, max_len)


def train_one(pool, vocab, pad, args, seed, device):
    torch.manual_seed(seed)
    model = Discriminator(vocab, d_model=args.d_model, nhead=args.nhead,
                          num_layers=args.layers, dim_ff=args.dim_ff,
                          max_len=MAX_LEN, k_patterns=args.k, pad_id=pad).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = max(1, args.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / warm))
    masker = Masker(max_len=MAX_LEN, k=args.k)
    gen = torch.Generator().manual_seed(seed)            # sampling + corruption
    n = len(pool)
    model.train()
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch,), generator=gen)
        ii, attn, maskable, rtd_mask = pool.batch(idx)
        ii, attn = ii.to(device), attn.to(device)
        maskable = maskable.to(device)
        rtd_mask = rtd_mask.to(device)
        if args.generator == "mlm":
            raise NotImplementedError(
                "trained-MLM generator (ablation B.2.1) not built yet")
        corrupt, rtd, rmd = masker.corrupt(ii, maskable, vocab, generator=gen)
        rtd, rmd = rtd.to(device), rmd.to(device)
        rtd_logits, rmd_logits = model(corrupt, attn)
        loss, rtd_l, rmd_l = fall_loss(rtd_logits, rmd_logits, rtd, rmd,
                                       rtd_mask, args.mu, args.lam)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  seed{seed} step{step:6d}  loss={loss.item():.3f} "
                  f"rtd={rtd_l:.4f} rmd={rmd_l:.4f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"{args.scenario}_seed{seed}.pt")
    torch.save({"model": model.state_dict(),
                "config": {"vocab": vocab, "d_model": args.d_model,
                           "nhead": args.nhead, "layers": args.layers,
                           "dim_ff": args.dim_ff, "k": args.k,
                           "max_len": MAX_LEN, "pad": pad},
                "scenario": args.scenario, "seed": seed}, ckpt)
    print(f"  saved {ckpt}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="FALL phase-2 training")
    ap.add_argument("--windows-dir", required=True, help="e.g. data/processed/bgl")
    ap.add_argument("--scenario", required=True, help="s1..s5")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True, help="checkpoint dir")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--mu", type=float, default=50.0)
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--d-model", dest="d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim-ff", dest="dim_ff", type=int, default=1024)
    ap.add_argument("--generator", choices=["random", "mlm"], default="random")
    ap.add_argument("--log-every", dest="log_every", type=int, default=500)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok, cls, sep, pad = load_tokenizer(args.tokenizer)
    vocab = tok.get_vocab_size()
    train_path = os.path.join(args.windows_dir, args.scenario, "train.jsonl")
    pool = SegmentPool.from_jsonl(train_path, tok, cls, sep, pad)
    print(f"[{args.scenario}] segments={len(pool)} vocab={vocab} "
          f"device={args.device} generator={args.generator}", flush=True)
    for seed in (int(x) for x in args.seeds.split(",")):
        train_one(pool, vocab, pad, args, seed, args.device)


if __name__ == "__main__":
    main()
