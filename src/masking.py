"""masking.py - FALL reproduction. K frozen masking patterns + random-generator
corruption (spec A.1 / D1 / D5).

A Masker holds K=50 frozen binary patterns over a fixed max_len, generated once
with a seed (mask_rate=0.15, D5; position 0 / [CLS] never masked). corrupt()
picks one pattern per example, masks the eligible (content) positions, and
replaces them with uniformly random vocabulary ids -- D1 primary: a random,
untrained generator. Replacement draws exclude the first `num_special` ids
([PAD],[UNK],[CLS],[SEP]) so substitutes are "words from the vocabulary".

corrupt() returns:
  corrupt_ids [B,L] - input with masked positions replaced
  rtd_labels  [B,L] - 1 where a position was replaced AND changed, else 0
                      (ELECTRA convention: a coincidental match stays 0)
  rmd_targets [B]   - which pattern (0..K-1) was applied (RMD target)
"""
from __future__ import annotations

import torch

K_PATTERNS = 50
MASK_RATE = 0.15
MAX_LEN = 128
PATTERN_SEED = 20240601


class Masker:
    def __init__(self, max_len=MAX_LEN, k=K_PATTERNS, mask_rate=MASK_RATE,
                 seed=PATTERN_SEED):
        self.k = k
        g = torch.Generator().manual_seed(seed)
        prob = torch.full((k, max_len), float(mask_rate))
        prob[:, 0] = 0.0                                   # never mask [CLS]@0
        self.patterns = torch.bernoulli(prob, generator=g).bool()   # [K, max_len]

    def corrupt(self, input_ids, maskable, vocab_size, num_special=4,
                generator=None):
        """input_ids [B,L] long; maskable [B,L] bool (content positions eligible
        for masking). Returns (corrupt_ids, rtd_labels, rmd_targets)."""
        B, L = input_ids.shape
        dev = input_ids.device
        k = torch.randint(0, self.k, (B,), generator=generator)
        pat = self.patterns[k, :L].to(dev)                          # [B,L]
        repl = torch.randint(num_special, vocab_size, (B, L),
                             generator=generator).to(dev)
        apply = pat & maskable
        corrupt = torch.where(apply, repl, input_ids)
        rtd = (apply & (corrupt != input_ids)).long()
        return corrupt, rtd, k.to(dev)
