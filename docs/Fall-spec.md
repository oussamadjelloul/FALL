# FALL Reproduction Specification v3 (Thunderbird / BGL)

**Source:** Jeong et al., "FALL: Prior Failure Detection in Large Scale System Based on Language Model," _IEEE TDSC_ 22(1), 2025.
**Purpose:** Faithful baseline reproduction alongside the LogFiT work, with our corrections/improvements isolated as ablations. Restructured from v2 into two explicit blocks — **(A) Reproduction** and **(B) Ablations** — under the classification policy below.

---

## 0. Classification policy

A single test decides where each implementation choice goes:

> **Did the author commit to it?**
>
> - **Yes →** the reproduction uses _exactly what the author committed to_. Any alternative of ours (correction or improvement) goes to Block B as an ablation.
> - **No (genuinely silent) →** fill the gap with a default. The fill belongs _inside_ the reproduction; it does **not** need an ablation, because there is nothing to be unfaithful to.

Three refinements handle the messy cases:

1. **Typo vs. improvement.** "The author's equation is wrong" splits in two. If the author's _intent is unambiguous_ (a malformed equation whose correct form is fixed by the surrounding text and lineage), the reproduction silently uses the corrected form and documents it — **you do not ablate a typo**. If instead the author made a _defensible choice you would improve on_, the reproduction keeps theirs and your version becomes an ablation.
2. **Contradiction tiebreaker.** When the author commits to two incompatible things, "use what the author used" is undefined. Rule: **follow the author's described mechanism over inherited formal boilerplate.** The reproduction takes the narrated behaviour; the other branch becomes an ablation.
3. **Gap-fills are lineage-faithful first.** For silent items, the most faithful default is **whatever DATE [30] / ELECTRA [31] did** (FALL's stated lineage), then generic best practice. Justify a gap-fill as "what the method they built on does," not merely "reasonable."

**Tags used below:** `[PAPER]` author-specified · `[GAP→lineage]` gap-fill from DATE/ELECTRA · `[GAP→default]` gap-fill, generic default · `[TIEBREAK]` contradiction resolved by refinement 2 · `[TYPO]` typo repair per refinement 1.

---

## 1. Scope and success criteria

Headline results (Tables IV–IX) use a **private Samsung/Korea University corpus** (33M messages) — unreproducible. **Target = Tables X (Thunderbird) and XI (BGL)** from loghub, the only public experiments, reporting **DATE vs. FALL only**.

Because the private data is gone _and_ the paper publishes no TB/BGL pipeline, **absolute AUC matching is not the success criterion.** Reproduction succeeds if it reproduces the paper's **claims**:

1. FALL ≥ DATE on AUC across scenarios;
2. FALL avg-FP substantially below DATE;
3. sharpening and partial-token each improve over the DATE baseline (Table VIII direction);
4. performance roughly flat across the five scenarios (FALL's stated robustness).

(This mirrors the LogFiT reproduction stance: diagnose gaps to methodology, don't treat a digit mismatch as failure.)

---

# BLOCK A — REPRODUCTION

## A.1 Reproduction core `[PAPER]` — fixed, no freedom

| Item                            | Value                                                              | Source                         |
| ------------------------------- | ------------------------------------------------------------------ | ------------------------------ |
| Masking patterns `K`            | 50                                                                 | §IV-A                          |
| Sharpening temp `T`             | 1/2                                                                | §IV-A, Eq. 6                   |
| Partial-token fraction `1/n`    | 1/4 (`n=4`)                                                        | §IV-A, Eq. 7                   |
| Loss weights `μ / λ`            | 50 (RTD) / 100 (RMD)                                               | §IV-A, Eq. 2                   |
| Batch / steps                   | 128 / 20,000                                                       | §IV-A                          |
| Optimizer                       | AdamW                                                              | §IV-A                          |
| Seeds                           | 3 (mean ± std)                                                     | §IV-A                          |
| Window / stride                 | 30 / 15                                                            | §III-A                         |
| Max sequence length             | 128                                                                | §IV-A                          |
| Generator / discriminator depth | 1 layer / 4 layers                                                 | §IV-A                          |
| Heads                           | RTD (token binary) + RMD (50-class)                                | §III-B                         |
| Preprocessing                   | lowercase, strip punctuation, numbers→`NUM`, consecutive dedup     | §III-A                         |
| Grouping                        | per node, chronological                                            | §III-A                         |
| Oversized inputs                | **segment**, processed sequentially (not truncate)                 | §III-B                         |
| Training data                   | normal-only                                                        | §IV-A, Table III               |
| Test data                       | normal + abnormal                                                  | §IV-A                          |
| Inference                       | discriminator + RTD head only                                      | §III-C                         |
| Score pipeline                  | sharpen (Eq. 6) → top-(1/n) (Eq. 7)                                | §III-C                         |
| Scenarios                       | 5 × (lead, interval-end): (10,30)(30,60)(60,180)(180,300)(300,420) | Table VIII                     |
| Metrics                         | AUC-ROC + average number of FP                                     | §IV-B                          |
| Generator behaviour             | **random / uniform replacement from vocab**                        | §III-B `[TIEBREAK]` — see note |

**`[TIEBREAK]` on the generator:** §III-B narrates "a random generator … uniformly selecting words from the vocabulary," while Eq. 2/3 carry an `L_MLM` term implying a trained generator. The reproduction follows the **narrated mechanism** (random, untrained → `L_MLM` contributes no gradient → effective loss `μ·L_RTD + λ·L_RMD`). The trained-generator reading is **B.2.1**.

**`[TYPO]` on Eq. 4:** the paper prints `P_G(t^mask_i | · ; θ_D)`, mixing the generator symbol with the discriminator parameters. RTD is unambiguously per-token binary detection (text + ELECTRA/DATE lineage). The reproduction implements **per-token sigmoid + BCE**; no ablation.

## A.2 Gap-fills — author silent, filled lineage-first, _inside the reproduction_

| #   | Item                            | Fill                                                                                                                                                                              | Tag                                                     | Justification                                                                                                                                                                                                                                     |
| --- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D2  | Tokenizer + unit                | WordPiece trained on the NUM-normalized train split, ≈8–16k vocab, **subword**                                                                                                    | `[GAP→default]`                                         | Paper names no tokenizer for TB/BGL; "18,779 words" is a private-corpus word count, not a spec. Report tokens-per-log + vocab size as artifacts; pin vocab from data after first preprocessing pass. Stock `bert-base-uncased` = smoke test only. |
| D3  | Transformer dims                | `d_model=256`, `n_heads=4`, `d_ff=1024`, `dropout=0.1`, `gelu`, `max_pos=128`                                                                                                     | `[GAP→default]`                                         | Paper gives only layer counts. Small config suits limited vocab / short logs.                                                                                                                                                                     |
| D4  | LR / schedule                   | `lr=1e-4`, linear warmup 1,000, `weight_decay=0.01`                                                                                                                               | `[GAP→default]`                                         | Only "AdamW" given.                                                                                                                                                                                                                               |
| D5  | Per-pattern mask rate           | `r=0.15`                                                                                                                                                                          | `[GAP→lineage]`                                         | ELECTRA/DATE masking rate.                                                                                                                                                                                                                        |
| D7  | Avg-FP threshold                | 99th-pct of val-normal scores; plot FP-vs-threshold curve                                                                                                                         | `[GAP→default]`                                         | Paper confirms a threshold exists (§IV-C) but states no rule.                                                                                                                                                                                     |
| D8  | Split                           | chronological 70/15/15 per dataset, then fixed-seed shuffle **within each partition independently**                                                                               | `[GAP→default]` (ratios) / `[PAPER]` (mechanism, §IV-A) | Private-set numbers don't transfer. Never shuffle across partition boundaries (would re-cross the time split and leak). Training partition normal-only → its merge is a no-op.                                                                    |
| D9  | Special-token layout            | single `[CLS]` front, single trailing `[SEP]`, then `[PAD]`; no inter-log separators                                                                                              | `[GAP→default]`                                         | Paper names only `[CLS]`/`[PAD]`. Report effective-`L` distribution so the assumption is auditable.                                                                                                                                               |
| D10 | Cross-segment aggregation       | **pool RTD probs across all segments → one length-`L` vector → sharpen → top-`m`** = one score/window. Training: each segment is an independent example with its own pattern `k`. | `[GAP→default]`                                         | §III-B says "processed sequentially" but never how segments recombine; labels are per-window. 30 logs routinely exceed 126 tokens, so this fires on nearly every window — a robustness check is in B.2.3.                                         |
| D11 | OOV handling                    | abnormal/novel tokens → subword-decomposed by the trained tokenizer; `[UNK]` only when no subword path exists; report `[UNK]` rate                                                | `[GAP→default]`                                         | Tokenizer trained on normal logs; subword fallback avoids collapsing all novelty to one `[UNK]` signal.                                                                                                                                           |
| D12 | Lead-gap failures               | failure in `(t_end, t_end+ℓ]` but not in the target interval → window **normal** (by default)                                                                                     | `[GAP→default]`                                         | Paper (Fig. 2) silent.                                                                                                                                                                                                                            |
| D13 | RMD pooling                     | `[CLS]`-pooled → 50-way head                                                                                                                                                      | `[GAP→lineage]`                                         | Standard sequence-level pool in BERT/ELECTRA.                                                                                                                                                                                                     |
| —   | RTD label on coincidental match | random draw == original ⇒ label 0                                                                                                                                                 | `[GAP→lineage]`                                         | ELECTRA convention.                                                                                                                                                                                                                               |

> **Provenance:** TB/BGL field layouts and the `label != '-'` failure rule are **loghub conventions**, not the paper. The whole act of porting the temporal scenario-labeling onto TB/BGL is our reconstruction — the paper only states it was "applied to these datasets."
> **Acknowledged but unreproducible:** §III-A also describes a second, _expert-driven_ dedup of duplicate sequences/windows. Not algorithmically specified → we apply only consecutive message-level dedup and flag this as a deliberate divergence.

## A.3 Pipeline mechanics

**Preprocess (per line):** extract `(label, epoch_ts, node, content)`; normalize content (lowercase → strip punctuation → numbers→`NUM` → collapse whitespace); failure = `label != '-'`; drop a log whose normalized content equals the immediately preceding log's within the node stream.

**Window + label (per node, deduped stream):** window=30, stride=15, `t_end` = timestamp of `x_{i+29}`. For scenario `(ℓ, e)`: `abnormal` iff ∃ failure on the same node with `timestamp ∈ (t_end+ℓ, t_end+e]`, else `normal` (D12 governs the lead gap).

**Tokenize + lay out (D2, D9):** concatenate the 30 normalized logs → `[CLS] tok_1 … tok_h [SEP] [PAD]…` to 128; if `h+2 > 128` segment into ≤128 chunks (aggregate per D10); specials never masked, never scored.

**Masking (A.1, D5):** 50 frozen binary patterns (`RNG(seed,k)`, rate 0.15) over non-special positions; per example sample `k`, RMD target = `k`.

**Forward (training):** mask `P_k` → uniform random replacement `r_p` (A.1 tiebreaker) → `rtd_label[p] = 1 if r_p != original[p] else 0`. Discriminator (4-layer) → RTD (sigmoid BCE over non-`[CLS]`, non-PAD; Eq. 4 typo-repaired) + RMD (`[CLS]`-pooled CE vs `k`). Loss `μ·L_RTD + λ·L_RMD`. AdamW per D4, batch 128, 20k steps, 3 seeds.

**Score (inference):** pool RTD `P(replaced)` across all segments → length-`L` vector → sharpen `s_i = p_i² / Σ p_j²` → sort desc → `m = ⌊L/n⌋`, `n=4` → `score = (n/m)·Σ_{i=1..m} sorted_i` = **mean of top-`m` sharpened probs × constant `n`**. Do not re-divide by `L`.

**Evaluate:** first report **per-scenario positive-window counts per dataset**; flag any scenario with too few positives (likely short-lead on TB/BGL) as non-comparable. Then AUC-ROC (threshold-free) + avg-FP at D7 threshold (3 seeds). Absolute FP counts are not cross-corpus comparable.

---

# BLOCK B — ABLATIONS

Two kinds, kept separate for the viva trail: **B.1** replicates the _paper's own_ ablations (part of a faithful reproduction); **B.2** is _our_ deltas (corrections / improvements / robustness). Cost is flagged as **re-score** (reuse one trained checkpoint, ~free) or **retrain** (full 20k-step run × scenarios × datasets × seeds).

## B.1 Replication of the paper's ablations (Table VIII direction)

| #     | Variant                                         | vs.  | Cost      | Expected (paper)                                            |
| ----- | ----------------------------------------------- | ---- | --------- | ----------------------------------------------------------- |
| B.1.1 | DATE baseline (no sharpening, no partial-token) | FALL | retrain\* | FALL ≥ DATE, much lower FP (Tables VI/VII, X, XI)           |
| B.1.2 | DATE + sharpening only                          | DATE | re-score  | small AUC gain (~+0.0008/scenario, Table VIII)              |
| B.1.3 | DATE + partial-token only                       | DATE | re-score  | larger AUC gain (~+0.004/scenario) + big FP drop (Table IX) |

\*B.1.1 shares FALL's training objective; the difference is at scoring, so it can be **re-score** if the DATE scoring (mean over all tokens, no sharpening) is applied to the same checkpoint. Confirm whether DATE in the paper used an identical objective; if it differs, retrain.

## B.2 Our deltas (corrections / improvements / robustness)

| #     | Variant                                          | What it tests                                            | Cost        | Decision rule                                                                                                                                                          |
| ----- | ------------------------------------------------ | -------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B.2.1 | **Trained 1-layer MLM generator + live `L_MLM`** | the generator contradiction (A.1 tiebreaker)             | **retrain** | If reproduction (random gen) underperforms Tables X/XI and B.2.1 closes the gap → attribute the gap to D1, not to D3/D4 config. The single most consequential variant. |
| B.2.2 | Eq. 7 as **raw top-`m` sum** (no averaging)      | OCR hedge on the mean reading (D6)                       | re-score    | Fig. 4's `m/n` bound + `n/m` prefactor favor a mean; run both to confirm the reading isn't load-bearing. Affects ranking only if scores aren't length-normalized.      |
| B.2.3 | Segment aggregation = **max-of-chunk-scores**    | robustness of the D10 default (not an improvement claim) | re-score    | Since segmentation fires on nearly every window, show the headline AUC doesn't hinge on pool-vs-max.                                                                   |

> Everything not listed in Block B is a reproduction default, not a delta. Resist adding ablations for nuisance gap-fills (D3/D4/D5/D9/D11/D12/D13) — those are author-silent fills, and a _bounded sensitivity note_ (if cheap) is the most they warrant, never a full ablation.

---

## Module layout

```
fall-repro/
  configs/    bgl.yaml  thunderbird.yaml          # one block-A config + B variants as overrides
  src/
    preprocess.py  windowing.py  tokenizer_train.py  masking.py
    model.py       # disc 4L + RTD/RMD; optional 1L gen for B.2.1
    train.py       # μ/λ loss; --generator {random,mlm}
    score.py       # segment-pool → sharpen → top-m; --agg {pool,max}; --score {mean,sum}
    evaluate.py    # feasibility counts + AUC + avg-FP@D7
  docs/  fall-repro-spec-v3.md
```

---

## Appendix: unverifiable-from-paper (treat as assumptions)

`d_model/heads/d_ff/dropout` (D3) · LR/warmup/wd (D4) · mask rate (D5) · tokenizer type/size/unit + OOV (D2/D11) · avg-FP threshold (D7) · TB/BGL split ratios + 3-way val (D8) · `[SEP]`/separators/special-token budget (D9) · RMD pooling (D13) · cross-segment aggregation (D10) · lead-gap handling (D12) · generator trained-vs-not (A.1; Eq. 2/3 vs §III-B contradict) · Eq. 7 mean-vs-sum (D6; figure favors mean, OCR can't fully confirm).

**Carried-over review verdicts:** Gemini C1 (Eq. 7 prefactor distorts AUC) — rejected; the per-window `1/m` is the intended length-normalizing average, `n` is a harmless global constant. Gemini C4 (no 3-way val) — downgraded; a held-out val is required for D7. Gemini's cited Table III counts could not be verified from the source.
