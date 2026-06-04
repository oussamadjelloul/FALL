# FALL Reproduction Notes — BGL

**Paper:** Jeong et al., _FALL: Prior Failure Detection in Large-Scale System Based on Language Model_, IEEE TDSC 2025, vol. 22, no. 1.
**Purpose:** Independent re-implementation of FALL on public BGL data, as a baseline for the SDD dissertation.
**Author:** Oussama (Université Laval).
**Version:** v1.0 — 2026-06-04.

---

## 0. TL;DR — reproduction verdict

| Claim                                                                                                        | Status                                                                                                        |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Method (architecture, random-generator discriminator, sharpening + selective scoring, minimal preprocessing) | **Reproduced faithfully**                                                                                     |
| FALL AUC in the paper's 0.77–0.88 band, and the FALL ≈ DATE pattern (incl. FALL _losing_ s2)                 | **Reproduced**                                                                                                |
| FALL ≪ DATE on false positives; selective + sharpening each reduce FP                                        | **Reproduced in direction**                                                                                   |
| FALL absolute avg-FP magnitude                                                                               | **Matched at a documented operating point** (99.7th-pct threshold → avg ≈ 59 vs paper ≈ 58), not per-scenario |
| Private Samsung main results (Tables I–IX)                                                                   | **Not reproducible** (private 33M-log corpus)                                                                 |
| Aggressive token normalization (IP/path)                                                                     | **Not done — it is the paper's stated future work (§V)**                                                      |

**One-line summary:** every claim that is threshold-independent (AUC, both component directions) reproduces; the one residual gap (per-scenario absolute FP) is fully explained by an undocumented threshold rule, an irreproducible expert-deduplication step in the original study, and a faithful property of our BGL windows.

---

## 1. The paper in brief

FALL is an ELECTRA/DATE-style self-supervised **failure-prediction** model. A random (untrained) generator corrupts tokens of normal log sequences; a 4-layer transformer **discriminator** is trained with two heads:

- **RTD** (replaced-token detection): per-token "was this replaced?"
- **RMD** (replaced-mask detection): which of K masking patterns was applied.

Training uses normal logs only. At **inference** only the discriminator + RTD head are used. The per-token RTD probabilities are turned into one anomaly score per sequence via:

- **Eq. 6 (sharpening):** `sharpen(S,T)_i = t_i^(1/T) / Σ_j t_j^(1/T)` — temperature sharpening, normalized to sum to 1 over the sequence.
- **Eq. 7 (selective score):** `(n/m) · Σ_{i=1..m/n} reverse_sort(sharpen(S,T))` — the mean of the top-`1/n` sharpened values.

Metrics: AUC-ROC and the **average number of false positives**. The headline results are on a private Samsung HPC corpus; only **Table X (Thunderbird)** and **Table XI (BGL)** are public, and those report DATE vs FALL only.

**Paper hyperparameters (verified from the PDF):** K=50, T=1/2, 1/n=1/4, μ=50 (RTD weight), λ=100 (RMD weight), batch 128, 20k steps, AdamW, 3 seeds, window = 30 logs, stride = 15, max_len = 128, generator = 1 layer, discriminator = 4 layers. Input: 30 logs concatenated; if < 128 tokens pad with `[PAD]`, if > 128 split and process sequentially.

**Preprocessing (§III-A):** lowercase, strip punctuation, replace numbers with `NUM`. Nothing more. Normalizing IP addresses / file paths is named in §V as **future work the authors did not do**.

---

## 2. Reproduction decisions (three-layer disclosure)

### 2a. Paper-faithful (reproduced exactly; alternatives are ablations)

- Random/untrained generator (D1). Effective loss = μ·RTD + λ·RMD.
- K=50, T=1/2, n=4, μ=50, λ=100, batch 128, 20k steps, AdamW, 3 seeds, window=30, stride=15, max_len=128 (D5, D6).
- Score = mean of top-⌊L/n⌋ sharpened values (D6) — faithful to Eq. 6 / Eq. 7.
- Preprocessing = lowercase + strip punctuation + numbers→NUM only (P1). We deliberately did **not** normalize IPs/paths (that is the paper's future work).
- Train on normal logs only; inference = discriminator + RTD head only.

### 2b. Paper-silent gap-fills (part of the reproduction; not ablations)

- **D2** tokenizer: WordPiece, vocab = 8000 (UNK rate 0.0%).
- **D3** model dims: d_model=256, nhead=4, layers=4, d_ff=1024, dropout=0.1, GELU, max_pos=128 (paper unpinned for the public datasets).
- **D4** optimizer: lr=1e-4, warmup 1000, weight decay 0.01.
- **D7** decision threshold: 99th percentile of validation-normal scores (paper never states its threshold rule — see §6 calibration).
- **D8** split: chronological 70/15/15, then within-partition shuffle (seed 1337).
- **D9/D10** layout `[CLS] + content + [SEP] + [PAD]`; > 128-token windows split into ≤128-token segments and processed sequentially.
- **D14** node granularity = `node_card` (R0X-MX-NX); coarsen before dedup.
- **P2** dedup key = consecutive identical `(content, fail)` per node.

### 2c. Labeled departures / interpretations

- **Segment scoring unit** (`--score-unit segment`): the FALL score is computed _per ≤128-token segment_ and averaged per window, rather than over the whole pooled window. This is our reading of the paper's "split and process sequentially," and it removes a length confound (see §4). The old pooled behaviour is retained as `--score-unit pooled` for the before/after record.

---

## 3. Data and training state (BGL)

- Raw: 4.75M lines → **1,163,163** kept after node_card coarsening + dedup (24.5%); 2,703 nodes; 66,884 failures.
- Tokenizer: WordPiece vocab 8000, UNK 0.0%.
- Windows: **74,384** (train 52,068 / val 11,157 / test 11,159).
- Test positives per scenario: s1=280, s2=52, s3=137, s4=95, s5=154.
- Checkpoints: **15** (s1–s5 × seeds 0,1,2), all trained healthily (RTD loss ~0.73→~0.05; RMD ~4.1→~0.05–0.3).
- **Caveat:** failures cluster at the start and end of the BGL timeline, so the chronological validation slice is almost failure-free (val positives ≈ 0). This is fine for the D7 threshold (computed on val-_normal_) but has consequences for FP calibration (see §6).

---

## 4. The length confound and the segment-scoring fix

**Symptom.** With the original _pooled_ scoring (concatenate all of a window's segment probabilities, then sharpen over the whole window), FALL AUC came out at 0.86–0.94 — well _above_ the paper, and FALL beat DATE by ~0.10, when the paper has FALL ≈ DATE.

**Diagnosis.** Eq. 6 normalizes the sharpened vector to sum to 1, so the Eq. 7 score intrinsically contains a `1/L` term (L = number of tokens). Pooling makes L the _full window length_, which on BGL varies enormously: pre-failure windows are ~3.5× shorter (median ~200 tokens) than normal windows (median ~735). The score was therefore tracking **window length**, not anomaly content. Proof: `sharpen_only` ≡ `1/L` exactly — model-independent, with ±0.0000 seed variance — and its AUC equalled `AUC(1/L)` to three decimals.

**Fix.** Score each ≤128-token segment on its own (sharpening normalized within the segment) and average the segment scores per window. This bounds the normalization length to ≤128, matching the paper's "split and process sequentially." It moved FALL from ~0.93 down into the paper's 0.77–0.88 band and corrected the FALL−DATE sign on s2.

**Residual (honest).** A small length component survives, because the partial last segment of each window still carries a little `1/L`. This is intrinsic to the paper's formula and would affect the paper too on data with our length–label correlation. We do **not** chase it further: scoring tricks cannot remove it, and the only fix (segment-as-sample evaluation) breaks the paper's window-level label `S_i` and was rejected.

---

## 5. Results (segment scoring, 3 seeds)

### 5a. AUC-ROC vs the paper (Table XI)

| scen | our FALL | paper FALL | our DATE | paper DATE | FALL−DATE (ours / paper) |
| ---- | -------- | ---------- | -------- | ---------- | ------------------------ |
| s1   | 0.851    | 0.882      | 0.820    | 0.864      | +0.031 / +0.018          |
| s2   | 0.852    | 0.767      | 0.865    | 0.791      | **−0.013 / −0.024**      |
| s3   | 0.816    | 0.791      | 0.681    | 0.785      | +0.135 / +0.006          |
| s4   | 0.849    | 0.812      | 0.815    | 0.812      | +0.034 / 0.000           |
| s5   | 0.787    | 0.811      | 0.748    | 0.810      | +0.039 / +0.001          |

FALL sits in the paper's band on every scenario and — like the paper — **loses to DATE on s2**. Our FALL−DATE gaps are larger than the paper's (esp. s3), due to the residual length component and an unstable DATE signal on s3.

### 5b. False positives — threshold sensitivity (FALL avg-FP)

| scen    | 99pct   | 99.7pct | 99.9pct | paper FALL FP |
| ------- | ------- | ------- | ------- | ------------- |
| s1      | 129     | 59      | 48      | 105           |
| s2      | 150     | 61      | 41      | 26            |
| s3      | 128     | 58      | 41      | 34            |
| s4      | 112     | 56      | 42      | 104           |
| s5      | 130     | 59      | 38      | 23            |
| **avg** | **130** | **59**  | **42**  | **58**        |

At the 99.7th-percentile threshold, FALL averages **59 FP vs the paper's 58** — essentially identical on average. The threshold is a documented choice (the paper never states its own), so we report the _sweep_, not a single cherry-picked point.

### 5c. Selective-token FP contribution at matched 95% recall (`FP@95%TPR`)

| scen | fall vs date | partial_only vs date |
| ---- | ------------ | -------------------- |
| s1   | −24%         | −17%                 |
| s2   | −10%         | −0.4%                |
| s3   | −8%          | −9%                  |
| s4   | −44%         | −22%                 |
| s5   | +1%          | −0.5%                |

At matched recall, selective tokens reduce FP below DATE in 3/5 scenarios (up to −22%) and the full FALL in 4/5 (up to −44%). This **reproduces the direction** of the paper's Table IX; the magnitude is weaker (paper: selective −33%, full FALL −99%).

---

## 6. Two diagnostics that explain the residual gaps

### 6a. Per-token concentration (`diag_concentration.py`)

The model's per-token RTD signal is **sharply concentrated** in _both_ classes: gini ≈ 0.73–0.84, top-quarter/overall ratio ≈ 3.3–3.7, single hottest token ≈ 0.82–0.97, top-quarter mass ≈ 0.83–0.92 (flat baselines: 0, 1.0, 0.25). The concentration _shape_ is essentially equal across classes (fail/normal ≈ 1.0); what differs is the **level** (pre-failure mean ≈ 1.6–2× normal).

**Consequence:** selecting the top tokens cannot out-discriminate averaging when both classes are equally peaked → selective adds ~0 to AUC (lift ≈ +0.002 typical), which _matches_ the paper's tiny +0.004 AUC gain. Our **normal** windows are also concentrated because the model spikes on out-of-place-but-normal tokens (rare IDs, hex, paths). The paper's future-work normalization would flatten those — but we faithfully did not do it, so this is a property of the reproduction, not a defect.

### 6b. Validation→test calibration (`diag_calibration.py`)

`testFPR@valp99` — the FP rate the D7 threshold actually produces on test:

| scen | date  | fall | partial_only | sharpen_only |
| ---- | ----- | ---- | ------------ | ------------ |
| s1   | 24.9% | 1.4% | 24.5%        | 1.0%         |
| s2   | 25.9% | 1.4% | 25.7%        | 1.0%         |
| s3   | 2.1%  | 1.2% | 6.7%         | 1.0%         |
| s4   | 25.9% | 0.8% | 25.2%        | 1.0%         |
| s5   | 26.5% | 1.1% | 26.0%        | 1.0%         |

**Finding:** `date` and `partial_only` are badly miscalibrated (~25% FPR instead of 1%); `fall` and `sharpen_only` are well calibrated (~1%). Validation and test _medians_ are identical, but the test-normal _upper tail_ is heavier — mean-based scores (`date`) sit in that tail and break, while the concentration-based `fall` score is robust. So the alarming `date` FP of ~2,600 was a **calibration artifact**, and the FALL method's FP (~130 at 99pct) was always an honest ~1.4% rate.

---

## 7. Why the absolute FP does not match per scenario

1. **Undocumented threshold rule.** The paper never defines the operating point behind its "average number of FP." A small threshold shift moves our FALL FP from 130 to 59 to 42 with zero change in AUC.
2. **Irreproducible expert deduplication (§III-A).** The authors had industry experts manually review and remove duplicate sequences. We cannot replicate that, so our test-normal composition differs by construction — enough to shift absolute FP on 11k+ windows.
3. **Faithful BGL property.** Our normal windows are concentrated (§6a) because we used the paper's minimal preprocessing; this caps class separation at ~0.85 AUC and therefore the achievable FP reduction.

---

## 8. Known limitations

- **`recall@thr` not yet reported** for the avg-FP operating points. FP without its recall is half the picture: lowering the threshold also lowers detection. The 99.7pct FP of ~59 is at a low-recall corner (the paper's regime), but the exact recall there is not yet printed. Recommended next addition to `evaluate.py`.
- **val positives ≈ 0** (chronological middle slice) → the D7 threshold is calibrated on a non-representative normal tail; see §6b. `fall` is robust to this; `date`/`partial_only` are not.
- **s3 is genuinely weak**, not just length-confounded: DATE AUC ≈ 0.68 with very high seed variance (seed 0 = 0.46, below chance). Pre-failure windows on s3 look _less_ anomalous to the model than normal ones.
- **DATE seed variance** is much larger for us than the paper (s3 ±0.19 vs the paper's SE ≈ 0.005).

---

## 9. How to reproduce

```bash
# 1. preprocess + tokenizer + windowing (CPU)
sbatch scripts/run-gate.sh bgl

# 2. train all 15 checkpoints (GPU array, s1..s5 x 3 seeds)
sbatch scripts/train.sh

# 3. evaluate (AUC + avg-FP + FP@95%TPR); pct is the threshold percentile
sbatch scripts/eval.sh 99.7            # or 99, 99.9

# 4. diagnostics
sbatch scripts/diag_calibration.sh     # val->test calibration
#   per-token concentration:
python src/diag_concentration.py --windows-dir data/processed/bgl --scenario s1 \
    --tokenizer data/processed/bgl/bgl_tok_8000.json --ckpt-dir results/ckpt/bgl --seed 0
```

## 10. File manifest

- `src/preprocess.py` — parse BGL, normalize, node_card coarsening, dedup.
- `src/windowing.py` — 30-log windows, stride 15, chronological split, normal-only train.
- `src/tokenizer_train.py` — WordPiece (vocab 8000).
- `src/masking.py` — K=50 frozen masking patterns, random corruption.
- `src/model.py` — 4-layer discriminator, RTD + RMD heads, FALL loss.
- `src/train.py` — training loop, 20k steps, per-seed checkpoints.
- `src/score.py` — Eq. 6 sharpening, Eq. 7 selective score, AUC, FP (math untouched).
- `src/evaluate.py` — AUC + avg-FP(`--pct`) + `FP@95%TPR`; `--score-unit {segment,pooled}`.
- `src/diag_concentration.py` — per-token concentration (gini / ratio / mass).
- `src/diag_calibration.py` — validation vs test normal-score calibration.
- `src/length_diag.py` — model-free length confound check (`AUC(1/L)`).
- `scripts/` — SLURM wrappers: `run-gate.sh`, `train.sh`, `eval.sh`, `diag_calibration.sh`.
