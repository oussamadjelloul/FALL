# FALL Reproduction — Sequence vs Label Sharpening on BGL & Thunderbird

**Version:** v1.0
**Date:** 2026-06-08
**Scope:** Documents the two readings of FALL's Eq. 6 sharpening ("sequence" vs
"label") evaluated on BGL (full) and Thunderbird (first-10M subset), using the
same trained discriminators (no retraining; scoring-only changes). Supersedes
§4 of `fall-reproduction-notes-v1.md` (the length-confound / segment-scoring
framing).

## Changelog
- **v1.0** — Initial. Adds `fall_label` / `sharpen_label_only` (Eq. 6 read over
  the L=2 RTD labels) alongside the existing `fall` / `sharpen_only` (Eq. 6 read
  over the L sequence tokens). Full AUC + avg-FP tables for both datasets, the
  `sharpen_only ≡ 1/L` model-independence evidence, the Thunderbird length
  diagnostic, and the revised interpretation.

---

## 1. Headline finding

FALL's published anomaly score is **dominated by a window-length confound**. Its
two reported advantages over the DATE baseline — higher AUC and dramatically
lower FP — reproduce **only** under the literal sequence reading of Eq. 6, and
**only** when failure windows are shorter than normal windows (as in BGL, and
evidently the authors' Thunderbird). On the standard first-10M Thunderbird
subset, where failure windows are ~4.7× longer, the same score **inverts below
random** (AUC ≈ 0.30). Once length is controlled — the label reading, which is
the technically correct application of sharpening — FALL's sharpening and
selective-token machinery **add nothing over plain DATE** (≈ DATE on BGL, below
DATE on Thunderbird).

This is a method/benchmark robustness result, not an implementation defect: the
reproduction's DATE and partial_only land in the paper's AUC band, the inversion
follows necessarily from two measured facts (failures 4.7× longer; score ∝ 1/L),
and both readings are faithful to a defensible interpretation of an equation the
paper specifies ambiguously.

---

## 2. The Eq. 6 ambiguity (the two readings)

Eq. 6: `sharpen(S,T)_i = t_i^(1/T) / Σ_{j=1}^{L} t_j^(1/T)`, with T = 1/2.

The paper's prose states "L … represent the number of labels," but the symbol
`t_j` was defined three pages earlier as the j-th **token** of the sequence
`S = (t_1,…,t_h)`. The two readings:

- **Sequence reading** (`sharpen`, `fall`, `sharpen_only`): sum over the L tokens
  of the window. The sharpened vector sums to 1, so the score carries a 1/L
  factor → length-dependent. This is the literal-symbol reading.
- **Label reading** (`sharpen_label`, `fall_label`, `sharpen_label_only`): per
  token, sum over the L=2 RTD classes {replaced, original}, i.e.
  `p_i^2 / (p_i^2 + (1-p_i)^2)`. No coupling across tokens → length-invariant.
  This matches the prose ("number of labels") and the MixMatch provenance of
  sharpening (a transform of a probability distribution).

All other pipeline elements (top-(1/n) selection with n=4, segment scoring with
max_len=128, etc.) are identical across both readings.

---

## 3. Experimental setup

| item | value |
|---|---|
| BGL | full LogHub BGL (4.7M lines), node_card granularity |
| Thunderbird | first 10,000,000 lines (verified: 353,794 alert lines), node=full |
| windowing | 30 logs/window, stride 15, 5 scenarios (lead/end per paper Tables II/III) |
| model | DATE/ELECTRA-style; gen 1-layer, disc 4-layer; max_len 128; K=50 |
| training | normal-only; AdamW; batch 128; 20,000 steps; seeds {0,1,2} |
| sharpening | T = 1/2; selection 1/n = 1/4 |
| score unit | segment (each ≤128-token chunk scored, averaged per window) |
| threshold | 99.7th percentile of validation-normal scores |

Modes evaluated (all from the SAME checkpoints): `fall` (sequence),
`sharpen_only` (sequence, = 1/L), `fall_label` (label), `sharpen_label_only`
(label), `partial_only` (top-¼ of raw probs, no sharpening), `date` (mean of all
raw probs).

---

## 4. Results — AUC-ROC (mean ± std over 3 seeds)

### 4.1 Thunderbird (first-10M)

| scen | fall (seq) | sharpen_only (=1/L) | fall_label | sharpen_label_only | partial_only | date |
|---|---|---|---|---|---|---|
| s1 | 0.308 ± 0.012 | 0.319 ± 0.001 | 0.847 ± 0.022 | 0.850 ± 0.021 | 0.895 ± 0.014 | 0.916 ± 0.011 |
| s2 | 0.298 ± 0.038 | 0.322 ± 0.001 | 0.935 ± 0.015 | 0.935 ± 0.017 | 0.945 ± 0.019 | 0.940 ± 0.032 |
| s3 | 0.312 ± 0.010 | 0.319 ± 0.001 | 0.888 ± 0.015 | 0.890 ± 0.015 | 0.914 ± 0.011 | 0.927 ± 0.021 |
| s4 | 0.316 ± 0.010 | 0.319 ± 0.000 | 0.874 ± 0.032 | 0.876 ± 0.033 | 0.906 ± 0.034 | 0.917 ± 0.048 |
| s5 | 0.287 ± 0.043 | 0.319 ± 0.001 | 0.860 ± 0.025 | 0.865 ± 0.025 | 0.894 ± 0.029 | 0.915 ± 0.039 |
| **mean** | **0.304** | **0.320** | **0.881** | **0.883** | **0.911** | **0.923** |

### 4.2 BGL (full)

| scen | fall (seq) | sharpen_only (=1/L) | fall_label | sharpen_label_only | partial_only | date |
|---|---|---|---|---|---|---|
| s1 | 0.851 ± 0.004 | 0.843 ± 0.000 | 0.812 ± 0.014 | 0.813 ± 0.015 | 0.818 ± 0.017 | 0.820 ± 0.022 |
| s2 | 0.852 ± 0.002 | 0.852 ± 0.000 | 0.856 ± 0.018 | 0.858 ± 0.018 | 0.861 ± 0.017 | 0.865 ± 0.020 |
| s3 | 0.816 ± 0.003 | 0.809 ± 0.001 | 0.715 ± 0.134 | 0.713 ± 0.136 | 0.711 ± 0.144 | 0.681 ± 0.192 |
| s4 | 0.849 ± 0.002 | 0.851 ± 0.000 | 0.803 ± 0.015 | 0.803 ± 0.016 | 0.812 ± 0.016 | 0.815 ± 0.018 |
| s5 | 0.787 ± 0.001 | 0.792 ± 0.001 | 0.731 ± 0.014 | 0.731 ± 0.014 | 0.740 ± 0.013 | 0.748 ± 0.011 |
| **mean** | **0.831** | **0.829** | **0.783** | **0.784** | **0.788** | **0.786** |

---

## 5. Results — avg-FP at 99.7th percentile (mean over 3 seeds)

### 5.1 Thunderbird

| scen | fall (seq) | sharpen_only | fall_label | sharpen_label_only | partial_only | date |
|---|---|---|---|---|---|---|
| s1 | 395 | 247 | 142 | 106 | 126 | 54 |
| s2 | 387 | 187 | 171 | 92 | 178 | 76 |
| s3 | 391 | 189 | 289 | 217 | 272 | 125 |
| s4 | 375 | 193 | 258 | 214 | 244 | 144 |
| s5 | 395 | 193 | 313 | 255 | 276 | 126 |

(At AUC ≈ 0.30 the `fall` FP figures are uninterpretable — an inverted ranking
makes FP at any threshold meaningless.)

### 5.2 BGL

| scen | fall (seq) | sharpen_only | fall_label | sharpen_label_only | partial_only | date |
|---|---|---|---|---|---|---|
| s1 | 59 | 32 | 1578 | 1600 | 1825 | 1839 |
| s2 | 61 | 32 | 1485 | 1493 | 1735 | 1823 |
| s3 | 58 | 32 | 1162 | 1138 | 1201 | 1138 |
| s4 | 56 | 31 | 1194 | 1180 | 1601 | 1665 |
| s5 | 59 | 34 | 1155 | 1155 | 1576 | 1673 |
| **mean** | **59** | **32** | **1315** | **1313** | **1388** | **1628** |

The BGL `fall` mean of **~59 FP** matches the paper's Table XI signature
(~58 FP). But `sharpen_only` (= 1/L) gives the **lowest** FP (~32), and the
length-invariant `fall_label` gives ~1315 ≈ `date` ~1628. So FALL's "low FP" on
BGL is a property of the low-variance 1/L score bunching the normals at the
99.7th percentile, **not** better discrimination. The FP@95%TPR (matched-recall)
figures confirm this: at equal recall, BGL `fall` ≈ 4044 FP vs `date` ≈ 5343 vs
`fall_label` ≈ 4475 — comparable, no real FP advantage.

---

## 6. Diagnosis — the length confound

**`sharpen_only ≡ 1/L`, model-independent.** Its seed std is ±0.000–0.001 on
both datasets (a learned signal would vary across seeds), confirming it is a pure
function of token count. Its AUC is **0.83 on BGL** (failures shorter → 1/L
points the right way) and **0.32 on Thunderbird** (failures longer → 1/L
inverts). `fall` (sequence) tracks it on both datasets (BGL 0.831 ≈ 0.829;
TB 0.304 ≈ 0.320), so `fall` is reading window length, not anomaly.

**Thunderbird length diagnostic** (`length_diag.py`, s1, N=500, seeded):

| | median tokens |
|---|---|
| normal windows | 284 |
| failure windows | 1330 (≈4.7×) |

`AUC(1/L) = 0.0051` (pooled). Equivalently `AUC(length) ≈ 0.995` — raw window
token-length separates failure from normal almost perfectly on this subset, and
the learned RTD signal (`date` 0.92) is **below** what pure length achieves. The
Thunderbird subset is largely surface-separable by verbosity.

**Sign flip.** BGL pre-failure windows are shorter than normal; Thunderbird's are
longer. The sequence reading's 1/L factor therefore helps on BGL and inverts on
Thunderbird — the same equation, opposite conclusion, decided entirely by the
dataset's length–label correlation.

**Score-unit interaction.** The 1/L leak depends on the score unit only for the
sequence reading: segment caps the effective L at ≤128 (TB `fall` 0.31), pooled
uses the full window length (TB `fall` → ~AUC(1/L) ≈ 0.005). The label reading is
unit-invariant (no length term). BGL windows are short enough that segment ≈
pooled there.

---

## 7. Interpretation

1. **Sequence reading = length detector.** Not anomaly detection; its quality is
   contingent on the dataset's length–label correlation.
2. **Label reading = length-invariant, tracks the learned signal.** `fall_label`
   ≈ `sharpen_label_only` ≈ `partial_only` on both datasets.
3. **Under the label (length-clean) reading, FALL ≤ DATE.** TB: `fall_label`
   0.881 < `date` 0.923. BGL: `fall_label` 0.783 ≈ `date` 0.786. Decomposition
   (TB s1): `date` 0.916 → squash-all (`sharpen_label_only`) 0.850 → top-¼
   squashed (`fall_label`) 0.847 — the **sharpening squash** is the dominant
   cost (−0.066), selection adds little after squashing. Both are fixed post-hoc
   transforms of the RTD probabilities and can only redistribute, not add,
   discriminative information.
4. **The paper's headline results are length-confounded.** FALL > DATE on BGL and
   the ~58 FP signature reproduce only under the sequence reading; both vanish
   under the label reading.

This is consistent with the paper's own ablation (Table VIII): even on the data
FALL was designed for, sharpening added only +0.0008 AUC and selective tokens
+0.004. On a benchmark that violates those design assumptions (limited
vocabulary, anomaly concentrated in a few alert tokens), the small positives turn
into small negatives.

---

## 8. Which reading did the authors use?

**Most likely the sequence reading.** Only the sequence reading reproduces the
authors' BGL signature (FALL > DATE on s1/s3/s4/s5; ~58 avg-FP, matching
Table XI). The label reading would have produced FALL ≈ DATE (no improvement),
which contradicts their reported gains. Their non-inverted Thunderbird (Table X:
FALL > DATE except s2) then implies their Thunderbird failure windows were **not**
longer than normal — a different subset and/or their aggressive preprocessing
("repetitive and sequential logs … leaving one instance," plus expert review of
duplicate sequences, §III-A) collapsing the verbose failure bursts that inflate
length on the standard first-10M subset.

**Caveat:** without the authors' code this is the best-supported inference from
their BGL signature, not proof. The paper publishes neither the public-dataset
subset definition nor per-class token-length statistics.

---

## 9. Status of prior §4 (superseded)

§4 of `fall-reproduction-notes-v1.md` framed the 1/L as "intrinsic to the formula,
mitigated by segment scoring." That analysis is correct *given the sequence
reading*, but is superseded by the deeper finding here:

- The 1/L is intrinsic only to the **sequence reading**; the label reading has no
  length term, so segment scoring is unnecessary there.
- Segment scoring mitigates but cannot rescue the sequence reading on Thunderbird
  (0.005 → 0.31, still inverted).
- The substantive conclusion is no longer "length confound, partially fixable" but
  "the score is length-dominated, and removing the confound erases the claimed
  improvement over DATE."

---

## 10. Caveats & open items

- Paper gains were marginal to begin with (+0.0008 / +0.004 AUC in Table VIII);
  this exposes a fragile claim, not a gross error.
- BGL s3 is a high-variance scenario (±0.13–0.19 across modes; only 137 test
  positives) — treat its point estimates loosely.
- Author-implementation inference (§8) is unverified against their code.
- Optional confirmations: (a) full pooled-unit sweep on both datasets to lock the
  2×2×2; (b) re-run with their aggressive "repetitive/sequential" dedup applied
  to test whether `medLen_fail` collapses toward `medLen_norm` and `fall`
  (sequence) recovers — pinning the dedup as the authors' escape from inversion.
