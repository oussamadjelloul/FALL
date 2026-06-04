# FALL — BGL Reproduction

Independent re-implementation of **FALL** (Jeong et al., _FALL: Prior Failure Detection in Large-Scale System Based on Language Model_, IEEE Transactions on Dependable and Secure Computing, vol. 22, no. 1, 2025), evaluated on the public **BGL** dataset. Built as a baseline for ongoing log-anomaly-detection research.

FALL is a self-supervised log **failure-prediction** model in the ELECTRA/DATE family: a random generator corrupts tokens of normal log windows, and a transformer discriminator learns replaced-token detection (RTD) and replaced-mask detection (RMD). At inference, the per-token RTD probabilities are turned into one anomaly score per window via temperature **sharpening** (Eq. 6) and **selective** top-token scoring (Eq. 7).

## Reproduction status

| Aspect                                                                                                       | Status                                                                                  |
| ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| Method — architecture, random-generator discriminator, sharpening + selective scoring, minimal preprocessing | Reproduced faithfully                                                                   |
| FALL AUC in the paper's 0.77–0.88 band + the FALL ≈ DATE pattern (incl. FALL losing s2)                      | Reproduced                                                                              |
| FALL ≪ DATE on false positives; selective + sharpening reduce FP                                             | Reproduced in direction                                                                 |
| FALL absolute avg-FP                                                                                         | Matched at a documented operating point (99.7th-pct threshold → avg ≈ 59 vs paper ≈ 58) |
| Private Samsung main results                                                                                 | Not reproducible (private 33M-log corpus)                                               |
| Aggressive IP/path normalization                                                                             | Out of scope — it is the paper's stated future work (§V)                                |

Full methodology, decisions, and diagnostics: [`docs/fall-reproduction-notes-v1.md`](docs/fall-reproduction-notes-v1.md).

## Key results (BGL, segment scoring, 3 seeds)

AUC-ROC vs paper Table XI:

| scenario      | our FALL | paper FALL | our DATE | paper DATE |
| ------------- | -------- | ---------- | -------- | ---------- |
| s1 (10–30s)   | 0.851    | 0.882      | 0.820    | 0.864      |
| s2 (30–60s)   | 0.852    | 0.767      | 0.865    | 0.791      |
| s3 (60–180s)  | 0.816    | 0.791      | 0.681    | 0.785      |
| s4 (180–300s) | 0.849    | 0.812      | 0.815    | 0.812      |
| s5 (300–420s) | 0.787    | 0.811      | 0.748    | 0.810      |

FALL avg-FP at the 99.7th-percentile threshold averages **59** vs the paper's **58**.

## Repository layout

```
src/        implementation: preprocess, windowing, tokenizer, masking,
            model, train, score, evaluate, diagnostics
scripts/    SLURM wrappers: run-gate, train, eval, diag_calibration
docs/       reproduction notes + spec
data/       processed windows + tokenizer (generated)
results/    checkpoints + evaluation output (generated)
```

## Quickstart (SLURM / Compute Canada)

```bash
sbatch scripts/run-gate.sh bgl     # 1. preprocess + tokenizer + windowing (CPU)
sbatch scripts/train.sh            # 2. train 15 checkpoints (s1..s5 x 3 seeds, GPU)
sbatch scripts/eval.sh 99.7        # 3. evaluate: AUC + avg-FP + FP@95%TPR
sbatch scripts/diag_calibration.sh # 4. validation->test calibration diagnostic
```

`evaluate.py` flags:

- `--score-unit {segment,pooled}` — `segment` (default) is the length-confound-corrected scoring; `pooled` reproduces the original length-leaking behaviour.
- `--pct` — decision-threshold percentile for avg-FP (e.g. 99, 99.7, 99.9).
- `--tpr` — matched-recall operating point for the `FP@<tpr>%TPR` column (default 0.95).

## Notable findings

- **Length confound.** The pooled FALL score is proportional to `1/L` (window length). Because BGL pre-failure windows are ~3.5× shorter than normal windows, naive scoring inflates AUC by tracking length rather than content. Fixed by scoring per ≤128-token segment (`--score-unit segment`), matching the paper's "split and process sequentially."
- **Concentration.** The per-token RTD signal is sharply concentrated in _both_ classes, so selective scoring adds ~0 to AUC (matching the paper's +0.004); the discriminating signal lives in the overall probability level.
- **Calibration.** The `date` baseline's large false-positive count is a validation→test calibration artifact (heavier test-normal tail); the `fall` score is well calibrated (~1% FPR).

## Caveats

- Reproduces the **public** BGL results only; the private-corpus headline numbers cannot be reproduced.
- Absolute FP magnitude depends on an undocumented threshold rule and an irreproducible expert-deduplication step in the original study (§III-A).
- `recall@thr` is not yet reported alongside avg-FP (planned addition).

## Citation

Original method:

> S. Jeong et al., "FALL: Prior Failure Detection in Large-Scale System Based on Language Model," _IEEE Transactions on Dependable and Secure Computing_, vol. 22, no. 1, 2025.

This repository is an independent reproduction for research/baseline purposes and is not affiliated with the original authors.
