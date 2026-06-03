#!/bin/bash
#SBATCH --job-name=fall-calib
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=fall-calib-%j.out

# Validation-vs-test normal-score calibration check across all five BGL scenarios.
# Usage:
#   sbatch scripts/diag_calibration.sh        # seed=0 (default)
#   sbatch scripts/diag_calibration.sh 1      # override seed
set -eo pipefail

SEED="${1:-0}"

cd ~/FALL
source ~/sdd_activate.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TOK=data/processed/bgl/bgl_tok_8000.json

echo "=== FALL calibration | seed=${SEED} | $(date) ==="
for SC in s1 s2 s3 s4 s5; do
  python src/diag_calibration.py \
    --windows-dir data/processed/bgl \
    --scenario "$SC" \
    --tokenizer "$TOK" \
    --ckpt-dir results/ckpt/bgl \
    --seed "$SEED"
done
echo "=== done | $(date) ==="
