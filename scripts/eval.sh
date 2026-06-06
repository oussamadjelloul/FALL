#!/bin/bash
#SBATCH --job-name=fall-eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=fall-eval-%j.out

# FALL evaluation across all five BGL scenarios (AUC + avg-FP + FP@95%TPR).
# Usage:
#   sbatch scripts/eval.sh             # pct=99.7, seeds=0,1,2 (defaults)
#   sbatch scripts/eval.sh 99          # override threshold percentile
#   sbatch scripts/eval.sh 99.7 0,1,2  # override pct and seeds
set -eo pipefail

PCT="${1:-99.7}"
SEEDS="${2:-0,1,2}"
DATASET="${DATASET:-bgl}"      # override: DATASET=thunderbird sbatch scripts/eval.sh
VOCAB="${VOCAB:-8000}"

cd ~/FALL
source ~/sdd_activate.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TOK="data/processed/${DATASET}/${DATASET}_tok_${VOCAB}.json"

echo "=== FALL eval | score-unit=segment | pct=${PCT} | seeds=${SEEDS} | $(date) ==="
for SC in s1 s2 s3 s4 s5; do
  python src/evaluate.py \
    --windows-dir "data/processed/${DATASET}" \
    --scenario "$SC" \
    --tokenizer "$TOK" \
    --ckpt-dir "results/ckpt/${DATASET}" \
    --seeds "$SEEDS" \
    --score-unit segment \
    --pct "$PCT"
done
echo "=== done | $(date) ==="
