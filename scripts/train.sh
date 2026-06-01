#!/bin/bash
#SBATCH --job-name=fall_train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-4
#SBATCH --output=results/logs/fall_train_%A_%a.out

# FALL phase-2 training on GPU.
#   Full sweep (5 scenarios x 3 seeds), one array task per scenario:
#       sbatch scripts/train.sh
#   Smoke run (s1, seed 0, short steps) on a single GPU:
#       sbatch --array=0 scripts/train.sh smoke 2000
# Arg1: mode {full|smoke} (default full).  Arg2: smoke step count (default 2000).
# If --gres=gpu:a100:1 is rejected on your partition, change it to gpu:1.

set -euo pipefail

cd ~/FALL
source ~/sdd_activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

DATASET=bgl
TOK="data/processed/${DATASET}/${DATASET}_tok_8000.json"   # D2 = 8000
WIN="data/processed/${DATASET}"
OUT="results/ckpt/${DATASET}"
mkdir -p "${OUT}" results/logs

SCENARIOS=(s1 s2 s3 s4 s5)
MODE="${1:-full}"

if [[ "${MODE}" == "smoke" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID:-0}" != "0" ]]; then
    echo "smoke: skipping array idx ${SLURM_ARRAY_TASK_ID}"; exit 0
  fi
  SC="s1"; SEEDS="0"; STEPS="${2:-2000}"
else
  IDX="${SLURM_ARRAY_TASK_ID:-0}"
  SC="${SCENARIOS[$IDX]}"; SEEDS="0,1,2"; STEPS="20000"
fi

echo "==== train ${DATASET} ${SC} seeds=${SEEDS} steps=${STEPS} (mode=${MODE}) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/train.py \
  --windows-dir "${WIN}" \
  --scenario "${SC}" \
  --tokenizer "${TOK}" \
  --out "${OUT}" \
  --seeds "${SEEDS}" \
  --steps "${STEPS}"

echo "==== done ${SC} ===="
