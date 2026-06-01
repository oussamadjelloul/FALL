#!/bin/bash
#SBATCH --job-name=fall_gate
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/logs/fall_gate_%j.out

# FALL reproduction - gate 3 (tokenizer) + gate 4 (windowing feasibility).
# CPU-only. Run as a batch job:   sbatch scripts/run-gate.sh [dataset]
# or inside an allocation:        salloc ... ; bash scripts/run-gate.sh [dataset]
# dataset defaults to bgl; pass "thunderbird" once its parser exists.

set -euo pipefail

cd ~/FALL
source ~/sdd_activate.sh

DATASET="${1:-bgl}"
case "${DATASET}" in
  bgl)         RAW=data/raw/bgl/BGL.log ;;
  thunderbird) RAW=data/raw/tbird/Thunderbird.log ;;
  *) echo "unknown dataset: ${DATASET}" >&2; exit 1 ;;
esac

PROC_DIR="data/processed/${DATASET}"
PRE="${PROC_DIR}/${DATASET}_pre.jsonl"
mkdir -p "${PROC_DIR}" results/logs

echo "==================== 1. preprocess (${DATASET}) ===================="
# D14: node_card granularity (R0X-MX-NX). Use full for the sensitivity run.
python src/preprocess.py --input "${RAW}" --dataset "${DATASET}" --out "${PRE}" \
  --node-level node_card

echo "==================== 2. gate 3 - tokenizer ===================="
for V in 8000 12000 16000; do
  echo "---- vocab_size=${V} ----"
  python src/tokenizer_train.py --input "${PRE}" --vocab-size "${V}" \
    --out "${PROC_DIR}/${DATASET}_tok_${V}.json"
done

echo "==================== 3. gate 4 - windowing ===================="
# windowing.py appends /<dataset>/<scenario>/ under --out, so windows land in
# data/processed/<dataset>/<scenario>/{train,val,test}.jsonl
python src/windowing.py --input "${PRE}" --dataset "${DATASET}" --out data/processed

echo "==================== done (${DATASET}) ===================="
