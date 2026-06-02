#!/bin/bash
#SBATCH --job-name=fall_eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=results/logs/fall_eval_%j.out

# FALL evaluation: AUC-ROC + avg-FP per scoring mode, all 5 scenarios, 3 seeds.
# Produces the Table XI (FALL vs DATE) + Table VIII (ablation) BGL comparison.
#   sbatch scripts/eval.sh
# Forward-only, so it is fast; if --gres=gpu:a100:1 is rejected use gpu:1.

set -euo pipefail

cd ~/FALL
source ~/sdd_activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

DATASET=bgl
TOK="data/processed/${DATASET}/${DATASET}_tok_8000.json"
WIN="data/processed/${DATASET}"
CKPT="results/ckpt/${DATASET}"
mkdir -p results/logs results/eval

for SC in s1 s2 s3 s4 s5; do
  echo "######## evaluate ${SC} ########"
  python src/evaluate.py \
    --windows-dir "${WIN}" --scenario "${SC}" \
    --tokenizer "${TOK}" --ckpt-dir "${CKPT}" \
    --seeds 0,1,2 --modes fall,date,sharpen_only,partial_only
done

echo "######## eval done ########"
