#!/bin/bash
#SBATCH --job-name=halluc_eval
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=hallucination_eval/logs/%j.out
#SBATCH --error=hallucination_eval/logs/%j.err

set -e

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate sdpo

# Set OPENAI_API_KEY in your environment before running this script
# export OPENAI_API_KEY="your-key-here"

EVAL_DIR="/fsx/ubuntu/repos/SDPO_custom/hallucination_eval"

# --- Configuration ---
# Which experiment to evaluate (think or nothink)
TRAJ_DIR="/fsx/ubuntu/repos/SDPO_custom/ttrl_runs/codeio-codeio_mini_sdpo_rl-qwen3-1.7b-ml.p4de.24xlarge-v1_sdpo_rl_loggen/trajectories"
OUTPUT="${EVAL_DIR}/results_codeio_sdpo_rl.csv"
STRIDE=10       # evaluate every 10th step (469 files)
MAX_SAMPLES=64  # per step (0 = all 256)

echo "Starting hallucination evaluation..."
echo "Trajectory dir: ${TRAJ_DIR}"
echo "Output: ${OUTPUT}"

python ${EVAL_DIR}/evaluate.py \
    --traj_dir "${TRAJ_DIR}" \
    --step_stride ${STRIDE} \
    --max_samples ${MAX_SAMPLES} \
    --output "${OUTPUT}" \
    --concurrency 30 \
    --model gpt-4o-mini

echo ""
echo "=== Analysis ==="
python ${EVAL_DIR}/analyze.py "${OUTPUT}" --plot

echo "Done!"
