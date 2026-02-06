#!/bin/bash
#SBATCH --job-name=sdpo
#SBATCH --partition=mi3258x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/work1/agrawal/devvrit/SDPO_custom/logs/sdpo_%j.log
#SBATCH --error=/work1/agrawal/devvrit/SDPO_custom/logs/sdpo_%j.err

# =============================================================================
# Single-Node SDPO Training Script for AMD MI250 GPUs
# Usage: sbatch submit_single.sh <script_name> [extra args...]
# Example: sbatch submit_single.sh math 0.5
# =============================================================================

set -e

# Check if script name was provided
if [ -z "$1" ]; then
    echo "Error: No script name provided"
    echo "Usage: sbatch submit_single.sh <script_name>"
    echo "Available scripts: tooluse, lcb, physics"
    exit 1
fi

SCRIPT_NAME=$1
shift

# Validate script exists
if [ ! -f "$WORK/SDPO_custom/run/${SCRIPT_NAME}.sh" ]; then
    echo "Error: Script run/${SCRIPT_NAME}.sh does not exist"
    echo "Available scripts:"
    ls -1 $WORK/SDPO_custom/run/*.sh 2>/dev/null | xargs -n 1 basename | sed 's/\.sh$//' || echo "None found"
    exit 1
fi

# Create logs directory
mkdir -p $WORK/SDPO_custom/logs

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Single-node training"
echo "Script: run/${SCRIPT_NAME}.sh"
echo "============================================"

# Change to the working directory
cd $WORK/SDPO_custom

# =============================================================================
# RUN TRAINING (Ray will auto-initialize on single node)
# =============================================================================

# Run the script via container
./container_exec.sh ./run/${SCRIPT_NAME}.sh "$@"

echo "============================================"
echo "Training complete!"
echo "============================================"
