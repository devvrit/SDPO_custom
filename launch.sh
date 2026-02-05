#!/bin/bash

# =============================================================================
# Smart SDPO Training Launcher
# Automatically handles single-node and multi-node configurations
#
# Usage:
#   ./launch.sh <script_name> [nnodes] [time_hours]
#
# Examples:
#   ./launch.sh tooluse           # Single node (8 GPUs), 24h
#   ./launch.sh tooluse 1         # Single node (8 GPUs), 24h
#   ./launch.sh tooluse 2         # Multi-node (16 GPUs), 24h
#   ./launch.sh tooluse 2 12      # Multi-node (16 GPUs), 12h
# =============================================================================

set -e

# Parse arguments
SCRIPT_NAME=${1:-""}
NNODES=${2:-1}
TIME_HOURS=${3:-12}

# Validate script name
if [ -z "$SCRIPT_NAME" ]; then
    echo "Error: No script name provided"
    echo ""
    echo "Usage: ./launch.sh <script_name> [nnodes] [time_hours]"
    echo ""
    echo "Available scripts:"
    ls -1 run/*.sh 2>/dev/null | xargs -n 1 basename | sed 's/\.sh$//' || echo "  None found"
    echo ""
    echo "Examples:"
    echo "  ./launch.sh tooluse       # 1 node, 12h"
    echo "  ./launch.sh tooluse 2     # 2 nodes, 12h"
    echo "  ./launch.sh tooluse 2 24  # 2 nodes, 24h"
    exit 1
fi

# Validate script exists
if [ ! -f "run/${SCRIPT_NAME}.sh" ]; then
    echo "Error: Script run/${SCRIPT_NAME}.sh does not exist"
    echo "Available scripts:"
    ls -1 run/*.sh 2>/dev/null | xargs -n 1 basename | sed 's/\.sh$//' || echo "  None found"
    exit 1
fi

# Validate nnodes
if ! [[ "$NNODES" =~ ^[0-9]+$ ]] || [ "$NNODES" -lt 1 ]; then
    echo "Error: nnodes must be a positive integer (got: $NNODES)"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "============================================"
echo "SDPO Training Launcher"
echo "============================================"
echo "Script:     run/${SCRIPT_NAME}.sh"
echo "Nodes:      $NNODES"
echo "GPUs:       $((NNODES * 8))"
echo "Time limit: ${TIME_HOURS}h"
echo "Mode:       $([ $NNODES -eq 1 ] && echo 'Single-node (simple)' || echo 'Multi-node (Ray cluster)')"
echo "============================================"

# Submit job with appropriate parameters
JOB_ID=$(sbatch \
    --job-name=sdpo \
    --partition=mi2508x \
    --nodes=$NNODES \
    --ntasks-per-node=1 \
    --exclusive \
    --time=${TIME_HOURS}:00:00 \
    --output=logs/sdpo_%j.log \
    --error=logs/sdpo_%j.err \
    --export=ALL,NNODES=$NNODES,SCRIPT_NAME=$SCRIPT_NAME \
    --parsable \
    submit_dynamic.sh)

echo ""
echo "Job submitted: $JOB_ID"
echo ""
echo "Monitor with:"
echo "  tail -f logs/sdpo_${JOB_ID}.log"
echo "  tail -f logs/sdpo_${JOB_ID}.err"
echo ""
