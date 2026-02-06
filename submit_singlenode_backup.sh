#!/bin/bash

# =============================================================================
# General SDPO Job Submission Script
#
# Usage:
#   ./submit.sh -p PARTITION -c CONFIG -s SUFFIX [-m MODEL] [-d DATA] [-r] [extra hydra args...]
#
# Examples:
#   ./submit.sh -p mi2508x -c general -s exp1
#   ./submit.sh -p mi2508x -c general -s exp1 -d datasets/sciknoweval/physics/
#   ./submit.sh -p mi3258x -c general_rl_coef -s exp2
#   ./submit.sh -p mi2508x -c general -s exp3 -r                          # auto-requeue on timeout
#   ./submit.sh -p mi2508x -c general -s exp3 -m allenai/OLMo-3-7B-Instruct
#   ./submit.sh -p mi2508x -c general -s exp3 actor_rollout_ref.actor.optim.lr=1e-6
#
# Flags:
#   -p  SLURM partition (e.g., mi2508x, mi3258x)
#   -c  Run script name from run_scripts/ (e.g., general, general_rl_coef)
#   -s  Experiment suffix (used in job name and passed to run script)
#   -m  Model path (default: Qwen/Qwen3-8B)
#   -d  Data path (default: datasets/tooluse)
#   -r  Auto-requeue: resubmit the job when approaching the time limit
#
# Partition-specific behavior:
#   mi2508x: Enables FSDP optimizer and param offloading automatically
#
# Run scripts live in run_scripts/ and should NOT include offloading args.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: ./submit.sh -p PARTITION -c CONFIG -s SUFFIX [-m MODEL] [-d DATA] [-r] [extra hydra args...]"
    echo ""
    echo "Required flags:"
    echo "  -p  SLURM partition (e.g., mi2508x, mi3258x)"
    echo "  -c  Run script name from run_scripts/ (e.g., general, general_rl_coef)"
    echo "  -s  Experiment suffix"
    echo ""
    echo "Optional flags:"
    echo "  -m  Model path (default: Qwen/Qwen3-8B)"
    echo "  -d  Data path (default: datasets/tooluse)"
    echo "  -r  Auto-requeue on time limit (resubmits with same args)"
    echo ""
    echo "Available run scripts:"
    ls -1 "${SCRIPT_DIR}/run_scripts/"*.sh 2>/dev/null | xargs -n 1 basename | sed 's/\.sh$//' || echo "  None found"
    exit 1
}

# Parse flags
PARTITION="mi2508x"
CONFIG="general"
SUFFIX="$(date +%d%m_%H%M)"
MODEL="Qwen/Qwen3-8B"
DATA="datasets/tooluse"
REQUEUE=false

while getopts "p:c:s:m:d:r" opt; do
    case $opt in
        p) PARTITION="$OPTARG" ;;
        c) CONFIG="$OPTARG" ;;
        s) SUFFIX="$OPTARG" ;;
        m) MODEL="$OPTARG" ;;
        d) DATA="$OPTARG" ;;
        r) REQUEUE=true ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

# Remaining args are extra Hydra overrides
USER_EXTRA_ARGS="$*"

# Validate required flags
if [ -z "$PARTITION" ] || [ -z "$CONFIG" ] || [ -z "$SUFFIX" ]; then
    echo "Error: -p, -c, and -s are all required."
    echo ""
    usage
fi

# Validate run script exists
RUN_SCRIPT="run_scripts/${CONFIG}.sh"
if [ ! -f "${SCRIPT_DIR}/${RUN_SCRIPT}" ]; then
    echo "Error: Run script ${RUN_SCRIPT} does not exist."
    echo ""
    echo "Available run scripts:"
    ls -1 "${SCRIPT_DIR}/run_scripts/"*.sh 2>/dev/null | xargs -n 1 basename | sed 's/\.sh$//' || echo "  None found"
    exit 1
fi

# Build partition-specific Hydra args
PARTITION_ARGS=""
if [ "$PARTITION" = "mi2508x" ]; then
    PARTITION_ARGS="actor_rollout_ref.actor.fsdp_config.optimizer_offload=true actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.rollout.gpu_memory_utilization=0.45"
fi

# Combine all extra Hydra args: partition-specific + user-provided
EXTRA_HYDRA_ARGS=""
if [ -n "$PARTITION_ARGS" ]; then
    EXTRA_HYDRA_ARGS="$PARTITION_ARGS"
fi
if [ -n "$USER_EXTRA_ARGS" ]; then
    EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:+$EXTRA_HYDRA_ARGS }$USER_EXTRA_ARGS"
fi

# Short model name for job name (e.g. Qwen/Qwen3-8B -> qwen3-8b)
MODEL_SHORT=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')

# Short data name for job name (e.g. datasets/sciknoweval/physics/ -> physics)
DATA_SHORT=$(basename "${DATA%/}")

# Job name: data-config-model-partition-suffix
JOB_NAME="${DATA_SHORT}-${CONFIG}-${MODEL_SHORT}-${PARTITION}-${SUFFIX}"

# Create logs directory
mkdir -p "${SCRIPT_DIR}/logs"

echo "============================================"
echo "Submitting SDPO Training Job"
echo "============================================"
echo "Job name:   $JOB_NAME"
echo "Partition:  $PARTITION"
echo "Config:     $CONFIG"
echo "Suffix:     $SUFFIX"
echo "Model:      $MODEL"
echo "Data:       $DATA"
echo "Run script: $RUN_SCRIPT"
if [ -n "$PARTITION_ARGS" ]; then
    echo "Offloading: enabled (mi2508x)"
else
    echo "Offloading: disabled"
fi
if [ -n "$USER_EXTRA_ARGS" ]; then
    echo "Extra args: $USER_EXTRA_ARGS"
fi
if [ "$REQUEUE" = true ]; then
    echo "Requeue:    enabled (resubmits 120s before time limit)"
fi
echo "============================================"

# Build sbatch flags
SBATCH_FLAGS=(
    --job-name="$JOB_NAME"
    --partition="$PARTITION"
    --nodes=1
    --ntasks-per-node=1
    --exclusive
    --time=12:00:00
    --output="${SCRIPT_DIR}/logs/${JOB_NAME}_%j.log"
    --error="${SCRIPT_DIR}/logs/${JOB_NAME}_%j.err"
    --export="ALL,EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS},MODEL_PATH=${MODEL},DATA_PATH=${DATA}"
)

# If requeue enabled, tell SLURM to send SIGUSR1 120s before time limit
if [ "$REQUEUE" = true ]; then
    SBATCH_FLAGS+=(--signal=B:USR1@120)
fi

# Build the requeue command that the job will use to resubmit itself
REQUEUE_CMD="${SCRIPT_DIR}/submit.sh -p ${PARTITION} -c ${CONFIG} -s ${SUFFIX} -m ${MODEL} -d ${DATA} -r ${USER_EXTRA_ARGS}"

# Submit to SLURM
# EXTRA_HYDRA_ARGS is exported so the run script picks it up via its EXTRA_HYDRA_ARGS check
sbatch "${SBATCH_FLAGS[@]}" <<EOF
#!/bin/bash

echo "============================================"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "Partition: ${PARTITION}"
echo "Config: ${CONFIG}"
echo "Suffix: ${SUFFIX}"
echo "Model: ${MODEL}"
echo "Requeue: ${REQUEUE}"
echo "============================================"

cd ${SCRIPT_DIR}

if [ "${REQUEUE}" = true ]; then
    # Trap SIGUSR1 sent by SLURM before time limit
    requeue() {
        echo ""
        echo "============================================"
        echo "Time limit approaching — requeuing job..."
        echo "============================================"
        ${REQUEUE_CMD}
        exit 0
    }
    trap requeue USR1

    # Run in background so the trap can fire while we wait
    ./container_exec.sh ./${RUN_SCRIPT} ${SUFFIX} &
    wait \$!
else
    ./container_exec.sh ./${RUN_SCRIPT} ${SUFFIX}
fi

echo "============================================"
echo "Training complete!"
echo "============================================"
EOF
