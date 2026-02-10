#!/bin/bash

# =============================================================================
# General SDPO Job Submission Script
#
# Usage:
#   ./submit.sh -p PARTITION -c CONFIG -s SUFFIX [-n NNODES] [-m MODEL] [-d DATA] [-e EXP_NAME] [-r] [extra hydra args...]
#
# Examples:
#   ./submit.sh -p mi2508x -c general -s exp1
#   ./submit.sh -p mi2508x -c general -s exp1 -d datasets/sciknoweval/physics/
#   ./submit.sh -p mi3258x -c general_rl_coef -s exp2
#   ./submit.sh -p mi2508x -c general -s exp3 -r                          # auto-requeue on timeout
#   ./submit.sh -p mi2508x -c general -s exp3 -m allenai/OLMo-3-7B-Instruct
#   ./submit.sh -p mi2508x -c general -s exp3 actor_rollout_ref.actor.optim.lr=1e-6
#   ./submit.sh -p mi2508x -c general -s exp3 -n 2                        # multi-node (2 nodes)
#   ./submit.sh -p mi2508x -c general -s exp3 -e "OLD-EXP-NAME"           # resume old run
#   ./submit.sh -p mi3008x -c general -s exp1 ray_kwargs.ray_init.num_cpus=64
#   ./submit.sh -p mi2508x -c general -s exp3 trainer.log_trace_interval=5 actor_rollout_ref.rollout.interruption.enable=true
#
# Flags:
#   -p  SLURM partition (e.g., mi2508x, mi3258x)
#   -c  Run script name from run_scripts/ (e.g., general, general_rl_coef)
#   -s  Experiment suffix (used in job name and passed to run script)
#   -n  Number of nodes (default: 1; multi-node uses Ray cluster via _multinode_worker.sh)
#   -m  Model path (default: Qwen/Qwen3-8B)
#   -d  Data path (default: datasets/tooluse)
#   -e  Experiment name override (for resuming old runs with different naming)
#   -r  Auto-requeue: resubmit the job when approaching the time limit
#
# Partition-specific behavior:
#   mi2508x: Enables FSDP optimizer and param offloading automatically
#
# Data-specific behavior:
#   polaris*: Sets model_dtype to bfloat16 (actor + ref)
#
# Run scripts live in run_scripts/ and should NOT include offloading args.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: ./submit.sh -p PARTITION -c CONFIG -s SUFFIX [-n NNODES] [-m MODEL] [-d DATA] [-e EXP_NAME] [-r] [extra hydra args...]"
    echo ""
    echo "Required flags:"
    echo "  -p  SLURM partition (e.g., mi2508x, mi3258x)"
    echo "  -c  Run script name from run_scripts/ (e.g., general, general_rl_coef)"
    echo "  -s  Experiment suffix"
    echo ""
    echo "Optional flags:"
    echo "  -n  Number of nodes (default: 1; >= 2 enables multi-node Ray cluster)"
    echo "  -m  Model path (default: Qwen/Qwen3-8B)"
    echo "  -d  Data path (default: datasets/tooluse)"
    echo "  -e  Experiment name override (for resuming old runs with different naming)"
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
NNODES=1
MODEL="Qwen/Qwen3-8B"
DATA="datasets/tooluse"
EXP_NAME_OVERRIDE=""
REQUEUE=false

while getopts "p:c:s:n:m:d:e:r" opt; do
    case $opt in
        p) PARTITION="$OPTARG" ;;
        c) CONFIG="$OPTARG" ;;
        s) SUFFIX="$OPTARG" ;;
        n) NNODES="$OPTARG" ;;
        m) MODEL="$OPTARG" ;;
        d) DATA="$OPTARG" ;;
        e) EXP_NAME_OVERRIDE="$OPTARG" ;;
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

# Validate NNODES
if ! [[ "$NNODES" =~ ^[0-9]+$ ]] || [ "$NNODES" -lt 1 ]; then
    echo "Error: -n must be a positive integer (got: $NNODES)"
    exit 1
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

# Validate worker script exists for multi-node
if [ "$NNODES" -ge 2 ] && [ ! -f "${SCRIPT_DIR}/_multinode_worker.sh" ]; then
    echo "Error: _multinode_worker.sh not found in ${SCRIPT_DIR} (required for multi-node)"
    exit 1
fi

# Build partition-specific Hydra args
PARTITION_ARGS=""
if [ "$PARTITION" = "mi3008x" ]; then
    PARTITION_ARGS="ray_kwargs.ray_init.num_cpus=64"
fi

# Build data-specific Hydra args
DATA_ARGS=""
if [[ "$DATA" == *polaris* ]]; then
    DATA_ARGS="actor_rollout_ref.rollout.gpu_memory_utilization=0.3"
elif [ "$CONFIG" = "cotrain" ]; then
    DATA_ARGS="actor_rollout_ref.rollout.gpu_memory_utilization=0.4"
fi

# Build multi-node Hydra args
MULTINODE_ARGS=""
if [ "$NNODES" -ge 2 ]; then
    MULTINODE_ARGS="trainer.nnodes=$NNODES +ray_kwargs.ray_init.address=auto actor_rollout_ref.actor.fsdp_config.fsdp_size=8"
fi

# Combine all extra Hydra args: partition + data + multinode + user
EXTRA_HYDRA_ARGS=""
if [ -n "$PARTITION_ARGS" ]; then
    EXTRA_HYDRA_ARGS="$PARTITION_ARGS"
fi
if [ -n "$DATA_ARGS" ]; then
    EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:+$EXTRA_HYDRA_ARGS }$DATA_ARGS"
fi
if [ -n "$MULTINODE_ARGS" ]; then
    EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:+$EXTRA_HYDRA_ARGS }$MULTINODE_ARGS"
fi
if [ -n "$USER_EXTRA_ARGS" ]; then
    EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:+$EXTRA_HYDRA_ARGS }$USER_EXTRA_ARGS"
fi

# Short model name for job name (e.g. Qwen/Qwen3-8B -> qwen3-8b)
MODEL_SHORT=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')

# Short data name for job name (e.g. datasets/sciknoweval/physics/ -> physics)
DATA_SHORT=$(basename "${DATA%/}")

# Job name: data-config-model-partition[-Nn]-suffix
if [ "$NNODES" -ge 2 ]; then
    JOB_NAME="${DATA_SHORT}-${CONFIG}-${MODEL_SHORT}-${PARTITION}-${NNODES}n-${SUFFIX}"
else
    JOB_NAME="${DATA_SHORT}-${CONFIG}-${MODEL_SHORT}-${PARTITION}-${SUFFIX}"
fi

# Create logs directory
mkdir -p "${SCRIPT_DIR}/logs"

echo "============================================"
if [ "$NNODES" -ge 2 ]; then
    echo "Submitting Multi-Node SDPO Training Job"
else
    echo "Submitting SDPO Training Job"
fi
echo "============================================"
echo "Job name:   $JOB_NAME"
echo "Partition:  $PARTITION"
if [ "$NNODES" -ge 2 ]; then
    echo "Nodes:      $NNODES"
    echo "GPUs:       $((NNODES * 8))"
fi
echo "Config:     $CONFIG"
echo "Suffix:     $SUFFIX"
echo "Model:      $MODEL"
echo "Data:       $DATA"
echo "Run script: $RUN_SCRIPT"
if [ -n "$DATA_ARGS" ]; then
    echo "Polaris opts: model_dtype=bfloat16"
fi
if [ -n "$PARTITION_ARGS" ]; then
    echo "Offloading: enabled (mi2508x)"
else
    echo "Offloading: disabled"
fi
if [ -n "$EXP_NAME_OVERRIDE" ]; then
    echo "Exp name:   $EXP_NAME_OVERRIDE (override)"
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
    --nodes="$NNODES"
    --ntasks-per-node=1
    --exclusive
    --time=12:00:00
    --output="${SCRIPT_DIR}/logs/${JOB_NAME}_%j.log"
    --error="${SCRIPT_DIR}/logs/${JOB_NAME}_%j.err"
)

# If requeue enabled, tell SLURM to send SIGUSR1 120s before time limit
if [ "$REQUEUE" = true ]; then
    SBATCH_FLAGS+=(--signal=B:USR1@120)
fi

# Build the requeue command that the job will use to resubmit itself
REQUEUE_CMD="${SCRIPT_DIR}/submit.sh -p ${PARTITION} -c ${CONFIG} -s ${SUFFIX} -n ${NNODES} -m ${MODEL} -d ${DATA}"
if [ -n "$EXP_NAME_OVERRIDE" ]; then
    REQUEUE_CMD="$REQUEUE_CMD -e '${EXP_NAME_OVERRIDE}'"
fi
if [ "$REQUEUE" = true ]; then
    REQUEUE_CMD="$REQUEUE_CMD -r"
fi
if [ -n "$USER_EXTRA_ARGS" ]; then
    REQUEUE_CMD="$REQUEUE_CMD $USER_EXTRA_ARGS"
fi

# Submit to SLURM
if [ "$NNODES" -ge 2 ]; then
    # Multi-node: use _multinode_worker.sh (handles Ray cluster setup)
    sbatch "${SBATCH_FLAGS[@]}" \
        --export="ALL,SCRIPT_DIR=${SCRIPT_DIR},RUN_SCRIPT=${RUN_SCRIPT},SUFFIX=${SUFFIX},EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS},MODEL_PATH=${MODEL},DATA_PATH=${DATA},JOB_NAME=${JOB_NAME},EXP_NAME_OVERRIDE=${EXP_NAME_OVERRIDE},REQUEUE=${REQUEUE},REQUEUE_CMD=${REQUEUE_CMD}" \
        "${SCRIPT_DIR}/_multinode_worker.sh"
else
    # Single-node: inline sbatch script
    sbatch "${SBATCH_FLAGS[@]}" \
        --export="ALL,EXTRA_HYDRA_ARGS=${EXTRA_HYDRA_ARGS},MODEL_PATH=${MODEL},DATA_PATH=${DATA},JOB_NAME=${JOB_NAME},EXP_NAME_OVERRIDE=${EXP_NAME_OVERRIDE}" \
        <<EOF
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
fi
