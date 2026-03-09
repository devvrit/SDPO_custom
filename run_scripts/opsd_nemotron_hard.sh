#!/bin/bash

# Usage: ./run_scripts/opsd_nemotron_hard.sh [experiment_name_suffix]
#
# OPSD Nemotron Hard Verified experiment:
#   - Dataset: nemotron_hard_verified (45K hard math, summary solutions as reference)
#   - max_response_length=16384 (matching eval generation length)
#   - enable_thinking=true for student rollout
#   - max_reprompt_len=8192 (teacher sees more of reference solution)
#   - max_model_len=26624 (= 8192 reprompt + 16384 response + 2048 buffer)
#   - GPU_MEMORY_UTILIZATION=0.80 (safe with topk distillation)
#   - Same hyperparams as opsd_v1_longctx

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="opsd"

DATA_PATH="${DATA_PATH:-datasets/nemotron_hard_verified}"

# Hyperparameters (from paper Table 6)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=1       # Paper: 1 rollout per prompt
LR=2e-5                    # Paper: 2e-5
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export N_GPUS_PER_NODE=8

# Long-context specific settings
MAX_RESPONSE_LENGTH=16384
MAX_MODEL_LEN=26624        # 8192 (reprompt) + 16384 (response) + 2048 (buffer)
MAX_REPROMPT_LEN=8192      # 2x default, teacher sees more of reference solution
GPU_MEMORY_UTILIZATION=0.80  # Can be high with topk distillation (no full-vocab tensor OOM)

# =============================================================================
# SETUP
# =============================================================================

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH=$PROJECT_ROOT:${WORK:+$WORK/python_packages:}$PYTHONPATH

# Directory configuration
BASE_DIR="${PROJECT_ROOT}"
LOG_DIR="${BASE_DIR}/output"
CKPT_DIR="${BASE_DIR}/ttrl_runs"
CUSTOM_REWARD_PATH="${BASE_DIR}/verl/utils/reward_score/feedback/__init__.py"

# Allow overriding experiment name suffix
SUFFIX=${1:-"local"}

export USER=${USER:-$(whoami)}

echo $PROJECT_ROOT
echo $PYTHONPATH
echo $USER

# =============================================================================
# EXECUTION
# =============================================================================

# Experiment name
if [ -n "$EXP_NAME_OVERRIDE" ]; then
    EXP_NAME="$EXP_NAME_OVERRIDE"
elif [ -n "$JOB_NAME" ]; then
    EXP_NAME="$JOB_NAME"
else
    MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
    EXP_NAME="OPSD-nemotron-hard-${MODEL_NAME}-${SUFFIX}"
fi

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.project_name=OPSD-${USER} \
trainer.group_name=$EXP_NAME \
trainer.experiment_name=$EXP_NAME \
trainer.nnodes=1 \
trainer.n_gpus_per_node=8 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.model.lora_rank=64 \
actor_rollout_ref.model.lora_alpha=128 \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
actor_rollout_ref.actor.use_kl_loss=false \
actor_rollout_ref.rollout.val_kwargs.n=4 \
actor_rollout_ref.rollout.val_kwargs.temperature=1.2 \
actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.apply_chat_template_kwargs={enable_thinking:true} \
actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_REPROMPT_LEN \
vars.dir=$BASE_DIR \
vars.log_dir=$LOG_DIR \
vars.ckpt_dir=$CKPT_DIR \
vars.task=$DATA_PATH \
custom_reward_function.path=$CUSTOM_REWARD_PATH \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"


echo "----------------------------------------------------------------"
echo "Starting OPSD Nemotron Hard Verified Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH (nemotron_hard_verified, 45K hard math)"
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_NAME"
echo "max_response_length: $MAX_RESPONSE_LENGTH"
echo "max_model_len: $MAX_MODEL_LEN"
echo "max_reprompt_len: $MAX_REPROMPT_LEN"
echo "enable_thinking: true"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "LoRA: rank=64, alpha=128"
echo "----------------------------------------------------------------"

# Append extra Hydra args if provided (e.g. offloading injected by submit.sh)
FINAL_ARGS="$ARGS"
if [ -n "$EXTRA_HYDRA_ARGS" ]; then
    FINAL_ARGS="$ARGS $EXTRA_HYDRA_ARGS"
    echo "Extra Hydra args: $EXTRA_HYDRA_ARGS"
fi

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $FINAL_ARGS
