#!/bin/bash

# Usage: ./run_physics.sh [alpha] [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo"

# Physics dataset
DATA_PATH="datasets/sciknoweval/physics/"

# Hyperparameters (from experiments/generalization/run_sdpo_all.sh)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-5
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=${1:-0.5}
MODEL_PATH="Qwen/Qwen3-8B"
export N_GPUS_PER_NODE=8

# Directory configuration (from SDPO user.yaml)
BASE_DIR="/work1/agrawal/devvrit/SDPO_custom"
LOG_DIR="${BASE_DIR}/output"
CKPT_DIR="${BASE_DIR}/ttrl_runs"
CUSTOM_REWARD_PATH="${BASE_DIR}/verl/utils/reward_score/feedback/__init__.py"

# GPU memory utilization (from SDPO user.yaml)
GPU_MEMORY_UTILIZATION=0.45

# Allow overriding experiment name suffix
SUFFIX=${2:-"local_physics"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located (go up one level from run/)
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH=$WORK/python_packages:$PROJECT_ROOT:$PYTHONPATH

# Configure wandb to work in containerized environment
export WANDB_MODE=offline

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}

echo $PROJECT_ROOT
echo $PYTHONPATH
echo $USER

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="LOCAL-SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-lambda${LAMBDA}-clip_adv_high${CLIP_ADV_HIGH}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.project_name=SDPO-${USER} \
trainer.group_name=$EXP_NAME \
trainer.experiment_name=$EXP_NAME \
trainer.nnodes=1 \
trainer.n_gpus_per_node=8 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
vars.dir=$BASE_DIR \
vars.log_dir=$LOG_DIR \
vars.ckpt_dir=$CKPT_DIR \
vars.task=$DATA_PATH \
custom_reward_function.path=$CUSTOM_REWARD_PATH \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
actor_rollout_ref.actor.fsdp_config.param_offload=false"


echo "----------------------------------------------------------------"
echo "Starting Local SDPO Training for Physics"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "----------------------------------------------------------------"

# Append extra Hydra args if provided (from multi-node setup)
FINAL_ARGS="$ARGS"
if [ -n "$EXTRA_HYDRA_ARGS" ]; then
    FINAL_ARGS="$ARGS $EXTRA_HYDRA_ARGS"
    echo "Extra Hydra args: $EXTRA_HYDRA_ARGS"
fi

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $FINAL_ARGS
