#!/bin/bash

# Word sorting SDPO + CISPO student RL — THINKING MODE enabled.
#
# Combines word_sorting_sdpo_rl.sh with thinking-mode settings:
#   - enable_thinking=true via apply_chat_template_kwargs
#   - Config: sdpo_think (thinking-appropriate reprompt templates)
#   - max_response_length: 1024 → 8192 (thinking generates far more tokens)
#   - max_prompt_length: 512 (unchanged, word sorting prompts are short)
#   - max_model_len: 16384 (accommodates reprompt + thinking response)
#   - max_reprompt_len: 4096
#
# Key settings matching grpo_cispo RL:
#   - rl_loss_mode=cispo (not vanilla PPO)
#   - clip_ratio_low=1.0, clip_ratio_high=3.0 (asymmetric [0, 4])
#   - norm_adv_by_std_in_grpo=False (mean-centered only)
#
# Usage: ./run_scripts/word_sorting_sdpo_rl_think.sh [experiment_name_suffix]
# Note: MODEL_PATH and DATA_PATH can be set via environment (from submit.sh).

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo_think"

DATA_PATH="${DATA_PATH:-datasets/word_sorting}"

# Hyperparameters
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-6
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=1.0
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export N_GPUS_PER_NODE=8

# Thinking-mode sequence lengths
MAX_RESPONSE_LENGTH=8192
MAX_PROMPT_LENGTH=512
MAX_MODEL_LEN=16384       # prompt + reprompt + thinking response
MAX_REPROMPT_LEN=4096     # word sorting reprompts are short

# =============================================================================
# SETUP
# =============================================================================

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH=$PROJECT_ROOT:${WORK:+$WORK/python_packages:}$PYTHONPATH

BASE_DIR="${PROJECT_ROOT}"
LOG_DIR="${BASE_DIR}/output"
CKPT_DIR="${BASE_DIR}/ttrl_runs"
CUSTOM_REWARD_PATH="${BASE_DIR}/verl/utils/reward_score/feedback/__init__.py"

GPU_MEMORY_UTILIZATION=0.8

SUFFIX=${1:-"local"}

export USER=${USER:-$(whoami)}

echo $PROJECT_ROOT
echo $PYTHONPATH
echo $USER

# =============================================================================
# EXECUTION
# =============================================================================

if [ -n "$EXP_NAME_OVERRIDE" ]; then
    EXP_NAME="$EXP_NAME_OVERRIDE"
elif [ -n "$JOB_NAME" ]; then
    EXP_NAME="$JOB_NAME"
else
    MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
    EXP_NAME="LOCAL-WORDSORT-SDPO-RL-THINK-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-${MODEL_NAME}-${SUFFIX}"
fi

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.project_name=SDPO-${USER} \
trainer.group_name=SDPO-${USER} \
trainer.experiment_name=$EXP_NAME \
trainer.nnodes=1 \
trainer.n_gpus_per_node=8 \
trainer.total_epochs=10 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.self_distillation.distillation_topk=20 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01 \
actor_rollout_ref.actor.self_distillation.success_reward_threshold=1.0 \
actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
actor_rollout_ref.rollout.val_kwargs.n=4 \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.apply_chat_template_kwargs={enable_thinking:true} \
max_model_len=$MAX_MODEL_LEN \
actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_REPROMPT_LEN \
vars.dir=$BASE_DIR \
vars.log_dir=$LOG_DIR \
vars.ckpt_dir=$CKPT_DIR \
vars.task=$DATA_PATH \
custom_reward_function.path=$CUSTOM_REWARD_PATH \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
data.val_files=[\${vars.dir}/\${vars.task}/test.parquet] \
actor_rollout_ref.actor.self_distillation.rl_loss_coef=1.0 \
actor_rollout_ref.actor.self_distillation.rl_loss_mode=cispo \
actor_rollout_ref.actor.self_distillation.sdpo_loss_coef=0.02 \
actor_rollout_ref.actor.clip_ratio_low=1.0 \
actor_rollout_ref.actor.clip_ratio_high=3.0 \
algorithm.norm_adv_by_std_in_grpo=False"


echo "----------------------------------------------------------------"
echo "Starting Word Sorting SDPO+CISPO RL Training (THINKING MODE)"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_NAME"
echo "enable_thinking: true"
echo "max_response_length: $MAX_RESPONSE_LENGTH"
echo "max_prompt_length: $MAX_PROMPT_LENGTH"
echo "max_model_len: $MAX_MODEL_LEN"
echo "max_reprompt_len: $MAX_REPROMPT_LEN"
echo "----------------------------------------------------------------"

FINAL_ARGS="$ARGS"
if [ -n "$EXTRA_HYDRA_ARGS" ]; then
    FINAL_ARGS="$ARGS $EXTRA_HYDRA_ARGS"
    echo "Extra Hydra args: $EXTRA_HYDRA_ARGS"
fi

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $FINAL_ARGS \
    $'actor_rollout_ref.actor.self_distillation.reprompt_template="{prompt}{solution}{feedback}\n\nCarefully re-read the sorting instructions and provide the correctly sorted comma-separated list."'
