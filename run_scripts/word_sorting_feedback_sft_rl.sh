#!/bin/bash

# Word sorting RL + Feedback Prediction SFT (no SDPO distillation).
#
# The model learns via two signals:
#   1. RL (CISPO) from environment reward scores
#   2. SFT on predicting the true environment feedback for its own answer
#
# The SDPO machinery is used for input construction (loss_mode=sdpo) but
# sdpo_loss_coef=0 disables the distillation loss itself.
#
# Key settings:
#   - rl_loss_mode=cispo, rl_loss_coef=1.0
#   - feedback_sft_loss_coef=0.1
#   - sdpo_loss_coef=0 (no distillation)
#   - include_environment_feedback=True (populates feedback from reward fn)
#
# Usage: ./run_scripts/word_sorting_feedback_sft_rl.sh [experiment_name_suffix]
# Note: MODEL_PATH and DATA_PATH can be set via environment (from submit.sh).

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo"

DATA_PATH="${DATA_PATH:-datasets/word_sorting}"

# Hyperparameters
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-6
FEEDBACK_SFT_COEF=0.1
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export N_GPUS_PER_NODE=8

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
    EXP_NAME="LOCAL-WORDSORT-FBSFT-RL-train${TRAIN_BATCH_SIZE}-fbcoef${FEEDBACK_SFT_COEF}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-${MODEL_NAME}-${SUFFIX}"
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
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
actor_rollout_ref.actor.self_distillation.alpha=1.0 \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01 \
actor_rollout_ref.actor.self_distillation.success_reward_threshold=1.0 \
actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
actor_rollout_ref.rollout.val_kwargs.n=4 \
data.max_response_length=1024 \
data.max_prompt_length=512 \
max_model_len=2048 \
vars.dir=$BASE_DIR \
vars.log_dir=$LOG_DIR \
vars.ckpt_dir=$CKPT_DIR \
vars.task=$DATA_PATH \
custom_reward_function.path=$CUSTOM_REWARD_PATH \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
actor_rollout_ref.actor.self_distillation.rl_loss_coef=1.0 \
actor_rollout_ref.actor.self_distillation.rl_loss_mode=cispo \
actor_rollout_ref.actor.self_distillation.sdpo_loss_coef=0 \
actor_rollout_ref.actor.self_distillation.feedback_sft_loss_coef=$FEEDBACK_SFT_COEF \
actor_rollout_ref.actor.self_distillation.include_environment_feedback=True \
actor_rollout_ref.actor.clip_ratio_low=1.0 \
actor_rollout_ref.actor.clip_ratio_high=3.0 \
algorithm.norm_adv_by_std_in_grpo=False"


echo "----------------------------------------------------------------"
echo "Starting Word Sorting RL + Feedback Prediction SFT"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "feedback_sft_loss_coef: $FEEDBACK_SFT_COEF"
echo "----------------------------------------------------------------"

FINAL_ARGS="$ARGS"
if [ -n "$EXTRA_HYDRA_ARGS" ]; then
    FINAL_ARGS="$ARGS $EXTRA_HYDRA_ARGS"
    echo "Extra Hydra args: $EXTRA_HYDRA_ARGS"
fi

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $FINAL_ARGS \
    $'actor_rollout_ref.actor.self_distillation.reprompt_template="{prompt}{solution}{feedback}\n\nCarefully re-read the sorting instructions and provide the correctly sorted comma-separated list."' \
    $'actor_rollout_ref.actor.self_distillation.feedback_sft_prompt="Check the sorting attempt above for correctness. Verify each adjacent pair is in the right ASCII/Unicode order by comparing character by character (digits 0-9 [ASCII 48-57] < uppercase A-Z [65-90] < lowercase a-z [97-122]; shorter prefixes come first). If incorrect, state the correct sorted order as a comma-separated list, list any missing or extra words, then identify specific misordered adjacent pairs with their ASCII values (e.g. \\"You placed \'apple\' before \'Banana\', but \'B\' has ASCII value 66 vs \'a\' has ASCII value 97, so \'Banana\' should come first\\"). If the sorting is already correct, say so."'
