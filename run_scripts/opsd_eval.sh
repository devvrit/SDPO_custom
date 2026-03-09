#!/bin/bash

# Usage: ./submit.sh -c opsd_eval -s SUFFIX [-d DATA] [extra hydra args...]
#
# Evaluates an OPSD-trained model on the paper's Table 2 benchmarks:
#   - AIME 2024 (30 problems)
#   - AIME 2025 (30 problems)
#   - HMMT February 2025 (30 problems)
#   - AMO-Bench (50 problems)
#
# Evaluation settings (paper Table 5):
#   - 16 samples per prompt (average@16)
#   - Temperature 1.2, top_p 0.95
#   - max_new_tokens 38912 (thinking mode enabled)
#   - val_only mode (no training)
#
# Examples:
#   # Evaluate base model (no checkpoint):
#   ./submit.sh -c opsd_eval -s base -d datasets/openthoughts_math
#
#   # Evaluate a specific checkpoint:
#   ./submit.sh -c opsd_eval -s step150 -d datasets/openthoughts_math \
#     trainer.resume_mode=resume_path trainer.resume_from_path=/path/to/global_step_150

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="opsd"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"

# First positional arg is the suffix (passed by submit.sh)
SUFFIX=${1:-"eval"}

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH=$PROJECT_ROOT:${WORK:+$WORK/python_packages:}$PYTHONPATH

BASE_DIR="${PROJECT_ROOT}"
EVAL_DIR="${BASE_DIR}/datasets/eval_benchmarks"
CUSTOM_REWARD_PATH="${BASE_DIR}/verl/utils/reward_score/feedback/__init__.py"

# Paper Table 5: max_new_tokens=38912 with thinking enabled
MAX_RESPONSE_LENGTH=38912
MAX_PROMPT_LENGTH=2048
MAX_MODEL_LEN=40960

# Maximize KV cache for long-context eval
GPU_MEMORY_UTILIZATION=0.85
export N_GPUS_PER_NODE=8
export USER=${USER:-$(whoami)}

# Eval dataset files
VAL_FILES="[${EVAL_DIR}/aime_2024.parquet,${EVAL_DIR}/aime_2025.parquet,${EVAL_DIR}/hmmt_feb_2025.parquet,${EVAL_DIR}/amo_bench.parquet]"

# =============================================================================
# EXPERIMENT NAME
# =============================================================================

# SUFFIX already set from $1 above or defaults to "eval"

if [ -n "$EXP_NAME_OVERRIDE" ]; then
    EXP_NAME="$EXP_NAME_OVERRIDE"
elif [ -n "$JOB_NAME" ]; then
    EXP_NAME="$JOB_NAME"
else
    MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
    EXP_NAME="OPSD-eval-${MODEL_NAME}-${SUFFIX}"
fi

# =============================================================================
# BUILD ARGS
# =============================================================================

ARGS="trainer.project_name=OPSD-eval-${USER} \
trainer.group_name=$EXP_NAME \
trainer.experiment_name=$EXP_NAME \
trainer.nnodes=1 \
trainer.n_gpus_per_node=8 \
trainer.val_only=true \
trainer.val_before_train=true \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.model.lora_rank=64 \
actor_rollout_ref.model.lora_alpha=128 \
actor_rollout_ref.rollout.n=1 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
actor_rollout_ref.rollout.val_kwargs.temperature=1.2 \
actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
max_model_len=$MAX_MODEL_LEN \
actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
data.apply_chat_template_kwargs.enable_thinking=true \
data.train_files=[${BASE_DIR}/datasets/openthoughts_math/train.parquet] \
data.val_files=${VAL_FILES} \
vars.dir=$BASE_DIR \
vars.log_dir=${BASE_DIR}/output \
vars.ckpt_dir=${BASE_DIR}/ttrl_runs \
vars.task=datasets/openthoughts_math \
custom_reward_function.path=$CUSTOM_REWARD_PATH \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"

echo "----------------------------------------------------------------"
echo "OPSD Evaluation (Paper Table 2 Benchmarks)"
echo "Experiment: $EXP_NAME"
echo "Model: $MODEL_PATH"
echo "Benchmarks: AIME24, AIME25, HMMT-Feb25, AMO-Bench"
echo "Eval: n=16, temp=1.2, top_p=0.95"
echo "Max response: $MAX_RESPONSE_LENGTH tokens (thinking enabled)"
echo "Max model len: $MAX_MODEL_LEN"
echo "----------------------------------------------------------------"

# Append extra Hydra args if provided
FINAL_ARGS="$ARGS"
if [ -n "$EXTRA_HYDRA_ARGS" ]; then
    FINAL_ARGS="$ARGS $EXTRA_HYDRA_ARGS"
    echo "Extra Hydra args: $EXTRA_HYDRA_ARGS"
fi

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "datasets/openthoughts_math" $FINAL_ARGS
