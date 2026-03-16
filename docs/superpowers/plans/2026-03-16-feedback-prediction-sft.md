# Feedback Prediction SFT Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a feedback-prediction SFT auxiliary loss to the RL training loop — the model is trained to predict what feedback it would receive for its answer, supervised on the true environment feedback.

**Architecture:** Piggyback on the existing SDPO infrastructure. During `_maybe_build_self_distillation_batch`, build a new "feedback prediction" input sequence: `[prompt + assistant:model_response + user:critique_prompt + assistant:true_feedback]`. During the actor update in `dp_actor.py`, forward pass through this sequence and compute cross-entropy SFT loss on just the true feedback tokens. This is controlled by `feedback_sft_loss_coef` (new config field). When enabled alongside `sdpo_loss_coef=0`, this gives RL + feedback SFT without distillation.

**Tech Stack:** Python, PyTorch, Hydra/OmegaConf, verl framework

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `verl/workers/config/actor.py` | Modify | Add `feedback_sft_loss_coef` and `feedback_sft_prompt` to `SelfDistillationConfig` |
| `verl/trainer/config/actor/actor.yaml` | Modify | Add matching YAML fields for Hydra |
| `verl/trainer/ppo/ray_trainer.py` | Modify | Build feedback prediction input sequences in `_maybe_build_self_distillation_batch` |
| `verl/workers/actor/dp_actor.py` | Modify | Compute SFT cross-entropy loss on feedback tokens during `update_policy` |
| `run_scripts/word_sorting_feedback_sft_rl.sh` | Create | New run script: RL + feedback SFT, no SDPO |

---

## Chunk 1: Config + YAML Registration

### Task 1: Add config fields to SelfDistillationConfig

**Files:**
- Modify: `verl/workers/config/actor.py:39-119` (SelfDistillationConfig)

- [ ] **Step 1: Add `feedback_sft_loss_coef` and `feedback_sft_prompt` fields**

In `verl/workers/config/actor.py`, add these two fields to `SelfDistillationConfig` (after `anti_hallucination_system_prompt`, before `__post_init__`):

```python
    feedback_sft_loss_coef: float = 0.0  # Weight for feedback-prediction SFT loss (0 = disabled)
    feedback_sft_prompt: str = ""  # Critique prompt appended after model response; true feedback is the SFT target
```

- [ ] **Step 2: Add matching YAML fields**

In `verl/trainer/config/actor/actor.yaml`, inside the `self_distillation:` block (after the `anti_hallucination_system_prompt` entry), add:

```yaml
  # Weight for feedback-prediction SFT auxiliary loss (0 = disabled).
  # When >0, the model is trained to predict the environment feedback for its
  # own answer. Input: [prompt + response + critique_prompt]. Target: true feedback tokens.
  feedback_sft_loss_coef: 0.0

  # Critique prompt inserted between the model's response and the true feedback.
  # The model sees [prompt][response][this prompt] and is SFT'd to produce [true feedback].
  # Pass domain-specific prompts via the run script.
  feedback_sft_prompt: ""
```

- [ ] **Step 3: Verify Hydra resolution**

Run: `cd /fsx/ubuntu/repos/SDPO_custom && conda run -n sdpo python -c "from verl.workers.config.actor import SelfDistillationConfig; c = SelfDistillationConfig(); print(c.feedback_sft_loss_coef, repr(c.feedback_sft_prompt))"`

Expected: `0.0 ''`

- [ ] **Step 4: Commit**

```bash
git add verl/workers/config/actor.py verl/trainer/config/actor/actor.yaml
git commit -m "feat: add feedback_sft_loss_coef and feedback_sft_prompt config fields"
```

---

## Chunk 2: Build feedback prediction inputs in ray_trainer.py

### Task 2: Build feedback SFT input sequences

**Files:**
- Modify: `verl/trainer/ppo/ray_trainer.py:1000-1234` (`_maybe_build_self_distillation_batch`)

The feedback SFT input needs to be a chat-formatted sequence:
```
[system message (if any)]
[user: original prompt]
[assistant: model's response]
[user: critique prompt]
[assistant: true feedback]
```

We only SFT on the true feedback tokens. To track which tokens are the SFT target, we tokenize the sequence in two parts: (1) everything up to and including the critique prompt + generation prompt, and (2) the true feedback. The SFT mask covers only part (2).

- [ ] **Step 1: Add feedback SFT sequence construction to `_maybe_build_self_distillation_batch`**

After the existing teacher input construction (around line 1161, after `teacher_position_ids = ...`), and before the distillation mask computation (around line 1164), add a block that builds the feedback SFT inputs. This block is gated on `feedback_sft_loss_coef > 0`.

The logic:
1. For each sample where `feedback_list[i]` is not None and non-empty, build a chat message sequence:
   - system messages (from `raw_prompt[i][:-1]`)
   - user: original prompt text (`prompt_texts[i]`)
   - assistant: model response text (`response_texts[i]`)
   - user: critique prompt (`feedback_sft_prompt` from config)
   - assistant: true feedback string (`feedback_list[i]`)
2. Tokenize the full sequence (with the feedback as the final assistant turn).
3. Also tokenize a "prefix" version (everything except the final assistant feedback) to determine where the SFT target tokens start.
4. Create `feedback_sft_input_ids`, `feedback_sft_attention_mask`, `feedback_sft_position_ids`, `feedback_sft_labels`, and `feedback_sft_mask` (1 for samples with feedback, 0 otherwise).

```python
        feedback_sft_loss_coef = self_distillation_cfg.get("feedback_sft_loss_coef", 0.0)
        feedback_sft_data = {}
        if feedback_sft_loss_coef > 0.0:
            feedback_sft_prompt_text = self_distillation_cfg.get("feedback_sft_prompt", "")
            if not feedback_sft_prompt_text:
                raise ValueError(
                    "feedback_sft_prompt must be set when feedback_sft_loss_coef > 0"
                )

            # Build full messages (with feedback as final assistant turn)
            # and prefix messages (without feedback) for each sample
            full_messages_list = []
            prefix_messages_list = []
            has_feedback = []
            for i in range(batch_size):
                fb = feedback_list[i]
                if fb and isinstance(fb, str) and fb.strip():
                    sys_msgs = list(batch.non_tensor_batch["raw_prompt"][i][:-1])
                    full_msgs = sys_msgs + [
                        {"role": "user", "content": prompt_texts[i]},
                        {"role": "assistant", "content": response_texts[i]},
                        {"role": "user", "content": feedback_sft_prompt_text},
                        {"role": "assistant", "content": fb},
                    ]
                    prefix_msgs = sys_msgs + [
                        {"role": "user", "content": prompt_texts[i]},
                        {"role": "assistant", "content": response_texts[i]},
                        {"role": "user", "content": feedback_sft_prompt_text},
                    ]
                    full_messages_list.append(full_msgs)
                    prefix_messages_list.append(prefix_msgs)
                    has_feedback.append(True)
                else:
                    # Placeholder — will be masked out
                    sys_msgs = list(batch.non_tensor_batch["raw_prompt"][i][:-1])
                    placeholder = sys_msgs + [
                        {"role": "user", "content": prompt_texts[i]},
                    ]
                    full_messages_list.append(placeholder)
                    prefix_messages_list.append(placeholder)
                    has_feedback.append(False)

            # Tokenize full sequences (with feedback as content, not generation)
            full_tok = self.tokenizer.apply_chat_template(
                full_messages_list,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                continue_final_message=False,
                add_generation_prompt=False,
                enable_thinking=False,
                max_length=self_distillation_cfg.max_reprompt_len,
                padding=True,
                truncation=True,
            )
            # Tokenize prefix (everything before the feedback)
            prefix_tok = self.tokenizer.apply_chat_template(
                prefix_messages_list,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                continue_final_message=False,
                add_generation_prompt=True,
                enable_thinking=False,
                max_length=self_distillation_cfg.max_reprompt_len,
                padding=True,
                truncation=True,
            )

            full_ids = full_tok["input_ids"].to(device)
            full_mask = full_tok["attention_mask"].to(device)

            # Build labels: -100 for prefix tokens, actual token ids for feedback tokens
            # The feedback tokens start at prefix_length for each sample
            prefix_lengths = prefix_tok["attention_mask"].sum(dim=1)  # (batch_size,)
            labels = torch.full_like(full_ids, -100)
            for i in range(batch_size):
                if has_feedback[i]:
                    plen = prefix_lengths[i].item()
                    # Set labels for feedback tokens (from prefix_length to end of valid tokens)
                    valid_len = full_mask[i].sum().item()
                    if plen < valid_len:
                        labels[i, plen:valid_len] = full_ids[i, plen:valid_len]

            fb_position_ids = compute_position_id_with_mask(full_mask)
            fb_sft_sample_mask = torch.tensor(has_feedback, dtype=torch.float32, device=device)

            feedback_sft_data = {
                "feedback_sft_input_ids": full_ids,
                "feedback_sft_attention_mask": full_mask,
                "feedback_sft_position_ids": fb_position_ids,
                "feedback_sft_labels": labels,
                "feedback_sft_mask": fb_sft_sample_mask,
            }

            # Metrics
            n_fb = sum(has_feedback)
            metrics["feedback_sft/samples_with_feedback"] = n_fb
            metrics["feedback_sft/fraction_with_feedback"] = n_fb / batch_size
            if n_fb > 0:
                fb_token_counts = [(labels[i] != -100).sum().item() for i in range(batch_size) if has_feedback[i]]
                metrics["feedback_sft/avg_feedback_tokens"] = sum(fb_token_counts) / len(fb_token_counts)
```

- [ ] **Step 2: Include feedback SFT tensors in the returned DataProto**

At the return statement (around line 1229), merge `feedback_sft_data` into the tensors dict:

Change:
```python
        return DataProto.from_dict(tensors={
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_position_ids": teacher_position_ids,
            "self_distillation_mask": self_distillation_mask,
        }), metrics
```

To:
```python
        result_tensors = {
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_position_ids": teacher_position_ids,
            "self_distillation_mask": self_distillation_mask,
        }
        result_tensors.update(feedback_sft_data)
        return DataProto.from_dict(tensors=result_tensors), metrics
```

- [ ] **Step 3: Add feedback SFT keys to `select_keys` in `update_policy`**

In `verl/workers/actor/dp_actor.py`, in `update_policy` (around line 742), where SDPO keys are added to `select_keys`, also add the feedback SFT keys:

```python
        if self_distillation_enabled and "feedback_sft_input_ids" in data.batch.keys():
            select_keys.extend([
                "feedback_sft_input_ids",
                "feedback_sft_attention_mask",
                "feedback_sft_position_ids",
                "feedback_sft_labels",
                "feedback_sft_mask",
            ])
```

- [ ] **Step 4: Commit**

```bash
git add verl/trainer/ppo/ray_trainer.py verl/workers/actor/dp_actor.py
git commit -m "feat: build feedback prediction SFT input sequences in trainer"
```

---

## Chunk 3: Compute feedback SFT loss in dp_actor.py

### Task 3: Add feedback SFT loss computation

**Files:**
- Modify: `verl/workers/actor/dp_actor.py:846-1000` (inside `update_policy`, after SDPO loss block)

- [ ] **Step 1: Add feedback SFT forward pass and loss computation**

After the SDPO + RL loss block (around line 943 where `_pending_rl_loss` is set), add the feedback SFT loss computation. This should be inside the `if self_distillation_enabled:` block, right before the `else:` at line 950.

The approach:
1. Check if `feedback_sft_loss_coef > 0` and `feedback_sft_input_ids` exists in `model_inputs`
2. Build a new input dict for the feedback SFT forward pass
3. Forward through `self.actor_module` (with gradients — this is SFT on the student)
4. Compute cross-entropy loss on the labeled tokens (where `labels != -100`)
5. Mask by `feedback_sft_mask` (per-sample mask for samples that have feedback)
6. Scale by `feedback_sft_loss_coef` and add to `pg_loss`

```python
                        # Feedback prediction SFT loss
                        feedback_sft_loss_coef = getattr(self_distillation_cfg, "feedback_sft_loss_coef", 0.0)
                        if feedback_sft_loss_coef > 0.0 and "feedback_sft_input_ids" in model_inputs:
                            fb_input_ids = model_inputs["feedback_sft_input_ids"]
                            fb_attention_mask = model_inputs["feedback_sft_attention_mask"]
                            fb_position_ids = model_inputs["feedback_sft_position_ids"]
                            fb_labels = model_inputs["feedback_sft_labels"]
                            fb_sample_mask = model_inputs["feedback_sft_mask"]

                            # Forward pass through student model (with gradients)
                            with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                                fb_outputs = self.actor_module(
                                    input_ids=fb_input_ids,
                                    attention_mask=fb_attention_mask,
                                    position_ids=fb_position_ids,
                                )
                                fb_logits = fb_outputs.logits  # (bsz, seq_len, vocab_size)

                            # Shift logits and labels for next-token prediction
                            # logits[:, :-1] predicts labels[:, 1:]
                            shift_logits = fb_logits[:, :-1, :].contiguous()
                            shift_labels = fb_labels[:, 1:].contiguous()

                            # Compute per-token cross-entropy (only on labeled positions)
                            # Use reduction='none' to get per-token losses
                            fb_ce_loss = torch.nn.functional.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                ignore_index=-100,
                                reduction='none',
                            ).view(shift_labels.shape)  # (bsz, seq_len-1)

                            # Mean over valid tokens per sample, then mean over samples with feedback
                            valid_token_mask = (shift_labels != -100).float()
                            per_sample_loss = (fb_ce_loss * valid_token_mask).sum(dim=1) / valid_token_mask.sum(dim=1).clamp(min=1.0)
                            # Mask to only samples that have feedback
                            fb_sft_loss = (per_sample_loss * fb_sample_mask).sum() / fb_sample_mask.sum().clamp(min=1.0)

                            pg_loss = pg_loss + feedback_sft_loss_coef * fb_sft_loss

                            micro_batch_metrics["actor/feedback_sft_loss"] = fb_sft_loss.detach().item()
                            micro_batch_metrics["actor/feedback_sft_loss_coef"] = feedback_sft_loss_coef
```

- [ ] **Step 2: Commit**

```bash
git add verl/workers/actor/dp_actor.py
git commit -m "feat: compute feedback prediction SFT loss in actor update"
```

---

## Chunk 4: Run script

### Task 4: Create word_sorting_feedback_sft_rl.sh

**Files:**
- Create: `run_scripts/word_sorting_feedback_sft_rl.sh`

- [ ] **Step 1: Create the run script**

Based on `word_sorting_sdpo_rl_cotrain.sh` but with:
- `loss_mode=sdpo` (required to activate the SDPO path which builds the feedback inputs)
- `sdpo_loss_coef=0` (disable distillation loss)
- `rl_loss_coef=1.0` + `rl_loss_mode=cispo` (keep RL)
- `feedback_sft_loss_coef=0.1` (new: enable feedback SFT)
- `feedback_sft_prompt` = domain-specific word-sorting critique prompt (passed via `$'...'` syntax like the reprompt_template)
- `include_environment_feedback=True` (so feedback_list is populated from the reward function)

The domain-specific feedback SFT prompt for word sorting:

```
Review your sorting attempt above. For each adjacent pair of words, check whether they are in the correct ASCII/Unicode order. Remember: digits (0-9, ASCII 48-57) come before uppercase letters (A-Z, ASCII 65-90), which come before lowercase letters (a-z, ASCII 97-122). Words are compared character by character from left to right; if one word is a prefix of another, the shorter word comes first. List any pairs that are out of order and explain why, citing the specific ASCII values of the differing characters. If all pairs are correctly ordered, state that the sorting is correct.
```

```bash
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
    $'actor_rollout_ref.actor.self_distillation.feedback_sft_prompt=Review your sorting attempt above. For each adjacent pair of words, check whether they are in the correct ASCII/Unicode order. Remember: digits (0-9, ASCII 48-57) come before uppercase letters (A-Z, ASCII 65-90), which come before lowercase letters (a-z, ASCII 97-122). Words are compared character by character from left to right; if one word is a prefix of another, the shorter word comes first. List any pairs that are out of order and explain why, citing the specific ASCII values of the differing characters. If all pairs are correctly ordered, state that the sorting is correct.'
```

- [ ] **Step 2: Make executable**

```bash
chmod +x run_scripts/word_sorting_feedback_sft_rl.sh
```

- [ ] **Step 3: Commit**

```bash
git add run_scripts/word_sorting_feedback_sft_rl.sh
git commit -m "feat: add word_sorting feedback prediction SFT + RL run script"
```

---

## Summary of changes

1. **Config** (`actor.py` + `actor.yaml`): 2 new fields — `feedback_sft_loss_coef` (float, default 0.0) and `feedback_sft_prompt` (str, default "")
2. **Trainer** (`ray_trainer.py`): ~70 lines in `_maybe_build_self_distillation_batch` to build feedback SFT input sequences with labels
3. **Actor** (`dp_actor.py`): ~5 lines to add keys to select_keys, ~30 lines for forward pass + cross-entropy loss computation
4. **Run script**: New `word_sorting_feedback_sft_rl.sh` with domain-specific critique prompt
