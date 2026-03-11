# OPSD Paper Reproduction Plan

Reproducing Table 2 from "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models" (arXiv:2601.18734).

## Goal

Train Qwen3-8B with OPSD (On-Policy Self-Distillation) on OpenThoughts math dataset and evaluate on 4 benchmarks to reproduce paper Table 2 results.

## Setup Completed

### Dataset
- **Training data**: `datasets/openthoughts_math/train.parquet` — 30K math samples from OpenThoughts-114k
- **Eval benchmarks** (140 problems total):
  - `datasets/eval_benchmarks/aime_2024.parquet` (30 problems)
  - `datasets/eval_benchmarks/aime_2025.parquet` (30 problems)
  - `datasets/eval_benchmarks/hmmt_feb_2025.parquet` (30 problems)
  - `datasets/eval_benchmarks/amo_bench.parquet` (50 problems)
- Prepared with `reproduce_paper/prepare_data.py`

### Config Files
- **`verl/trainer/config/opsd.yaml`** — Main OPSD training config
  - loss_mode: sdpo, alpha: 0.5, use_dataset_solution: true
  - teacher_update_rate: 0.0 (fixed teacher = initial model)
  - rl_loss_coef: 0.0 (pure distillation, no RL)
  - train_batch_size: 32, n=1 rollout per prompt
  - max_response_length: 2048, max_prompt_length: 2048, max_model_len: 6144
  - val_before_train: False, test_freq: 50
  - Solution/reprompt templates for teacher privileged info (Figure 2 of paper)

- **`run_scripts/opsd.sh`** — Training launch script
  - LoRA rank=64, alpha=128, target_modules=all-linear
  - LR: 2e-5, warmup_ratio: 0.1, cosine decay
  - gpu_memory_utilization: 0.75
  - val_kwargs: n=4, temp=1.2, top_p=0.95
  - 1 epoch (~937 steps at batch_size=32 over 30K samples)

- **`run_scripts/opsd_eval.sh`** — Standalone eval script (paper Table 5 settings)
  - max_response_length: 38912 (paper's max_new_tokens)
  - max_model_len: 40960
  - gpu_memory_utilization: 0.9
  - enable_thinking: true
  - val_kwargs: n=16, temp=1.2, top_p=0.95
  - val_only mode, supports checkpoint loading via CLI args

### Key Code Modifications
- **`verl/trainer/ppo/ray_trainer.py`**: Added `"reference_solution"` to preserved keys in `_get_gen_batch()` so teacher can access ground-truth solutions from dataset
- **`verl/workers/rollout/vllm_rollout/vllm_async_server.py`**: Conditional max_model_len override (only if not already set in config)
- **`verl/workers/rollout/sglang_rollout/async_sglang_server.py`**: Same conditional max_model_len fix

## Training Runs

### Run 1: Full-Vocabulary JSD (paper default)
- **Suffix**: `repro10`
- **Checkpoint dir**: `ttrl_runs/openthoughts_math-opsd-qwen3-8b-ml.p4de.24xlarge-repro10/`
- **Latest checkpoint**: global_step_150 (out of 937)
- **Config**: Full-vocab JSD distillation (distillation_topk: null)
- **Submit command**: `./submit.sh -c opsd -s repro10 -d datasets/openthoughts_math -t 72 -r`
- **Current job**: 117 (resuming from step 150)

### Run 2: Top-K=100 Distillation (ablation)
- **Suffix**: `topk100`
- **Checkpoint dir**: `ttrl_runs/openthoughts_math-opsd-qwen3-8b-ml.p4de.24xlarge-topk100/`
- **Latest checkpoint**: global_step_10 (out of 937)
- **Config**: Top-k=100 distillation
- **Submit command**: `./submit.sh -c opsd -s topk100 -d datasets/openthoughts_math -t 72 -r actor_rollout_ref.actor.self_distillation.distillation_topk=100`
- **Current job**: 118 (resuming from step 10)

### Training Speed
- ~190s/step on 1x p4de.24xlarge (8x A100 80GB)
- Generation dominates (~175s, 85% of step time)
- Colocated architecture: vLLM + FSDP share same GPUs with sleep/wake cycle
- Estimated ~50 hours for full 937 steps from scratch

## Eval Jobs

Paper eval settings (Table 5): n=16, temp=1.2, top_p=0.95, max_new_tokens=38912, thinking enabled.

| Job | Suffix | What | Submit |
|-----|--------|------|--------|
| 119 | base | Base Qwen3-8B (no training) | `./submit.sh -c opsd_eval -s base -d datasets/openthoughts_math -t 12` |
| 120 | repro10-step150 | Full-vocab @ step 150 | `./submit.sh -c opsd_eval -s repro10-step150 -d datasets/openthoughts_math -t 12 trainer.resume_mode=resume_path trainer.resume_from_path=.../global_step_150` |
| 121 | topk100-step10 | Top-k=100 @ step 10 | `./submit.sh -c opsd_eval -s topk100-step10 -d datasets/openthoughts_math -t 12 trainer.resume_mode=resume_path trainer.resume_from_path=.../global_step_10` |

## Paper Hyperparameters Reference

### Training (Table 6)
- Model: Qwen3-8B
- LoRA: rank=64, alpha=128
- LR: 2e-5, warmup ratio: 0.1, cosine schedule
- Batch size: 32, n=1 rollout per prompt
- Max completion length: 2048
- JSD alpha (beta): 0.5
- Teacher: initial model (no EMA), sees reference solution y*
- 1 epoch on 30K OpenThoughts math subset

### Evaluation (Table 5)
- Max new tokens: 38912
- Thinking mode: enabled
- Temperature: 1.2, top_p: 0.95
- 16 samples per prompt (avg@16, best@16, maj@16)

## Known Issues
- **Sympy errors in reward scoring**: Non-fatal TypeError/TimeoutError/AttributeError from math answer checker. Affected samples get score=0. Does not crash training.
- **Checkpoint cleanup**: `max_actor_ckpt_to_keep=1` means only the latest checkpoint retains full model weights. Older checkpoints keep only data.pt.
- **std_normalize_sdpo**: Normalization flag exists but is OFF (False) for OPSD runs. This is correct per paper.

## TODO
- [ ] Training runs complete (repro10: 150→937, topk100: 10→937)
- [ ] Eval base model results
- [ ] Eval intermediate checkpoints as training progresses
- [ ] Final eval at end of training for both runs
- [ ] Compare with paper Table 2 numbers
