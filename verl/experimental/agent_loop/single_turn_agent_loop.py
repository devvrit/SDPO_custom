# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

        # Interruption config
        rollout_cfg = self.config.actor_rollout_ref.rollout
        interruption_cfg = getattr(rollout_cfg, 'interruption', None)
        self.interruption_enabled = interruption_cfg is not None and getattr(interruption_cfg, 'enable', False)
        if self.interruption_enabled:
            self.max_thinking_tokens = interruption_cfg.max_thinking_tokens
            self.max_answer_tokens = interruption_cfg.max_answer_tokens
            self.interruption_text = interruption_cfg.interruption_text
            self.think_end_token = interruption_cfg.think_end_token
            self.mask_interruption_loss = interruption_cfg.mask_interruption_loss

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )

        if self.interruption_enabled:
            return await self._run_two_phase(prompt_ids, sampling_params, images, videos, multi_modal_data)

        # 3. generate sequences (original single-call path)
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        response_mask = [1] * len(output.token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )
        return output

    async def _run_two_phase(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images,
        videos,
        multi_modal_data: dict,
    ) -> AgentLoopOutput:
        """Two-phase generation with interruption injection."""
        metrics = {}

        # === Phase 1: Thinking ===
        phase1_params = dict(sampling_params)
        phase1_params["max_tokens"] = self.max_thinking_tokens
        phase1_params["stop"] = [self.think_end_token]
        phase1_params["include_stop_str_in_output"] = True

        with simple_timer("generate_sequences_phase1", metrics):
            phase1_output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=phase1_params,
                image_data=images,
                video_data=videos,
            )

        phase1_ids = list(phase1_output.token_ids)
        phase1_logprobs = list(phase1_output.log_probs) if phase1_output.log_probs else [0.0] * len(phase1_ids)

        # Check if </think> was produced naturally
        think_end_ids = self.tokenizer.encode(self.think_end_token, add_special_tokens=False)
        naturally_ended = (
            len(phase1_ids) >= len(think_end_ids)
            and phase1_ids[-len(think_end_ids):] == think_end_ids
        )

        interrupted = not naturally_ended

        # === Interruption injection ===
        interruption_ids = []
        interruption_logprobs = []
        if interrupted:
            interruption_ids = self.tokenizer.encode(self.interruption_text, add_special_tokens=False)
            interruption_logprobs = [0.0] * len(interruption_ids)

        # === Phase 2: Answer ===
        phase2_prompt_ids = prompt_ids + phase1_ids + interruption_ids
        phase2_params = dict(sampling_params)
        phase2_params["max_tokens"] = self.max_answer_tokens
        phase2_params["stop"] = ["<|im_end|>"]
        phase2_params["include_stop_str_in_output"] = True

        with simple_timer("generate_sequences_phase2", metrics):
            phase2_output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=phase2_prompt_ids,
                sampling_params=phase2_params,
                image_data=images,
                video_data=videos,
            )

        phase2_ids = list(phase2_output.token_ids)
        phase2_logprobs = list(phase2_output.log_probs) if phase2_output.log_probs else [0.0] * len(phase2_ids)

        # Check if phase 2 was truncated (hit max tokens without producing <|im_end|>)
        im_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        phase2_truncated = not (
            len(phase2_ids) >= len(im_end_ids)
            and phase2_ids[-len(im_end_ids):] == im_end_ids
        )

        # === Combine ===
        response_ids = phase1_ids + interruption_ids + phase2_ids
        response_logprobs = phase1_logprobs + interruption_logprobs + phase2_logprobs

        # Build response_mask
        phase1_mask = [1] * len(phase1_ids)
        if interrupted and self.mask_interruption_loss:
            interruption_mask = [0] * len(interruption_ids)
        else:
            interruption_mask = [1] * len(interruption_ids)
        phase2_mask = [1] * len(phase2_ids)
        response_mask = phase1_mask + interruption_mask + phase2_mask

        # Truncate to response_length
        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]
        response_logprobs = response_logprobs[: self.response_length]

        # Extra fields for metrics
        extra_fields = {
            "interrupted": interrupted,
            "phase1_length": len(phase1_ids),
            "phase2_length": len(phase2_ids),
            "phase2_truncated": phase2_truncated,
        }

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=None,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
        return output
