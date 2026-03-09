"""External feedback generation via Gemini API for self-distillation.

This module provides utilities to:
- Extract summaries (post-</think> text) from model responses
- Build feedback prompts from student summaries
- Call the Gemini API asynchronously to generate structured feedback
- Orchestrate group-level feedback generation across UID groups
"""

import asyncio
import logging
import os
import random
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_EXTERNAL_FEEDBACK_PROMPT_TEMPLATE = """You are analyzing student attempts at solving a math problem to create helpful feedback for a NEW student who will attempt this problem for the first time.

Problem: {problem}
Ground Truth Answer: {answer}

Student Summaries (their final answers after thinking):
{summaries}

First, reason through each student summary carefully. Analyze what each student did correctly and incorrectly. Consider whether different students may have taken valid alternative approaches.

Then, based on your analysis and the ground truth, create feedback specifically designed to help a NEW student who has never seen this problem before. The feedback should:
1. Warn about common mistakes, misconceptions, and pitfalls to avoid (learned from the attempts above)
2. Suggest effective problem-solving strategies and key concepts to consider (note: there may be multiple valid solution paths - do not assume only one correct method exists)
3. Provide hints about important reasoning steps without giving away the solution

Important guidelines:
- Do not leak the final answer in your feedback
- Be aware that multiple correct approaches may exist - avoid insisting on a single "correct" method if alternatives are valid. If multiple valid approaches exist, suggest all of them.
- Write the feedback as actionable guidance that will help a first-time solver improve their problem-solving process
- Frame the feedback as forward-looking advice (e.g., "Consider...", "Watch out for...", "A useful approach is...") rather than commentary on past attempts
- Warn about common mistakes, misconceptions, and pitfalls to avoid.

After your reasoning, provide your final summarized feedback inside <feedback> and </feedback> tags. This feedback will be given directly to a new student, so write it in second person (e.g., "You should consider...") and make it immediately useful for someone approaching this problem fresh.
"""

DEFAULT_PROXY_TEACHER_TEMPLATE = (
    "You are solving a math problem.\n"
    "Problem: {problem}\n\n"
    "You have received the following feedback from reviewing multiple solution attempts:\n"
    "Feedback: {feedback}\n\n"
    "Now solve the original problem. Do not mention or refer to the critique or the revision process. "
    "Use the feedback/critique only to improve correctness, clarity, and reasoning. Avoid using phrases "
    "like \"Correctly applying the critique...\" or \"Using the feedback...\", \"Given the hint:..\" etc., "
    "as your solution should stand alone. Basically, reading your thought process and summarized solution, "
    "it should not be apparent that you had access to any privileged information. "
    "Now think step by step and write your answer in \\boxed{{}} format. "
)


def extract_summary_from_response(response_text: str, filter_incomplete: bool = True) -> Optional[str]:
    """Extract the summary part from a model response (text after </think> tag).

    Args:
        response_text: Full response text from the model.
        filter_incomplete: If True, return None for traces without </think> tag.

    Returns:
        The summary text after </think>, or None if incomplete and filtering enabled.
    """
    if "</think>" in response_text:
        parts = response_text.split("</think>", 1)
        if len(parts) == 2:
            summary = parts[1].strip()
            return summary if summary else None

    if filter_incomplete:
        return None

    return response_text.strip() if response_text.strip() else None


def extract_feedback_from_response(
    text: str,
    start_tag: str = "<feedback>",
    end_tag: str = "</feedback>",
) -> str:
    """Parse feedback content from between start/end tags. Falls back to full text if tags missing.

    Args:
        text: Raw response text from the feedback model.
        start_tag: Opening tag to search for.
        end_tag: Closing tag to search for.

    Returns:
        Extracted feedback string.
    """
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)
    if start_idx == -1:
        return text
    elif end_idx == -1:
        return text[start_idx + len(start_tag):]
    else:
        return text[start_idx + len(start_tag):end_idx]


async def get_external_feedback(
    prompt: str,
    temperature: float = 0.0,
    max_output_tokens: int = 14000,
    model: str = "gemini-3-flash-preview",
    max_retries: int = 3,
    retry_delay_min: float = 15.0,
    retry_delay_max: float = 45.0,
) -> Optional[str]:
    """Call the Gemini API to generate feedback.

    Auth via GEMINI_API_KEY env var. Retries on failure with random delay.

    Args:
        prompt: The full prompt to send to the model.
        temperature: Sampling temperature.
        max_output_tokens: Maximum output tokens.
        model: Gemini model identifier.
        max_retries: Maximum number of retry attempts.
        retry_delay_min: Minimum seconds to wait between retries.
        retry_delay_max: Maximum seconds to wait between retries.

    Returns:
        The model response text, or None on total failure.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    last_error = None
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                ),
                timeout=300,  # 5 minute timeout per API call
            )
            return response.text
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = random.uniform(retry_delay_min, retry_delay_max)
                logger.warning(
                    f"External feedback generation failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f} seconds..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"External feedback generation failed after {max_retries} attempts: {last_error}"
                )
    return None


def build_feedback_prompt(
    problem: str,
    answer: str,
    summaries: list[str],
    template: str = "",
) -> str:
    """Format the feedback prompt template with problem, answer, and student summaries.

    Args:
        problem: The problem text.
        answer: The ground truth answer.
        summaries: List of student summary strings.
        template: Prompt template with {problem}, {answer}, {summaries} placeholders.
            Empty string uses DEFAULT_EXTERNAL_FEEDBACK_PROMPT_TEMPLATE.

    Returns:
        Formatted prompt string.
    """
    if not template:
        template = DEFAULT_EXTERNAL_FEEDBACK_PROMPT_TEMPLATE

    if summaries:
        summaries_text = "\n".join(
            [f"Student solution {i + 1}: {summary}" for i, summary in enumerate(summaries)]
        )
    else:
        summaries_text = "(No valid summaries available - students did not include summary tags)"

    return template.format(
        problem=problem,
        answer=answer,
        summaries=summaries_text,
    )


async def _generate_feedback_for_group(
    problem: str,
    answer: str,
    summaries: list[str],
    template: str,
    temperature: float,
    max_output_tokens: int,
    model: str,
    max_retries: int,
    retry_delay_min: float,
    retry_delay_max: float,
) -> Optional[str]:
    """Generate feedback for a single UID group."""
    prompt = build_feedback_prompt(problem, answer, summaries, template)
    raw = await get_external_feedback(
        prompt=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        model=model,
        max_retries=max_retries,
        retry_delay_min=retry_delay_min,
        retry_delay_max=retry_delay_max,
    )
    if raw is None:
        return None
    return extract_feedback_from_response(raw)


async def generate_external_feedback_for_groups(
    uid_to_indices: dict,
    response_texts: list[str],
    prompt_texts: list[str],
    ground_truths: list[str],
    template: str = "",
    temperature: float = 0.0,
    max_output_tokens: int = 14000,
    model: str = "gemini-3-flash-preview",
    max_retries: int = 3,
    retry_delay_min: float = 15.0,
    retry_delay_max: float = 45.0,
    filter_incomplete: bool = True,
    max_concurrent_requests: int = 600,
) -> dict:
    """Orchestrate concurrent feedback generation for all UID groups.

    For each UID group: extracts summaries from responses, builds a prompt,
    calls the Gemini API, and extracts the feedback. Concurrency is capped
    by a semaphore so at most max_concurrent_requests run in parallel.

    Args:
        uid_to_indices: Mapping from UID to list of sample indices in that group.
        response_texts: All response texts in the batch.
        prompt_texts: All prompt texts in the batch.
        ground_truths: Ground truth answer for each sample.
        template: Prompt template (empty => default).
        temperature: Sampling temperature for the feedback model.
        max_output_tokens: Max tokens for feedback generation.
        model: Gemini model identifier.
        max_retries: Max retry attempts per API call.
        retry_delay_min: Minimum seconds between retries.
        retry_delay_max: Maximum seconds between retries.
        filter_incomplete: Whether to filter out responses without </think>.
        max_concurrent_requests: Maximum number of Gemini API calls in flight at once.

    Returns:
        Dict mapping uid -> feedback_str or None.
    """
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _rate_limited_generate(uid, **kwargs):
        async with semaphore:
            return await _generate_feedback_for_group(**kwargs)

    tasks = {}
    for uid, indices in uid_to_indices.items():
        summaries = []
        for idx in indices:
            summary = extract_summary_from_response(response_texts[idx], filter_incomplete=filter_incomplete)
            if summary is not None:
                summaries.append(summary)

        problem = prompt_texts[indices[0]]
        answer = ground_truths[indices[0]] if indices[0] < len(ground_truths) else ""

        tasks[uid] = _rate_limited_generate(
            uid,
            problem=problem,
            answer=answer,
            summaries=summaries,
            template=template,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            model=model,
            max_retries=max_retries,
            retry_delay_min=retry_delay_min,
            retry_delay_max=retry_delay_max,
        )

    if not tasks:
        return {}

    uids = list(tasks.keys())
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    uid_to_feedback = {}
    for uid, result in zip(uids, results):
        if isinstance(result, Exception):
            logger.error(f"Feedback generation failed for uid={uid}: {result}")
            uid_to_feedback[uid] = None
        else:
            uid_to_feedback[uid] = result

    return uid_to_feedback
