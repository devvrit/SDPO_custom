"""
Reward function for the word_sorting task (from reasoning_gym).

The model is given a list of words and must sort them in ascending or descending
ASCII/Unicode order. Output is a comma-separated list. Scoring uses reasoning_gym's
built-in position-based partial credit (correct_positions / total_words), with
detailed feedback for SDPO reprompting.
"""

import re
from typing import Optional


def _extract_answer(text: str) -> Optional[str]:
    """Extract the comma-separated word list from the model's response.

    Tries:
    1. Last ```...``` fenced code block
    2. Last line that looks like a comma-separated word list
    """
    # 1. Fenced code blocks
    fence_matches = re.findall(r"```(?:\w*)?\s*([\s\S]*?)```", text)
    for candidate in reversed(fence_matches):
        candidate = candidate.strip()
        if "," in candidate:
            return candidate

    # 2. Scan lines from the end for comma-separated words
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        # A valid answer has at least one comma and only word-like tokens
        if "," in line and all(part.strip().replace("_", "").replace("-", "").isalnum() for part in line.split(",") if part.strip()):
            return line

    return None


def _build_feedback(pred_str: Optional[str], ground_truth: str) -> str:
    """Generate detailed feedback comparing predicted vs expected word ordering."""
    if pred_str is None:
        return (
            "No sorted word list found in your response. "
            "Provide your answer as a comma-separated list, e.g.: word_1, word_2, word_3"
        )

    expected_words = [w.strip() for w in ground_truth.split(",")]
    pred_words = [w.strip() for w in pred_str.split(",")]

    if pred_words == expected_words:
        return ""

    lines = ["Your answer is incorrect:"]

    expected_set = set(expected_words)
    pred_set = set(pred_words)
    missing = expected_set - pred_set
    extra = pred_set - expected_set

    if missing:
        lines.append(f"  - Missing words: {', '.join(sorted(missing))}")
    if extra:
        lines.append(f"  - Extra words not in the original list: {', '.join(sorted(extra))}")

    # Position feedback for words that are present but misplaced
    expected_pos = {w: i for i, w in enumerate(expected_words)}
    for i, word in enumerate(pred_words):
        if word in expected_pos and expected_pos[word] != i:
            lines.append(
                f"  - '{word}' should be at position {expected_pos[word] + 1}, not {i + 1}"
            )

    if len(lines) == 1:
        lines.append(f"  Expected: {ground_truth}")

    return "\n".join(lines)


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> dict:
    """
    Compute reward for a word_sorting prediction.

    Args:
        solution_str: The model's full response.
        ground_truth: The expected comma-separated sorted word list.
        extra_info: Optional dict with metadata.

    Returns:
        dict with keys: score, acc, pred, feedback, incorrect_format
    """
    pred_str = _extract_answer(solution_str)
    incorrect_format = 1 if pred_str is None else 0

    if pred_str is None:
        score = 0.0
    else:
        oracle_words = [w.strip() for w in ground_truth.split(",")]
        pred_words = [w.strip() for w in pred_str.split(",")]

        if pred_words == oracle_words:
            score = 1.0
        else:
            correct_positions = sum(
                1 for i, word in enumerate(pred_words)
                if i < len(oracle_words) and word == oracle_words[i]
            )
            score = correct_positions / len(oracle_words) if oracle_words else 0.0
            if sorted(pred_words) == sorted(oracle_words):
                score = max(score, 0.2)

    acc = 1.0 if score >= 1.0 else 0.0
    feedback = _build_feedback(pred_str, ground_truth) if score < 1.0 else ""

    return {
        "score": score,
        "acc": acc,
        "pred": pred_str or "",
        "feedback": feedback,
        "incorrect_format": incorrect_format,
    }
