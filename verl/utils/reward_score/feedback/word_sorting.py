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


def _explain_ascii_comparison(word_a: str, word_b: str) -> str:
    """Explain why word_a comes before word_b in ASCII ordering.

    Returns a human-readable explanation of the character-by-character comparison.
    """
    for i, (ca, cb) in enumerate(zip(word_a, word_b)):
        if ca != cb:
            oa, ob = ord(ca), ord(cb)
            # Build a readable reason for why ca < cb
            reason_parts = []
            if ca.isupper() and cb.islower():
                reason_parts.append(
                    f"uppercase letters come before lowercase in ASCII"
                )
            elif ca.islower() and cb.isupper():
                reason_parts.append(
                    f"lowercase letters come after uppercase in ASCII"
                )
            elif ca.isdigit() and cb.isalpha():
                reason_parts.append(
                    f"digits (0-9) come before all letters in ASCII"
                )
            elif ca.isalpha() and cb.isdigit():
                reason_parts.append(
                    f"letters come after digits (0-9) in ASCII"
                )
            reason_parts.append(f"'{ca}' has ASCII value {oa} vs '{cb}' has ASCII value {ob}")
            reason = "; ".join(reason_parts)
            pos_label = f"position {i + 1}" if i > 0 else "the first character"
            return (
                f"'{word_a}' < '{word_b}' because at {pos_label}, {reason}"
            )
    # One is a prefix of the other
    if len(word_a) < len(word_b):
        return (
            f"'{word_a}' < '{word_b}' because '{word_a}' is a prefix of "
            f"'{word_b}' (shorter strings come first)"
        )
    elif len(word_a) > len(word_b):
        return (
            f"'{word_b}' < '{word_a}' because '{word_b}' is a prefix of "
            f"'{word_a}' (shorter strings come first)"
        )
    return f"'{word_a}' == '{word_b}'"


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

    # Detect sort direction from expected order
    descending = len(expected_words) >= 2 and expected_words[0] > expected_words[-1]
    direction = "descending" if descending else "ascending"

    lines = [f"Your answer is incorrect. The correct {direction} ASCII/Unicode order is: {ground_truth}"]

    expected_set = set(expected_words)
    pred_set = set(pred_words)
    missing = expected_set - pred_set
    extra = pred_set - expected_set

    if missing:
        lines.append(f"  - Missing words: {', '.join(sorted(missing))}")
    if extra:
        lines.append(f"  - Extra words not in the original list: {', '.join(sorted(extra))}")

    # Find the first pair that the model got wrong and explain why
    # Compare adjacent pairs in the prediction to find misordered ones
    misordered_explanations = []
    for i in range(len(pred_words) - 1):
        a, b = pred_words[i], pred_words[i + 1]
        if descending:
            if a < b:  # wrong: in descending order, each word should be >= the next
                explanation = _explain_ascii_comparison(b, a)
                misordered_explanations.append(
                    f"  - You placed '{a}' before '{b}', but {explanation}, "
                    f"so '{b}' should come first in descending order"
                )
        else:
            if a > b:  # wrong: in ascending order, each word should be <= the next
                explanation = _explain_ascii_comparison(b, a)
                misordered_explanations.append(
                    f"  - You placed '{a}' before '{b}', but {explanation}, "
                    f"so '{b}' should come first in ascending order"
                )

    if misordered_explanations:
        lines.append("Misordered pairs in your answer:")
        # Show up to 3 misordered pairs to keep feedback concise
        lines.extend(misordered_explanations[:3])

    # Add a general ASCII reminder based on what the model seems confused about
    has_mixed_case = any(w[0].isupper() for w in expected_words) and any(
        w[0].islower() for w in expected_words
    )
    if has_mixed_case:
        lines.append(
            "Remember: In ASCII, all uppercase letters (A-Z, values 65-90) "
            "come before all lowercase letters (a-z, values 97-122). "
            "So 'Z' < 'a'."
        )

    has_digits = any(any(c.isdigit() for c in w) for w in expected_words)
    if has_digits:
        lines.append(
            "Remember: In ASCII, digits (0-9, values 48-57) come before "
            "all letters. So '9' < 'A' < 'a'."
        )

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
