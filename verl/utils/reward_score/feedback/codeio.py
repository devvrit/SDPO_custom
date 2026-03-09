"""
Reward function for the CodeIO task (from reasoning_gym).

The model is given a Python function and a set of inputs, and must predict
the output as a JSON value. Scoring is delegated to reasoning_gym's built-in
verifier (JSON tree edit distance), with an additional field-level feedback
message generated for SDPO reprompting.
"""

import json
import re
from typing import Optional

from reasoning_gym import get_score_answer_fn

# Cache the scoring function at module load time (avoids repeated registry lookups)
_score_fn = get_score_answer_fn("codeio")

# Safety limit: skip tree-edit-distance scoring when inputs are too large
# (the ZSS algorithm is O(n^2 m^2) and hangs on huge JSON trees)
_MAX_SCORE_INPUT_CHARS = 5000


def _extract_json(text: str) -> Optional[str]:
    """
    Extract the last valid JSON value from text.

    Tries, in order:
    1. Last ```json ... ``` fenced code block
    2. Last { ... } block (handles one level of nesting)
    3. Last [ ... ] block
    4. Last non-empty line that parses as JSON (for primitives: true/false/null/numbers)
    """
    # 1. Fenced code blocks
    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    for candidate in reversed(fence_matches):
        candidate = candidate.strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    # 2. Last { ... } block — greedy search to handle nested braces
    brace_start = text.rfind("{")
    if brace_start != -1:
        # Walk forward to find the matching closing brace
        depth = 0
        in_string = False
        escape = False
        for i in range(brace_start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break  # malformed brace block, fall through

    # 3. Last [ ... ] block
    bracket_start = text.rfind("[")
    if bracket_start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(bracket_start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[bracket_start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break

    # 4. Primitives — scan lines from the end
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
            return line
        except Exception:
            pass

    return None


def _build_feedback(pred_str: Optional[str], ground_truth: str) -> str:
    """Generate a human-readable comparison between the predicted and expected JSON."""
    if pred_str is None:
        return (
            f"No JSON found in your response. "
            f"Provide the answer as a JSON value, for example: {ground_truth}"
        )

    try:
        pred = json.loads(pred_str)
        expected = json.loads(ground_truth)
    except Exception:
        return f"Your answer could not be parsed as JSON. Expected: {ground_truth}"

    if pred == expected:
        return ""  # Correct — no feedback needed

    # Type mismatch
    if type(pred) != type(expected):
        return (
            f"Your answer has the wrong type: expected {type(expected).__name__}, "
            f"got {type(pred).__name__}. Expected: {ground_truth}"
        )

    # Dict comparison — field-level feedback
    if isinstance(expected, dict) and isinstance(pred, dict):
        lines = [f"Your answer {pred_str} is incorrect:"]
        missing = set(expected) - set(pred)
        extra = set(pred) - set(expected)
        wrong = {k for k in expected if k in pred and pred[k] != expected[k]}
        for k in sorted(missing):
            lines.append(f"  - missing key '{k}' (expected value: {json.dumps(expected[k])})")
        for k in sorted(extra):
            lines.append(f"  - unexpected key '{k}' (value: {json.dumps(pred[k])})")
        for k in sorted(wrong):
            lines.append(f"  - '{k}': expected {json.dumps(expected[k])}, got {json.dumps(pred[k])}")
        return "\n".join(lines)

    # Scalar / list mismatch — just show the diff
    return f"Your answer {pred_str} is incorrect. Expected: {ground_truth}"


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> dict:
    """
    Compute reward for a codeio prediction.

    Args:
        solution_str: The model's full response (may contain <think>...</think> + JSON).
        ground_truth: The expected JSON string (from reward_model.ground_truth).
        extra_info: Optional dict with 'split', 'truncated', etc.

    Returns:
        dict with keys: score, acc, pred, feedback, incorrect_format
    """
    # Extract the model's JSON answer from its response
    pred_str = _extract_json(solution_str)
    incorrect_format = 1 if pred_str is None else 0

    if pred_str is None:
        score = 0.0
    elif len(pred_str) > _MAX_SCORE_INPUT_CHARS or len(ground_truth) > _MAX_SCORE_INPUT_CHARS:
        # Skip expensive tree-edit-distance scoring for huge inputs.
        # Fall back to exact-match only (score 0 if not equal).
        try:
            score = 1.0 if json.loads(pred_str) == json.loads(ground_truth) else 0.0
        except (json.JSONDecodeError, TypeError):
            score = 0.0
    else:
        # Use reasoning_gym's built-in verifier (JSON tree edit distance)
        score = _score_fn(pred_str, {"answer": ground_truth})

    acc = 1.0 if score >= 1.0 else 0.0
    feedback = _build_feedback(pred_str, ground_truth) if score < 1.0 else ""

    return {
        "score": score,
        "acc": acc,
        "pred": pred_str or "",
        "feedback": feedback,
        "incorrect_format": incorrect_format,
    }
