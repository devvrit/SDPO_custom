import re
import json
from collections import Counter
from typing import Optional


def extract_actions(text: str) -> list[str]:
    """Extract all action names after 'Action:' occurrences."""
    actions = re.findall(r'Action:\s*(\w+)', text)
    return actions


def extract_action_inputs(text: str) -> dict:
    """Extract and merge all JSON blocks following 'Action Input:'.

    Uses brace-depth counting to handle nested JSON objects correctly.
    """
    combined_dict = {}
    for m in re.finditer(r'Action Input:\s*', text):
        start = m.end()
        json_str = _extract_braced_block(text, start)
        if json_str is not None:
            try:
                parsed = json.loads(json_str)
                combined_dict.update(parsed)
            except json.JSONDecodeError:
                pass
    return combined_dict


def _extract_braced_block(text: str, start: int) -> Optional[str]:
    """Extract a balanced {...} block starting at *start* in *text*."""
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def merge_action_inputs(action_inputs_list: list[dict]) -> dict:
    """Merge a list of action input dicts into a single dict."""
    combined = {}
    for d in action_inputs_list:
        if d:
            combined.update(d)
    return combined


def is_correct_format(text: str) -> bool:
    """Check if the text contains the expected Action/Action Input format."""
    pattern = re.compile(r"Action:.*?\nAction Input:.*?", re.DOTALL)
    return pattern.search(text) is not None


def _parse_tool_descriptions(prompt: str) -> dict[str, str]:
    """Parse tool name -> description mapping from the prompt text.

    Looks for individual tool documentation lines like:
        toolName: Does something useful.
        Parameters: ...
    """
    tool_descs: dict[str, str] = {}

    # Skip known header/section keywords
    _skip = {"name", "description", "parameters", "output", "format",
             "structure", "documentation", "use", "begin", "question",
             "thought", "action", "required", "object"}

    # Match lines like "toolName: Description text here.\nParameters: ..."
    # Using (.+) without DOTALL ensures we stay on one line.
    method_blocks = re.findall(
        r"^(\w+): (.+)\nParameters:",
        prompt, re.MULTILINE,
    )
    for name, desc in method_blocks:
        if name.lower() in _skip:
            continue
        tool_descs[name] = desc.strip()

    return tool_descs


def _desc_snippet(name: str, tool_descs: dict[str, str]) -> str:
    """Return ' — <description>' if available, else ''."""
    desc = tool_descs.get(name, "")
    if not desc:
        return ""
    # Strip trailing period to avoid double-period
    return f" — {desc.rstrip('.')}"


def _build_action_feedback(
    pred_actions: list[str],
    gt_actions: list[str],
    tool_descs: dict[str, str],
) -> list[str]:
    """Build feedback lines explaining action (tool name) mismatches."""
    lines = []

    pred_counter = Counter(pred_actions)
    gt_counter = Counter(gt_actions)

    if pred_counter == gt_counter:
        return lines

    # Tools that were called but shouldn't have been
    extra_actions = pred_counter - gt_counter
    # Tools that should have been called but weren't
    missing_actions = gt_counter - pred_counter

    if not pred_actions:
        lines.append("You did not call any tool.")
    elif len(extra_actions) > 0 and len(missing_actions) > 0:
        # Substitution case: called wrong tool(s) instead of the right one(s)
        for wrong_name in extra_actions:
            lines.append(
                f"You called '{wrong_name}'{_desc_snippet(wrong_name, tool_descs)} instead."
            )
    elif extra_actions:
        for action, count in extra_actions.items():
            times = f" ({count} times)" if count > 1 else ""
            lines.append(
                f"'{action}'{_desc_snippet(action, tool_descs)} was not needed{times}."
            )
    elif missing_actions:
        for action, count in missing_actions.items():
            times = f" ({count} times)" if count > 1 else ""
            lines.append(
                f"You also needed to call '{action}'{_desc_snippet(action, tool_descs)}{times}."
            )

    return lines


def _build_input_feedback(
    pred_inputs: dict,
    gt_inputs: dict,
) -> list[str]:
    """Build feedback lines explaining specific parameter mismatches."""
    lines = []

    if pred_inputs == gt_inputs:
        return lines

    pred_keys = set(pred_inputs.keys())
    gt_keys = set(gt_inputs.keys())

    missing_keys = gt_keys - pred_keys
    extra_keys = pred_keys - gt_keys
    common_keys = pred_keys & gt_keys

    if missing_keys:
        lines.append(
            f"Missing required parameter(s): {', '.join(sorted(missing_keys))}."
        )
        for k in sorted(missing_keys):
            gt_val = gt_inputs[k]
            lines.append(f"  - '{k}' should be: {json.dumps(gt_val)}")

    if extra_keys:
        lines.append(
            f"Unexpected parameter(s) provided: {', '.join(sorted(extra_keys))}."
        )

    wrong_values = {}
    for k in sorted(common_keys):
        if pred_inputs[k] != gt_inputs[k]:
            wrong_values[k] = (pred_inputs[k], gt_inputs[k])

    if wrong_values:
        lines.append("Incorrect parameter value(s):")
        for k, (pred_v, gt_v) in wrong_values.items():
            lines.append(
                f"  - '{k}': you provided {json.dumps(pred_v)}, "
                f"but the correct value is {json.dumps(gt_v)}"
            )

    return lines


def _build_format_feedback(solution: str, correct_format: bool) -> list[str]:
    """Build feedback for format issues."""
    if correct_format:
        return []

    lines = []
    has_action = re.search(r'Action:', solution)
    has_input = re.search(r'Action Input:', solution)

    if not has_action and not has_input:
        lines.append(
            "Your response does not use the required tool-call format. "
            "You must use:\n"
            "Action: <tool_name>\n"
            "Action Input: <json_parameters>"
        )
    elif not has_action:
        lines.append(
            "Missing 'Action:' line. Specify which tool to call with 'Action: <tool_name>'."
        )
    elif not has_input:
        lines.append(
            "Missing 'Action Input:' line. Provide tool parameters as JSON with "
            "'Action Input: {\"param\": \"value\"}'."
        )
    return lines


def compute_score(
    solution: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> dict:
    """
    Compute score for tooluse task.

    Args:
        solution: The model's response text
        ground_truth: JSON string containing list of dicts with 'Action' and 'Action_Input' keys
                      e.g., '[{"Action": "search", "Action_Input": "{\"query\": \"test\"}"}]'
        extra_info: Optional dict; if it contains 'description' or 'problem', the prompt
                    text is used to extract tool descriptions for richer feedback.

    Returns:
        dict with score, acc, pred, incorrect_format, feedback
    """
    # Parse ground truth
    try:
        gt_list = json.loads(ground_truth)
    except json.JSONDecodeError:
        # If ground_truth is already a list (passed directly), handle that case
        if isinstance(ground_truth, list):
            gt_list = ground_truth
        else:
            return {
                "score": 0.0,
                "acc": 0.0,
                "pred": "",
                "incorrect_format": 1,
                "feedback": "Failed to parse ground truth JSON",
            }

    # Extract ground truth actions and action inputs
    gt_actions = [item['Action'] for item in gt_list]
    gt_action_inputs_list = []
    for item in gt_list:
        try:
            parsed_input = json.loads(item['Action_Input']) if isinstance(item['Action_Input'], str) else item['Action_Input']
            gt_action_inputs_list.append(parsed_input)
        except (json.JSONDecodeError, KeyError):
            gt_action_inputs_list.append({})
    gt_action_inputs = merge_action_inputs(gt_action_inputs_list)

    # Extract predicted actions and action inputs from solution
    pred_actions = extract_actions(solution)
    pred_action_inputs = extract_action_inputs(solution)

    # Check correctness
    actions_correct = Counter(pred_actions) == Counter(gt_actions)
    action_inputs_correct = pred_action_inputs == gt_action_inputs

    # Both must be correct for full score
    is_correct = actions_correct and action_inputs_correct
    reward = 1.0 if is_correct else 0.0

    # Check format
    correct_format = is_correct_format(solution)

    # Build prediction string for logging
    prediction = f"Actions: {pred_actions}, Inputs: {pred_action_inputs}"

    if is_correct:
        return {
            "score": reward,
            "acc": reward,
            "pred": prediction,
            "incorrect_format": 0 if correct_format else 1,
            "feedback": "",
        }

    # --- Build rich feedback ---
    # Parse tool descriptions from the prompt if available
    prompt_text = ""
    if extra_info:
        prompt_text = extra_info.get("description", "") or extra_info.get("problem", "") or ""
    tool_descs = _parse_tool_descriptions(prompt_text) if prompt_text else {}

    feedback_lines = []

    # Lead with what the task required (what the correct tool does)
    if len(gt_actions) == 1:
        desc = tool_descs.get(gt_actions[0], "")
        if desc:
            feedback_lines.append(
                f"The task required calling '{gt_actions[0]}' to: {desc.rstrip('.')}."
            )
    else:
        task_parts = []
        for a in gt_actions:
            desc = tool_descs.get(a, "")
            task_parts.append(f"'{a}' ({desc.rstrip('.')})" if desc else f"'{a}'")
        feedback_lines.append(
            f"The task required calling: {', '.join(task_parts)}."
        )

    # Format feedback
    feedback_lines.extend(_build_format_feedback(solution, correct_format))

    # Action (tool name) feedback
    feedback_lines.extend(_build_action_feedback(pred_actions, gt_actions, tool_descs))

    # Parameter feedback (always give detail when there are params to compare)
    if not action_inputs_correct:
        feedback_lines.extend(_build_input_feedback(pred_action_inputs, gt_action_inputs))

    feedback = "\n".join(feedback_lines)

    return {
        "score": reward,
        "acc": reward,
        "pred": prediction,
        "incorrect_format": 0 if correct_format else 1,
        "feedback": feedback,
    }
