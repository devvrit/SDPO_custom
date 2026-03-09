# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import json
import os
import re
import subprocess
import sys
import threading
import time
from typing import Optional
from math_verify import parse as mv_parse, verify as mv_verify
from verl.utils.reward_score.prime_math.grader import math_equal

FORMAT_PENALTY = False

# ---------------------------------------------------------------------------
# Persistent subprocess for mario_is_equiv
# ---------------------------------------------------------------------------
# We run mario_is_equiv in a long-lived subprocess to avoid:
# 1. timeout_decorator overriding NaiveRewardManager's SIGALRM handler
# 2. CUDA context corruption from fork
# 3. Signal errors from threading
#
# The subprocess has its own signal space, so timeout_decorator works normally.
# Import cost (~2s for sympy) is paid once at first call, then each request
# is just JSON over stdin/stdout.
# ---------------------------------------------------------------------------

_MARIO_PER_CALL_TIMEOUT = 30  # seconds per is_equiv call

# The worker script for the persistent subprocess.
# Reads JSON lines from stdin, writes JSON lines to stdout.
# Each call has its own SIGALRM-based timeout as a safety net.
_MARIO_WORKER_SCRIPT = r'''
import json
import signal
import sys

# Silence stderr from sympy/timeout_decorator noise
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from math_evaluation import is_equiv

def _alarm_handler(signum, frame):
    raise TimeoutError("mario_is_equiv overall timeout")

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
        timeout = d.get("timeout", 30)
        # Set a hard per-call alarm as safety net
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)
        try:
            result = bool(is_equiv(d["answer"], d["pred"]))
        except TimeoutError:
            result = False
        except Exception:
            result = False
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        print(json.dumps({"ok": True, "result": result}), flush=True)
    except Exception as e:
        print(json.dumps({"ok": False, "result": False}), flush=True)
'''

_mario_proc = None
_mario_lock = threading.Lock()


def _get_mario_proc():
    """Get or start the persistent mario_is_equiv subprocess."""
    global _mario_proc
    if _mario_proc is not None and _mario_proc.poll() is None:
        return _mario_proc
    # Start a new subprocess
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    _mario_proc = subprocess.Popen(
        [sys.executable, "-c", _MARIO_WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    return _mario_proc


def _kill_mario_proc():
    """Kill the persistent subprocess (e.g. after a hung call)."""
    global _mario_proc
    if _mario_proc is not None:
        try:
            _mario_proc.kill()
            _mario_proc.wait(timeout=2)
        except Exception:
            pass
        _mario_proc = None


def _run_mario_is_equiv_subprocess(answer: str, pred: str) -> bool:
    """Call mario_is_equiv via the persistent subprocess."""
    with _mario_lock:
        try:
            proc = _get_mario_proc()
            payload = json.dumps({
                "answer": answer,
                "pred": pred,
                "timeout": _MARIO_PER_CALL_TIMEOUT,
            }) + "\n"
            proc.stdin.write(payload)
            proc.stdin.flush()

            # Read response with a wall-clock timeout via a thread
            response = [None]
            def _read():
                try:
                    response[0] = proc.stdout.readline()
                except Exception:
                    pass

            reader = threading.Thread(target=_read, daemon=True)
            reader.start()
            reader.join(timeout=_MARIO_PER_CALL_TIMEOUT + 5)

            if reader.is_alive() or not response[0]:
                # Subprocess hung — kill it so next call starts a fresh one
                print(f"[verify] mario_is_equiv subprocess hung, killing", flush=True)
                _kill_mario_proc()
                return False

            data = json.loads(response[0])
            return bool(data.get("result", False))
        except Exception:
            _kill_mario_proc()
            return False


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else ""#None


def remove_boxed(s: str) -> str:
    r"""Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = r"\boxed{"
    #assert s[: len(left)] == left, f"box error: {s}"
    #assert s[-1] == "}", f"box error: {s}"
    if s[: len(left)] == left and  s[-1] == "}":
        return s[len(left) : -1]
    else:
        return ""


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return extracted_pred == gt, extracted_pred


def verify(
    solution_str: str, answer: str, pause_tokens_index: Optional[list[int]] = None
) -> bool:
    """Verify if the solution is correct.

    mv_verify and math_equal are called directly in the main thread.
    math_equal uses multiprocessing (fork) internally which needs the main thread.

    mario_is_equiv runs in a daemon thread with signal module patched, because
    its timeout_decorator overrides the process-level SIGALRM handler.

    Returns:
        Tuple of (correct, extracted_prediction)
    """
    correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
    if pred is None:
        pred = ""

    if correct or pred == "":
        return correct, pred

    # try Math-Verify equivalence check (direct call, fast)
    if not correct:
        t0 = time.monotonic()
        try:
            gold_expr = mv_parse(answer)
            pred_expr = mv_parse(pred)
            correct = mv_verify(gold_expr, pred_expr)
        except Exception:
            pass
        elapsed = time.monotonic() - t0
        if elapsed > 3:
            print(f"[verify] mv_verify took {elapsed:.1f}s (pred={pred[:60]!r})", flush=True)

    # try sympy-based math_equal (direct call, has internal multiprocessing timeout)
    if not correct:
        t0 = time.monotonic()
        try:
            correct = math_equal(pred, answer, timeout=10.0)
        except Exception:
            pass
        elapsed = time.monotonic() - t0
        if elapsed > 15:
            print(f"[verify] math_equal took {elapsed:.1f}s (pred={pred[:60]!r})", flush=True)

    # mario_is_equiv via subprocess — completely isolated process with its own
    # signal space, so timeout_decorator works normally and can't interfere
    # with NaiveRewardManager's SIGALRM.
    if not correct:
        t0 = time.monotonic()
        try:
            correct = _run_mario_is_equiv_subprocess(answer, pred)
        except Exception:
            pass
        elapsed = time.monotonic() - t0
        if elapsed > 10:
            print(f"[verify] mario_is_equiv subprocess took {elapsed:.1f}s (pred={pred[:60]!r})", flush=True)

    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info = None,
    pause_tokens_index: Optional[list[int]] = None,
    format_feedback: bool = True,
    correctness_feedback: bool = True,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, 0 for incorrect)
    """
    split = extra_info.get("split", "test")
    was_truncated = extra_info.get("truncated", False)

    # Verify the solution
    correct, pred = verify(solution_str, ground_truth, pause_tokens_index)

    reward = 1.0 if correct else 0.0
    score = reward
    incorrect_format = pred is None or pred == ""
    was_truncated = extra_info.get("truncated", False)
    if FORMAT_PENALTY and split == "train" and incorrect_format and (not was_truncated):
        score -= 0.5

    # Generate explicit feedback for format errors (analogous to code feedback)
    feedback = ""
    if incorrect_format and not was_truncated and format_feedback:
        feedback = "Your answer had the wrong format. The solution must be given in the format: \\boxed{your_answer}."
    elif was_truncated and format_feedback:
        feedback = "Your response was truncated because it exceeded the maximum length."
    elif not correct and correctness_feedback:
        feedback = f"Your answer is incorrect. The correct answer is {ground_truth}."

    return {
        "score": score,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 1 if incorrect_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": feedback,
    }
