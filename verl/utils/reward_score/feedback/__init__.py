from verl.utils.reward_score.feedback import math
from verl.utils.reward_score.feedback import code
from verl.utils.reward_score.feedback import gpqa
from verl.utils.reward_score.feedback import mcq
from verl.utils.reward_score.feedback import tooluse
from verl.utils.reward_score.feedback import codeio
from verl.utils.reward_score.feedback import word_sorting


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    if data_source in ["code", "livecodebench", "humanevalplus"]:
        results = code.compute_score(solution_str, ground_truth, extra_info, sparse_rewards=True, max_test_cases=None)
    elif data_source in ["math", "math500", "dapo_math", "gsm8k", "Polaris-Dataset-53K", "aime24", "aime25", "hmmt25", "amo"]:
        results = math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["gpqa"]:
        results = gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["sciknoweval"]:
        results = mcq.compute_score(solution_str, ground_truth)
    elif data_source in ["tooluse"]:
        results = tooluse.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["codeio"]:
        results = codeio.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["word_sorting"]:
        results = word_sorting.compute_score(solution_str, ground_truth, extra_info)
    else:
        raise ValueError(f"Reward style {data_source} not found.")
    return results
