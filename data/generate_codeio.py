"""
Generate a CodeIO dataset from reasoning_gym for minimal SDPO experimentation.

Uses the 'codeio' task (output prediction: given Python code + inputs, predict output).
Writes train.json and test.json in the JSONL format expected by data/preprocess.py.

Usage:
    python data/generate_codeio.py --output_dir datasets/codeio --seed 42
"""

import argparse
import json
from pathlib import Path

import reasoning_gym

# Sanity limits — answers/prompts beyond these thresholds are unreasonable for
# a model with 4096-token response budget and cause the ZSS tree-edit-distance
# scorer in reasoning_gym to hang (O(n^2 m^2) on large JSON trees).
MAX_ANSWER_CHARS = 2000
MAX_PROMPT_CHARS = 8000
MAX_ANSWER_JSON_ELEMENTS = 50  # max keys (dict) or items (list) in the answer

# "Easy" mode filters — keep only problems a small model can plausibly solve
EASY_MAX_CODE_LINES = 70       # max lines in the prompt (code + boilerplate)
EASY_SCALAR_ONLY = True        # only int/float/bool/None answers (no dict/list/long-str)


def _count_json_elements(answer_str: str) -> int:
    """Count the number of top-level elements in a JSON answer."""
    try:
        obj = json.loads(answer_str)
    except (json.JSONDecodeError, TypeError):
        return 0
    if isinstance(obj, dict):
        return sum(_count_json_elements(json.dumps(v)) if isinstance(v, (dict, list)) else 1 for v in obj.values())
    if isinstance(obj, list):
        return len(obj)
    return 1


def _is_sane(example: dict, easy: bool = False) -> tuple[bool, str]:
    """Check whether an example is suitable for training.

    Returns (ok, reason) where reason explains why it was filtered.
    """
    answer = example.get("answer", "")
    question = example.get("question", "")

    if len(answer) > MAX_ANSWER_CHARS:
        return False, f"answer too long ({len(answer)} chars)"
    if len(question) > MAX_PROMPT_CHARS:
        return False, f"prompt too long ({len(question)} chars)"
    if _count_json_elements(answer) > MAX_ANSWER_JSON_ELEMENTS:
        return False, f"answer has too many JSON elements ({_count_json_elements(answer)})"

    if easy:
        # Filter for short code
        if question.count("\n") > EASY_MAX_CODE_LINES:
            return False, f"code too long ({question.count(chr(10))} lines, max {EASY_MAX_CODE_LINES})"
        # Filter for scalar answers only
        if EASY_SCALAR_ONLY:
            try:
                parsed = json.loads(answer)
            except (json.JSONDecodeError, TypeError):
                return False, "answer not parseable as JSON"
            if not isinstance(parsed, (int, float, bool, type(None))):
                return False, f"non-scalar answer type ({type(parsed).__name__})"

    return True, ""


def generate(output_dir: str, seed: int = 42, train_size: int = 9600, test_size: int = 500, easy: bool = False) -> None:
    # Over-sample to compensate for filtering (more aggressive when easy=True)
    oversample_factor = 6.0 if easy else 1.3
    total_requested = int((train_size + test_size) * oversample_factor)

    # input_prediction_probability=1.0 → output prediction mode
    # (given code + inputs, predict the output JSON)
    mode_str = "EASY " if easy else ""
    print(f"Sampling {total_requested} codeio problems ({mode_str}output prediction, seed={seed})...")
    ds = reasoning_gym.create_dataset(
        "codeio",
        size=total_requested,
        seed=seed,
        input_prediction_probability=1.0,
    )

    examples = []
    filter_reasons: dict[str, int] = {}
    for i in range(len(ds)):
        try:
            ex = ds[i]
        except Exception:
            filter_reasons["generation_error"] = filter_reasons.get("generation_error", 0) + 1
            continue
        ok, reason = _is_sane(ex, easy=easy)
        if ok:
            examples.append(ex)
        else:
            filter_reasons[reason] = filter_reasons.get(reason, 0) + 1

    print(f"Generated {len(examples)} valid examples out of {total_requested}")
    if filter_reasons:
        print("Filtered out:")
        for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1]):
            print(f"  {count:>5d}  {reason}")

    needed = train_size + test_size
    if len(examples) < needed:
        print(f"Warning: only {len(examples)}/{needed} examples after filtering (need more oversampling)")

    # Shuffle to avoid systematic ordering from reasoning_gym
    import random as _rng
    _rng.seed(seed)
    _rng.shuffle(examples)
    train_examples = examples[:train_size]
    test_examples = examples[train_size : train_size + test_size]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split, split_examples in [("train", train_examples), ("test", test_examples)]:
        path = out / f"{split}.json"
        with open(path, "w") as f:
            for idx, ex in enumerate(split_examples):
                row = {
                    "idx": idx,
                    "kind": "codeio",
                    "dataset": "codeio",
                    # answer is the expected JSON string (becomes ground_truth in preprocess.py)
                    "answer": ex["answer"],
                    "elo": "-",
                    # prompt is the full question (code + inputs + instruction)
                    "prompt": ex["question"],
                    "description": "Predict the output of a Python function given its inputs",
                    "tests": "-",
                    "embedding": [],
                    "system": None,
                }
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(split_examples)} {split} examples -> {path}")

    # Print dataset stats
    for split, split_examples in [("train", train_examples), ("test", test_examples)]:
        answer_lens = [len(ex["answer"]) for ex in split_examples]
        prompt_lens = [len(ex["question"]) for ex in split_examples]
        print(f"\n{split} stats:")
        print(f"  answer length:  min={min(answer_lens)}, max={max(answer_lens)}, mean={sum(answer_lens)/len(answer_lens):.0f}")
        print(f"  prompt length:  min={min(prompt_lens)}, max={max(prompt_lens)}, mean={sum(prompt_lens)/len(prompt_lens):.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate codeio dataset from reasoning_gym")
    parser.add_argument("--output_dir", default="datasets/codeio", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_size", type=int, default=9600, help="Number of training examples")
    parser.add_argument("--test_size", type=int, default=500, help="Number of test examples")
    parser.add_argument("--easy", action="store_true", help="Filter for easy problems (short code, scalar answers)")
    args = parser.parse_args()

    generate(
        output_dir=args.output_dir,
        seed=args.seed,
        train_size=args.train_size,
        test_size=args.test_size,
        easy=args.easy,
    )
