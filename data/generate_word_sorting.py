"""
Generate a word_sorting dataset from reasoning_gym for SDPO experimentation.

Uses the 'word_sorting' task: given a list of words, sort them in ascending or
descending ASCII order. Output is a comma-separated list.

Usage:
    python data/generate_word_sorting.py --output_dir datasets/word_sorting --seed 42
"""

import argparse
import json
import random
from pathlib import Path

import reasoning_gym


def generate(output_dir: str, seed: int = 42, train_size: int = 4800, test_size: int = 500) -> None:
    total_requested = train_size + test_size + 200  # small buffer

    print(f"Sampling {total_requested} word_sorting problems (seed={seed})...")
    ds = reasoning_gym.create_dataset(
        "word_sorting",
        size=total_requested,
        seed=seed,
    )

    examples = []
    for i in range(len(ds)):
        try:
            ex = ds[i]
            examples.append(ex)
        except Exception:
            continue

    print(f"Generated {len(examples)} valid examples out of {total_requested}")

    rng = random.Random(seed)
    rng.shuffle(examples)
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
                    "kind": "word_sorting",
                    "dataset": "word_sorting",
                    "answer": ex["answer"],
                    "elo": "-",
                    "prompt": ex["question"],
                    "description": "Sort words in ascending or descending ASCII order",
                    "tests": "-",
                    "embedding": [],
                    "system": None,
                }
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(split_examples)} {split} examples -> {path}")

    for split, split_examples in [("train", train_examples), ("test", test_examples)]:
        answer_lens = [len(ex["answer"]) for ex in split_examples]
        prompt_lens = [len(ex["question"]) for ex in split_examples]
        print(f"\n{split} stats:")
        print(f"  answer length:  min={min(answer_lens)}, max={max(answer_lens)}, mean={sum(answer_lens)/len(answer_lens):.0f}")
        print(f"  prompt length:  min={min(prompt_lens)}, max={max(prompt_lens)}, mean={sum(prompt_lens)/len(prompt_lens):.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate word_sorting dataset from reasoning_gym")
    parser.add_argument("--output_dir", default="datasets/word_sorting", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_size", type=int, default=4800, help="Number of training examples")
    parser.add_argument("--test_size", type=int, default=500, help="Number of test examples")
    args = parser.parse_args()

    generate(
        output_dir=args.output_dir,
        seed=args.seed,
        train_size=args.train_size,
        test_size=args.test_size,
    )
