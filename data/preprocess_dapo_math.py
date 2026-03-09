"""
Preprocess the open-r1/DAPO-Math-17k-Processed (en) dataset to parquet format
compatible with the SDPO training pipeline.

Usage:
    python data/preprocess_dapo_math.py --save_dir datasets/dapo_math_17k --eval_size 500
"""

import argparse
import os

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="datasets/dapo_math_17k")
    parser.add_argument("--eval_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading open-r1/DAPO-Math-17k-Processed (en subset)...")
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    print(f"Loaded {len(ds)} examples")

    def process_fn(example, idx):
        prompt_text = example["prompt"]
        solution = example["solution"]

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "data_source": "dapo_math",
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": "train", "index": idx},
        }

    ds = ds.map(process_fn, with_indices=True, remove_columns=ds.column_names)

    # Split into train and eval
    split = ds.train_test_split(test_size=args.eval_size, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    # Update split field in eval
    def set_eval_split(example):
        example["extra_info"] = {**example["extra_info"], "split": "test"}
        return example

    eval_ds = eval_ds.map(set_eval_split)

    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    os.makedirs(args.save_dir, exist_ok=True)
    train_ds.to_parquet(os.path.join(args.save_dir, "train.parquet"))
    eval_ds.to_parquet(os.path.join(args.save_dir, "test.parquet"))
    print(f"Saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
