"""
Hallucination detection for student trajectories.

Detects if the student model hallucinates having access to privileged
feedback/information that only the teacher receives. The teacher prompt
contains error feedback (e.g. "Actions mismatch: predicted [], expected [...]")
while the student prompt does NOT. If the student's response references such
feedback, that's a hallucination.

Usage:
    python evaluate.py --traj_dir <path> [--steps 1,10,50,100] [--max_samples 50] [--output results.csv]
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

SYSTEM_PROMPT = """\
You are an expert evaluator detecting hallucinations in AI model responses.

CONTEXT: We are training a student model via distillation from a teacher model. \
The teacher receives privileged feedback (e.g., error messages like "Actions mismatch: \
predicted [...], expected [...]", or the correct answer/action sequence). \
The student does NOT receive any such feedback — it only sees the original user question \
and tool descriptions.

YOUR TASK: Determine whether the student's response hallucinates having access to \
privileged information it was never given.

Hallucination indicators (student references things NOT in its prompt):
- Mentioning "mismatch", "error", "the system said", "feedback", "correction"
- Referring to expected vs predicted actions as if it saw evaluation output
- Claiming a previous attempt failed when no previous attempt is shown in the student prompt
- Knowing the correct answer/action without reasoning, as if it read privileged feedback
- Referencing "the example answer", "the initial answer", "the first response" when none exists in the student prompt

NOT hallucination:
- Normal chain-of-thought reasoning about the task
- Making mistakes or wrong tool calls (that's just being wrong, not hallucinating)
- Correctly solving the task through genuine reasoning

Respond with JSON:
{
    "hallucination": true/false,
    "explanation": "brief reason (1-2 sentences)"
}"""

USER_TEMPLATE = """\
## Student Prompt (what the student actually sees):
{prompt}

## Student Response:
{response}"""


async def evaluate_sample(client: AsyncOpenAI, record: dict, semaphore: asyncio.Semaphore, model: str) -> dict:
    """Evaluate a single trajectory record for hallucination."""
    async with semaphore:
        user_msg = USER_TEMPLATE.format(
            prompt=record["prompt"][:6000],  # truncate to stay within context
            response=record["response"][:4000],
        )
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=200,
            )
            result = json.loads(resp.choices[0].message.content)
        except Exception as e:
            result = {"hallucination": None, "explanation": f"API error: {e}"}

        return {
            "step": record["step"],
            "sample_idx": record["sample_idx"],
            "uid": record.get("uid", ""),
            "hallucination": result.get("hallucination"),
            "answer_correct": record["acc"] == 1.0,
            "has_privileged_info": record.get("has_privileged_info", False),
            "explanation": result.get("explanation", ""),
        }


async def evaluate_step_file(client: AsyncOpenAI, filepath: str, max_samples: int, semaphore: asyncio.Semaphore, model: str) -> list:
    """Evaluate all records in a single step JSONL file."""
    records = []
    with open(filepath) as f:
        for line in f:
            records.append(json.loads(line))

    if max_samples > 0 and len(records) > max_samples:
        # Sample evenly
        import random
        random.seed(42)
        records = random.sample(records, max_samples)

    tasks = [evaluate_sample(client, r, semaphore, model) for r in records]
    results = await asyncio.gather(*tasks)
    return list(results)


async def main():
    parser = argparse.ArgumentParser(description="Hallucination evaluation on student trajectories")
    parser.add_argument("--traj_dir", required=True, help="Path to trajectories/ directory")
    parser.add_argument("--steps", default=None, help="Comma-separated step numbers to evaluate (default: all)")
    parser.add_argument("--step_stride", type=int, default=1, help="Evaluate every N-th step")
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples per step (0=all)")
    parser.add_argument("--output", default="hallucination_results.csv", help="Output CSV path")
    parser.add_argument("--concurrency", type=int, default=30, help="Max concurrent API calls")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)
    if not traj_dir.exists():
        print(f"Error: {traj_dir} does not exist")
        sys.exit(1)

    # Discover step files
    step_files = sorted(traj_dir.glob("step_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"Found {len(step_files)} step files in {traj_dir}")

    if args.steps:
        requested = set(int(s) for s in args.steps.split(","))
        step_files = [f for f in step_files if int(f.stem.split("_")[1]) in requested]
    elif args.step_stride > 1:
        step_files = step_files[::args.step_stride]

    print(f"Evaluating {len(step_files)} steps")

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)

    all_results = []
    for i, sf in enumerate(step_files):
        step_num = int(sf.stem.split("_")[1])
        print(f"[{i+1}/{len(step_files)}] Evaluating step {step_num}...")
        results = await evaluate_step_file(client, str(sf), args.max_samples, semaphore, args.model)
        all_results.extend(results)

        # Print running stats
        halluc_count = sum(1 for r in results if r["hallucination"] is True)
        total = len(results)
        print(f"  -> {halluc_count}/{total} hallucinations ({100*halluc_count/total:.1f}%)")

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "sample_idx", "uid", "hallucination", "answer_correct", "has_privileged_info", "explanation"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults written to {output_path}")
    print(f"Total: {len(all_results)} samples evaluated")

    # Quick summary
    total_halluc = sum(1 for r in all_results if r["hallucination"] is True)
    total_correct = sum(1 for r in all_results if r["answer_correct"])
    print(f"Overall hallucination rate: {total_halluc}/{len(all_results)} ({100*total_halluc/len(all_results):.1f}%)")
    print(f"Overall accuracy: {total_correct}/{len(all_results)} ({100*total_correct/len(all_results):.1f}%)")

    # Hallucination rate among correct vs incorrect
    correct_halluc = sum(1 for r in all_results if r["answer_correct"] and r["hallucination"] is True)
    incorrect_halluc = sum(1 for r in all_results if not r["answer_correct"] and r["hallucination"] is True)
    n_correct = max(total_correct, 1)
    n_incorrect = max(len(all_results) - total_correct, 1)
    print(f"Hallucination rate (correct answers): {correct_halluc}/{n_correct} ({100*correct_halluc/n_correct:.1f}%)")
    print(f"Hallucination rate (incorrect answers): {incorrect_halluc}/{n_incorrect} ({100*incorrect_halluc/n_incorrect:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
