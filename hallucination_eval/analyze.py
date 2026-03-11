"""
Analyze hallucination evaluation results from CSV.

Usage:
    python analyze.py results.csv [--plot]
"""

import argparse
import csv
import sys
from collections import defaultdict


def load_results(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            r["step"] = int(r["step"])
            r["sample_idx"] = int(r["sample_idx"])
            r["hallucination"] = r["hallucination"] == "True"
            r["answer_correct"] = r["answer_correct"] == "True"
            r["has_privileged_info"] = r["has_privileged_info"] == "True"
            rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to hallucination_results.csv")
    parser.add_argument("--plot", action="store_true", help="Save plots")
    args = parser.parse_args()

    rows = load_results(args.csv_path)
    if not rows:
        print("No results found")
        sys.exit(1)

    print(f"Loaded {len(rows)} records\n")

    # Per-step stats
    by_step = defaultdict(list)
    for r in rows:
        by_step[r["step"]].append(r)

    print(f"{'Step':>6} | {'N':>5} | {'Halluc':>6} | {'Rate':>6} | {'Acc':>6} | {'Halluc|Correct':>15} | {'Halluc|Wrong':>13}")
    print("-" * 80)
    step_data = []
    for step in sorted(by_step):
        recs = by_step[step]
        n = len(recs)
        h = sum(1 for r in recs if r["hallucination"])
        acc = sum(1 for r in recs if r["answer_correct"])
        h_correct = sum(1 for r in recs if r["hallucination"] and r["answer_correct"])
        h_wrong = sum(1 for r in recs if r["hallucination"] and not r["answer_correct"])
        n_correct = max(sum(1 for r in recs if r["answer_correct"]), 1)
        n_wrong = max(sum(1 for r in recs if not r["answer_correct"]), 1)
        print(f"{step:>6} | {n:>5} | {h:>6} | {100*h/n:>5.1f}% | {100*acc/n:>5.1f}% | {100*h_correct/n_correct:>13.1f}% | {100*h_wrong/n_wrong:>11.1f}%")
        step_data.append((step, h/n, acc/n))

    # Privileged info breakdown
    priv_rows = [r for r in rows if r["has_privileged_info"]]
    no_priv_rows = [r for r in rows if not r["has_privileged_info"]]
    print(f"\n--- Privileged info breakdown ---")
    if priv_rows:
        h_priv = sum(1 for r in priv_rows if r["hallucination"])
        print(f"With privileged info:    {h_priv}/{len(priv_rows)} hallucinations ({100*h_priv/len(priv_rows):.1f}%)")
    if no_priv_rows:
        h_nopriv = sum(1 for r in no_priv_rows if r["hallucination"])
        print(f"Without privileged info: {h_nopriv}/{len(no_priv_rows)} hallucinations ({100*h_nopriv/len(no_priv_rows):.1f}%)")

    # Overall
    total_h = sum(1 for r in rows if r["hallucination"])
    print(f"\nOverall: {total_h}/{len(rows)} ({100*total_h/len(rows):.1f}%) hallucinations")

    if args.plot and step_data:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps, h_rates, acc_rates = zip(*step_data)
            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(steps, [100*x for x in h_rates], "r-o", markersize=3, label="Hallucination %")
            ax1.set_xlabel("Training Step")
            ax1.set_ylabel("Hallucination Rate (%)", color="r")
            ax2 = ax1.twinx()
            ax2.plot(steps, [100*x for x in acc_rates], "b-o", markersize=3, label="Accuracy %")
            ax2.set_ylabel("Accuracy (%)", color="b")
            fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
            plt.title("Hallucination Rate vs Accuracy Over Training")
            plt.tight_layout()
            out_path = args.csv_path.replace(".csv", "_plot.png")
            plt.savefig(out_path, dpi=150)
            print(f"\nPlot saved to {out_path}")
        except ImportError:
            print("\nMatplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
