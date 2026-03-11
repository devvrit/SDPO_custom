#!/usr/bin/env python3
"""
Collect eval results from OPSD eval job logs and print a summary table.

Usage:
    python reproduce_paper/collect_eval_results.py [logs_dir]

Scans logs_dir (default: logs/) for opsd_eval log files and extracts
per-dataset and aggregate accuracy metrics.
"""

import re
import sys
import os
from pathlib import Path
from collections import defaultdict


def parse_eval_log(log_path: str) -> dict:
    """Parse a single eval log file and extract accuracy metrics."""
    results = {}

    with open(log_path, "r") as f:
        content = f.read()

    # Extract the suffix from the job name in the filename
    # Format: openthoughts_math-opsd_eval-qwen3-8b-ml.p4de.24xlarge-{suffix}_{jobid}.log
    basename = os.path.basename(log_path)
    match = re.match(r"openthoughts_math-opsd_eval-qwen3-8b-ml\.p4de\.24xlarge-(.+?)_(\d+)\.log", basename)
    if match:
        results["suffix"] = match.group(1)
        results["job_id"] = match.group(2)
    else:
        results["suffix"] = basename
        results["job_id"] = "?"

    # Check if job completed (has val-core metrics)
    if "val-core" not in content:
        results["status"] = "no_results"
        return results

    results["status"] = "ok"

    # Auto-detect n from metrics (e.g., mean@4, mean@16)
    n_match = re.search(r"val-core/\w+/acc/mean@(\d+)", content)
    n_val = n_match.group(1) if n_match else "16"
    results["n"] = int(n_val)

    # Extract all val-core metrics from the summary line
    # Pattern: val-core/{dataset}/acc/{metric}:{value}
    metrics_pattern = rf"val-core/(\w+)/acc/(mean@{n_val}|best@{n_val}/mean|maj@{n_val}/mean):([\d.]+)"

    for m in re.finditer(metrics_pattern, content):
        dataset = m.group(1)
        metric_name = m.group(2).replace("/mean", "")
        value = float(m.group(3))

        if dataset not in results:
            results[dataset] = {}
        results[dataset][metric_name] = value

    # Also try the dict-style format: 'val-core/{dataset}/acc/{metric}': value
    dict_pattern = rf"'val-core/(\w+)/acc/(mean@{n_val}|best@{n_val}/mean|maj@{n_val}/mean)':\s*([\d.]+)"
    for m in re.finditer(dict_pattern, content):
        dataset = m.group(1)
        metric_name = m.group(2).replace("/mean", "")
        value = float(m.group(3))

        if dataset not in results:
            results[dataset] = {}
        # Don't overwrite if already found (prefer the summary line)
        if metric_name not in results[dataset]:
            results[dataset][metric_name] = value

    return results


def format_pct(value):
    """Format a float as percentage string."""
    if value is None:
        return "  -  "
    return f"{value*100:5.1f}%"


def main():
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else "logs"
    logs_path = Path(logs_dir)

    # Find all opsd_eval log files
    log_files = sorted(logs_path.glob("*opsd_eval*.log"))

    if not log_files:
        print(f"No opsd_eval log files found in {logs_dir}/")
        return

    # Parse all logs
    all_results = []
    for log_file in log_files:
        result = parse_eval_log(str(log_file))
        all_results.append(result)

    # Filter to only successful results
    valid_results = [r for r in all_results if r["status"] == "ok"]
    failed_results = [r for r in all_results if r["status"] != "ok"]

    if failed_results:
        print("Jobs without results (may still be running or failed):")
        for r in failed_results:
            print(f"  Job {r['job_id']}: {r['suffix']}")
        print()

    if not valid_results:
        print("No completed eval results found.")
        return

    # Determine which datasets are present
    all_datasets = set()
    for r in valid_results:
        for key in r:
            if key not in ("suffix", "job_id", "status", "n") and isinstance(r[key], dict):
                all_datasets.add(key)

    # Sort datasets: individual ones first, then aggregate
    dataset_order = []
    for d in ["aime24", "aime25", "hmmt25", "amo"]:
        if d in all_datasets:
            dataset_order.append(d)
    if "math" in all_datasets:
        dataset_order.append("math")
    # Add any others
    for d in sorted(all_datasets):
        if d not in dataset_order:
            dataset_order.append(d)

    # Header
    print("=" * 140)
    print("OPSD Eval Results Summary")
    print("=" * 140)
    print()

    # Build column headers
    header1 = f"{'Run':<30} {'Job':>5} {'n':>2}"
    header2 = f"{'':<30} {'':>5} {'':>2}"
    sep = "-" * 30 + " " + "-" * 5 + " " + "-" * 2

    for dataset in dataset_order:
        header1 += f" | {dataset:^19}"
        header2 += f" | {'avg':>5} {'best':>6} {'maj':>6}"
        sep += " + " + "-" * 19

    print(header1)
    print(header2)
    print(sep)

    # Sort results: base first, then repro10 by step, then topk100 by step
    def sort_key(r):
        s = r["suffix"]
        # Extract step number from various formats
        step_match = re.search(r"s(\d+)", s) or re.search(r"step(\d+)", s)
        step = int(step_match.group(1)) if step_match else 0

        if "base" in s:
            return (0, 0, s)
        elif "repro10" in s:
            return (1, step, s)
        elif "topk100" in s:
            return (2, step, s)
        return (3, step, s)

    valid_results.sort(key=sort_key)

    for r in valid_results:
        n_val = r.get("n", 16)
        row = f"{r['suffix']:<30} {r['job_id']:>5} {n_val:>2}"

        for dataset in dataset_order:
            data = r.get(dataset, {})
            avg = format_pct(data.get(f"mean@{n_val}"))
            best = format_pct(data.get(f"best@{n_val}"))
            maj = format_pct(data.get(f"maj@{n_val}"))
            row += f" | {avg} {best} {maj}"

        print(row)

    print(sep)
    print()
    print(f"Datasets: {', '.join(dataset_order)}")
    print(f"Metrics: avg = pass@n average, best = best of n, maj = majority vote of n")
    print(f"Eval settings: temp=1.2, top_p=0.95, max_new_tokens=38912, thinking=true")


if __name__ == "__main__":
    main()
