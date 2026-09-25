"""Evaluate ThyQC predictions with the bundled GT-QDM checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import run_student_gt_qdm_ablation_20260809 as thyqc


def flatten_probs(probabilities):
    values = []
    for task in thyqc.TASK_NAMES:
        values.extend(probabilities[task])
    return values


def load_features(path):
    anon_ids, sequences = [], []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            anon_ids.append(row["anon_id"])
            sequences.append([flatten_probs(token) for token in row["token_probs"]])
    return anon_ids, torch.tensor(sequences, dtype=torch.float32)


def load_labels(path, anon_ids):
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        labels_by_id = {row["anon_id"]: row for row in csv.DictReader(source)}
    return {
        task: np.asarray([thyqc.TASKS[task].index(labels_by_id[anon_id][task]) for anon_id in anon_ids])
        for task in thyqc.TASKS
    }


def load_state(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_seed(seed, output_dir):
    features = ROOT / "results" / "features" / f"seed{seed}_public_features.jsonl"
    checkpoint = ROOT / "weights" / "thyqc" / f"seed{seed}_gt_qdm_state.pt"
    labels = ROOT / "data" / "anonymized_test" / "labels.csv"

    anon_ids, inputs = load_features(features)
    y_true = load_labels(labels, anon_ids)
    model = thyqc.TemporalHead()
    model.load_state_dict(load_state(checkpoint))
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probabilities = {
            task: F.softmax(logits[task], dim=-1).cpu().numpy()
            for task in thyqc.TASKS
        }

    metrics = thyqc.compute_metrics(y_true, probabilities)
    metric_rows = [{"seed": seed, **row} for row in metrics]
    write_csv(
        output_dir / f"seed{seed}_metrics.csv",
        metric_rows,
        ["seed", "task", "Acc", "Macro-P", "Macro-R", "Macro-F1", "Macro-AUC"],
    )

    predictions = []
    for index, anon_id in enumerate(anon_ids):
        row = {"anon_id": anon_id}
        for task, classes in thyqc.TASKS.items():
            row[task] = classes[int(probabilities[task][index].argmax())]
        predictions.append(row)
    write_csv(
        output_dir / f"seed{seed}_predictions.csv",
        predictions,
        ["anon_id", *thyqc.TASK_NAMES],
    )
    return metric_rows


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, choices=(41, 42, 43))
    group.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "evaluation")
    args = parser.parse_args()

    seeds = (41, 42, 43) if args.all_seeds or args.seed is None else (args.seed,)
    all_rows = []
    for seed in seeds:
        rows = evaluate_seed(seed, args.output_dir)
        all_rows.extend(rows)
        mean_row = next(row for row in rows if row["task"] == "five_task_mean")
        print(json.dumps({
            "seed": seed,
            "Macro-F1": mean_row["Macro-F1"],
            "Macro-AUC": mean_row["Macro-AUC"],
        }))

    write_csv(
        args.output_dir / "all_seed_metrics.csv",
        all_rows,
        ["seed", "task", "Acc", "Macro-P", "Macro-R", "Macro-F1", "Macro-AUC"],
    )


if __name__ == "__main__":
    main()
