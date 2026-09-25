"""Train the original ThyQC GT-QDM head with the G2D-UOT objective."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_CODE_ROOT = os.environ.get("THYQC_CODE_ROOT", str(Path(__file__).resolve().parent))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)
import run_student_gt_qdm_ablation_20260809 as old

from g2d_uot_core import (clinical_relation_cost_matrix, flatten_task_probs,
                          semantic_cost_matrix,
                          unbalanced_sinkhorn_objective,
                          unbalanced_sinkhorn_transport_discrepancy)


def raw_teacher_targets(rows, teacher_bank):
    """Use pre-projection five-task probabilities directly as UOT source mass."""
    arrays = {task: [] for task in old.TASKS}
    hits = 0
    for row in rows:
        keys = [old.normalize_id(row.get(k)) for k in ("original_sample_id", "segment_id", "review_id", "sample_id")]
        item = next((teacher_bank[k] for k in keys if k in teacher_bank), None)
        hits += int(item is not None)
        for task in old.TASKS:
            if item is None:
                arrays[task].append(old.hard_distribution(row["student_label"][task], task))
            else:
                arrays[task].append(np.asarray(item[task], dtype=np.float32))
    return {task: np.stack(values).astype(np.float32) for task, values in arrays.items()}, hits


def evaluate(model, x, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        probs = {task: F.softmax(logits[task], dim=-1).cpu().numpy() for task in old.TASKS}
    return old.compute_metrics(y, probs), probs


def write_predictions(path, metadata, y, probs):
    fields = ["sample_id", "review_id"]
    for task, labels in old.TASKS.items():
        fields += [f"true_{task}", f"pred_{task}"] + [f"prob_{task}__{label}" for label in labels]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for i, meta in enumerate(metadata):
            row = {"sample_id": meta.get("sample_id"), "review_id": meta.get("review_id")}
            for task, labels in old.TASKS.items():
                row[f"true_{task}"] = labels[int(y[task][i])]
                row[f"pred_{task}"] = labels[int(np.argmax(probs[task][i]))]
                for j, label in enumerate(labels): row[f"prob_{task}__{label}"] = float(probs[task][i, j])
            writer.writerow(row)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prob-cache", required=True)
    ap.add_argument("--teacher-jsonl", required=True)
    ap.add_argument("--semantic-cost-bank", default="")
    ap.add_argument("--semantic-cost-mode", choices=("bank", "fixed_jaccard"), default="bank")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--init-state", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=260)
    ap.add_argument("--patience", type=int, default=55)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-prob", type=float, default=0.06)
    ap.add_argument("--lambda-g2d", type=float, default=0.06)
    ap.add_argument("--clinical-eta", type=float, default=1.0)
    ap.add_argument("--uot-epsilon", type=float, default=0.08)
    ap.add_argument("--uot-rho", type=float, default=0.50)
    ap.add_argument("--uot-iterations", type=int, default=30)
    ap.add_argument("--uot-loss-mode", choices=("full", "transport"), default="full")
    ap.add_argument("--validation-only", action="store_true",
                    help="Tune on validation only; never evaluate or write test predictions.")
    args = ap.parse_args()

    old.set_seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = old.deduplicate_rows([r for r in old.read_jsonl(args.manifest) if str(r.get("label_source", "")).startswith("doctor_gold")])
    splits = {s: [r for r in rows if r.get("student_training", {}).get("split") == s] for s in ("train", "val", "test")}
    cache = old.load_prob_cache(args.prob_cache)
    teacher = old.load_teacher_probs(args.teacher_jsonl)
    semantic_bank = None
    if args.semantic_cost_mode == "bank":
        if not args.semantic_cost_bank:
            raise ValueError("--semantic-cost-bank is required for bank mode")
        semantic_bank = torch.load(args.semantic_cost_bank, map_location="cpu", weights_only=False)["costs"]
    fixed_semantic = semantic_cost_matrix(old.TASKS)
    data = {}
    teacher_hits = {}
    for name in ("train", "val", "test"):
        x, y, _, meta, _ = old.rows_to_arrays(splits[name], cache, "global_ordered", teacher)
        source, hits = raw_teacher_targets(splits[name], teacher)
        sample_costs = []
        for item in meta:
            if semantic_bank is None:
                value = fixed_semantic
            else:
                keys = [old.normalize_id(item.get("sample_id")), old.normalize_id(item.get("review_id"))]
                value = next((semantic_bank[key] for key in keys if key in semantic_bank), None)
                if value is None:
                    raise KeyError(f"No sample-specific semantic cost for {keys}")
            sample_costs.append(torch.as_tensor(value, dtype=torch.float32))
        data[name] = (x, y, source, meta, torch.stack(sample_costs)); teacher_hits[name] = hits
    print(json.dumps({"event": "loaded", "method": "ThyQC_G2D_UOT_GT_QDM", "seed": args.seed,
        "splits": {k: len(v) for k, v in splits.items()}, "teacher_hits": teacher_hits,
        "input_shape": list(data["train"][0].shape), "source_probability_stage": "raw_pre_KG_projection"}), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = old.TemporalHead().to(device)  # exact legacy GT-QDM architecture
    if args.init_state:
        checkpoint = torch.load(args.init_state, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    class_weights = {t: old.class_weights(data["train"][1][t], len(old.TASKS[t])).to(device) for t in old.TASKS}
    train_x = torch.tensor(data["train"][0], dtype=torch.float32)
    teacher_t = old.soft_to_tensor(data["train"][2], device)
    clinical_cost = clinical_relation_cost_matrix(old.TASKS).to(device)
    initial_val_metrics, _ = evaluate(model, data["val"][0], data["val"][1], device)
    best_score = next(r for r in initial_val_metrics if r["task"] == "five_task_mean")["Macro-F1"]
    best_epoch, stale = 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [{"epoch": 0, "val_mean_macro_f1": best_score}]
    print(json.dumps({"event": "initial_checkpoint_eval", "val_mean_macro_f1": best_score,
        "test_evaluated": False}), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train(); order = torch.randperm(len(train_x)); totals = {k: 0.0 for k in (
            "loss", "cls", "prob", "g2d_uot", "uot_transport",
            "uot_source_kl", "uot_target_kl", "uot_negative_entropy"
        )}; batches = 0
        for start in range(0, len(order), args.batch_size):
            sel = order[start:start + args.batch_size]; idx = sel.numpy(); logits = model(train_x[sel].to(device))
            anchor = sum(F.cross_entropy(logits[t], torch.tensor(data["train"][1][t][idx], device=device), weight=class_weights[t]) for t in old.TASKS) / len(old.TASKS)
            prob = sum(F.kl_div(F.log_softmax(logits[t], dim=-1), teacher_t[t][sel].detach(), reduction="batchmean") for t in old.TASKS) / len(old.TASKS)
            student = {t: F.softmax(logits[t], dim=-1) for t in old.TASKS}
            source_mass = flatten_task_probs({t: teacher_t[t][sel] for t in old.TASKS}, list(old.TASKS))
            target_mass = flatten_task_probs(student, list(old.TASKS))
            cost = data["train"][4][sel].to(device) + args.clinical_eta * clinical_cost.unsqueeze(0)
            if args.uot_loss_mode == "transport":
                uot = unbalanced_sinkhorn_transport_discrepancy(
                    source_mass, target_mass, cost, epsilon=args.uot_epsilon,
                    rho=args.uot_rho, iterations=args.uot_iterations
                )
                zero = uot.detach() * 0
                uot_parts = {"transport": uot.detach(), "source_kl": zero,
                             "target_kl": zero, "negative_entropy": zero}
            else:
                uot, uot_parts = unbalanced_sinkhorn_objective(
                    source_mass, target_mass, cost, epsilon=args.uot_epsilon,
                    rho=args.uot_rho, iterations=args.uot_iterations, return_components=True
                )
            loss = anchor + args.lambda_prob * prob + args.lambda_g2d * uot
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            for k, v in (("loss", loss), ("cls", anchor), ("prob", prob), ("g2d_uot", uot)): totals[k] += float(v.detach().cpu())
            for key, value in uot_parts.items(): totals[f"uot_{key}"] += float(value.cpu())
            batches += 1
        val_metrics, _ = evaluate(model, data["val"][0], data["val"][1], device)
        val_score = next(r for r in val_metrics if r["task"] == "five_task_mean")["Macro-F1"]
        row = {"epoch": epoch, **{f"train_{k}": totals[k] / batches for k in totals}, "val_mean_macro_f1": val_score}
        history.append(row)
        if val_score > best_score:
            best_score, best_epoch = val_score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else: stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"event": "epoch_eval", **row, "best_epoch": best_epoch, "best_val": best_score}), flush=True)
        if stale >= args.patience: break

    model.load_state_dict(best_state)
    torch.save(best_state, out / "best_gt_qdm_state.pt")
    val_metrics, val_probs = evaluate(model, data["val"][0], data["val"][1], device)
    old.write_task_metrics(out / "best_validation_metrics.csv", [{"variant": "ThyQC_G2D_UOT", **r} for r in val_metrics])
    (out / "best_validation_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    config = {**vars(args), "method": "ThyQC_G2D_UOT", "architecture": "legacy_TemporalHead_GT_QDM",
        "input": "contact-sheet probability + 4 ordered sparse-frame probabilities", "classification_anchor": "physician CE",
        "pseudo_hard_anchor": "retained in frozen B4/frame probability producer", "probability_fidelity": True,
        "reasoning_state_loss": False, "independent_chain_loss": False, "independent_kg_loss": False,
        "inference_kg_projection": False, "teacher_source_mass": "raw pre-projection five-task probabilities",
        "semantic_cost": args.semantic_cost_mode, "clinical_cost": "legacy chain+KG relations",
        "uot_objective": ("relaxed-marginal Sinkhorn transport discrepancy" if args.uot_loss_mode == "transport"
                          else "transport + source generalized-KL + target generalized-KL - entropy"),
        "teacher_hits": teacher_hits, "best_epoch_by_validation": best_epoch}
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"event": "done", "best_epoch": best_epoch, "best_val_mean_macro_f1": best_score,
              "test_evaluated": False, "out_dir": str(out)}
    if not args.validation_only:
        test_metrics, test_probs = evaluate(model, data["test"][0], data["test"][1], device)
        old.write_task_metrics(out / "best_test_metrics_at_val_best.csv", [{"variant": "ThyQC_G2D_UOT", **r} for r in test_metrics])
        (out / "best_test_metrics_at_val_best.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        write_predictions(out / "best_test_predictions_at_val_best.csv", data["test"][3], data["test"][1], test_probs)
        mean = next(r for r in test_metrics if r["task"] == "five_task_mean")
        result.update({"test_evaluated": True, "test_mean_macro_f1": mean["Macro-F1"],
                       "test_mean_macro_auc": mean["Macro-AUC"]})
    print(json.dumps(result), flush=True)


if __name__ == "__main__": run()
