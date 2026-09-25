import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor

sys.path.insert(0, "/root/autodl-tmp/thyroid_qc/code")
import train_thyroid_internvl_student_kgkd_v1_22_20260718 as base


TASKS = base.TASKS


def norm_id(value):
    if value is None:
        return ""
    text = str(value)
    for suffix in ["__doctor_repeat0", "__doctor_repeat1", "__doctor_repeat2", "__doctor_repeat3", "__doctor_repeat4"]:
        text = text.replace(suffix, "")
    return text


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def hard_soft(task, label_idx, confidence=0.98):
    n = len(TASKS[task])
    eps = max(0.02, 1.0 - confidence)
    vals = [eps / max(1, n - 1) for _ in range(n)]
    vals[label_idx] = 1.0 - eps
    return vals


def load_teacher_probability_bank(path):
    bank = {}
    if not path or not Path(path).exists():
        return bank
    for row in read_jsonl(path):
        probs = row.get("probabilities") or {}
        if not probs:
            continue
        converted = {}
        for task, labels in TASKS.items():
            task_probs = probs.get(task, {})
            vals = [float(task_probs.get(label, 0.0)) for label in labels]
            s = sum(vals)
            if s <= 0:
                converted = {}
                break
            converted[task] = [v / s for v in vals]
        if not converted:
            continue
        conf = row.get("confidence") or {}
        min_conf = float(row.get("min_confidence") or min([float(conf.get(t, 0.75)) for t in TASKS]))
        item = {"probs": converted, "confidence": max(0.55, min(0.95, min_conf))}
        for key in [row.get("sample_id"), row.get("segment_id"), row.get("review_id")]:
            key = norm_id(key)
            if key:
                bank[key] = item
    return bank


def teacher_lookup(row, bank):
    for key in [row.get("original_sample_id"), row.get("segment_id"), row.get("review_id"), row.get("sample_id")]:
        key = norm_id(key)
        if key in bank:
            return bank[key]
    return None


def load_manifest_with_real_soft(path, teacher_bank, max_images=1):
    rows = []
    soft_hits = 0
    for row in read_jsonl(path):
        labels = {}
        for task, vocab in TASKS.items():
            value = row["student_label"][task]
            if value not in vocab:
                raise ValueError(f"Illegal label for {task}: {value}")
            labels[task] = vocab.index(value)
        source = str(row.get("label_source", "")).lower()
        teacher_item = teacher_lookup(row, teacher_bank)
        is_doctor = "doctor" in source
        if is_doctor:
            conf = 0.98
            soft_targets = {task: hard_soft(task, labels[task], conf) for task in TASKS}
        elif teacher_item is not None:
            soft_hits += 1
            conf = teacher_item["confidence"]
            soft_targets = teacher_item["probs"]
        else:
            conf = 0.78 if "teacher" in source else 0.70
            soft_targets = {task: hard_soft(task, labels[task], conf) for task in TASKS}
        paths = row.get("input_paths") or [row.get("input_path")]
        rows.append(
            {
                "review_id": row.get("review_id", ""),
                "sample_id": row["sample_id"],
                "case_id": row.get("case_id", ""),
                "image_path": paths[:max_images][0],
                "labels": labels,
                "soft_targets": soft_targets,
                "teacher_confidence": float(conf),
                "split": row.get("student_training", {}).get("split", row.get("split", "")),
                "label_source": row.get("label_source", ""),
                "soft_hit": bool((not is_doctor) and teacher_item is not None),
            }
        )
    return rows, soft_hits


def probs_from_logits(logits):
    return {task: F.softmax(logits[task], dim=-1) for task in TASKS}


def reasoning_state(probs):
    iu = probs["image_usability"]
    sv = probs["structure_visibility"]
    kf = probs["keyframe_status"]
    defect = probs["primary_defect"]
    action = probs["recommended_action"]
    state = torch.stack(
        [
            (iu[:, 0] + sv[:, 0] + kf[:, 0] + defect[:, 0] + action[:, 0]) / 5.0,
            (sv[:, 1] + kf[:, 1] + defect[:, 3] + action[:, 3]) / 4.0,
            (sv[:, 2] + kf[:, 2] + defect[:, 2] + action[:, 2]) / 4.0,
            (iu[:, 2] + kf[:, 2] + defect[:, 1] + action[:, 1]) / 4.0,
        ],
        dim=1,
    )
    return state / state.sum(dim=1, keepdim=True).clamp_min(1e-6)


def r2s_loss(logits, soft_targets):
    pred = reasoning_state(probs_from_logits(logits))
    target_probs = {task: soft_targets[task].to(pred.device) for task in TASKS}
    target = reasoning_state(target_probs).detach()
    return F.kl_div(torch.log(pred.clamp_min(1e-6)), target, reduction="batchmean")


def kg_relation_loss(logits):
    probs = probs_from_logits(logits)
    iu = probs["image_usability"]
    sv = probs["structure_visibility"]
    kf = probs["keyframe_status"]
    defect = probs["primary_defect"]
    action = probs["recommended_action"]
    penalty = 0.0
    penalty = penalty + (sv[:, 2] * kf[:, 0]).mean()
    penalty = penalty + (iu[:, 2] * (kf[:, 0] + action[:, 0])).mean()
    penalty = penalty + (defect[:, 1] * (action[:, 0] + action[:, 2] + action[:, 3])).mean()
    penalty = penalty + (defect[:, 2] * (action[:, 0] + action[:, 1] + action[:, 3])).mean()
    penalty = penalty + (defect[:, 3] * (action[:, 0] + action[:, 1] + action[:, 2])).mean()
    penalty = penalty + (defect[:, 0] * (1.0 - action[:, 0])).mean()
    return penalty


def kd_loss(logits, soft_targets, confidence, temperature):
    losses = []
    for task in TASKS:
        target = soft_targets[task].to(logits[task].device)
        logp = F.log_softmax(logits[task] / temperature, dim=-1)
        sample_loss = F.kl_div(logp, target, reduction="none").sum(dim=-1) * (temperature ** 2)
        losses.append((sample_loss * confidence.to(logits[task].device)).sum() / confidence.to(logits[task].device).sum().clamp_min(1e-8))
    return sum(losses) / len(losses)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ablation-level", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--log-every", type=int, default=80)
    parser.add_argument("--kd-weight", type=float, default=0.20)
    parser.add_argument("--r2s-weight", type=float, default=0.08)
    parser.add_argument("--chain-weight", type=float, default=0.08)
    parser.add_argument("--kg-weight", type=float, default=0.05)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    args = parser.parse_args()

    base.set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_bank = load_teacher_probability_bank(args.teacher_jsonl)
    rows, soft_hits = load_manifest_with_real_soft(args.manifest, teacher_bank, max_images=args.max_images)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(json.dumps({
        "event": "loaded_rows",
        "ablation_level": args.ablation_level,
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "teacher_bank": len(teacher_bank),
        "soft_hits": soft_hits,
        "out_dir": str(out_dir),
    }, ensure_ascii=False), flush=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = base.InternVLQCClassifier(args.model_path, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha).cuda()
    collator = base.Collator(processor)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_ds = base.ThyroidDataset(train_rows, None, args.max_num)
    val_ds = base.ThyroidDataset(val_rows, None, args.max_num)
    test_ds = base.ThyroidDataset(test_rows, None, args.max_num)
    sampler = base.build_sampler(train_rows, args.class_weight_power)
    if sampler is not None and args.samples_per_epoch and args.samples_per_epoch > 0:
        sampler.num_samples = args.samples_per_epoch
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, collate_fn=collator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)

    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "r2s": 0.0, "chain": 0.0, "kg": 0.0}
        for step, batch in enumerate(train_loader, 1):
            logits = model(batch)
            conf = batch["teacher_confidence"].cuda()
            ce = sum(base.confidence_weighted_ce(logits[t], batch["labels"][t].cuda(), conf) for t in TASKS) / len(TASKS)
            loss = ce
            kd = torch.tensor(0.0, device=ce.device)
            r2s = torch.tensor(0.0, device=ce.device)
            chain = torch.tensor(0.0, device=ce.device)
            kg = torch.tensor(0.0, device=ce.device)
            if args.ablation_level >= 2:
                kd = kd_loss(logits, batch["soft_targets"], conf, args.kd_temperature)
                loss = loss + args.kd_weight * kd
            if args.ablation_level >= 3:
                r2s = r2s_loss(logits, batch["soft_targets"])
                loss = loss + args.r2s_weight * r2s
            if args.ablation_level >= 4:
                chain = base.clinical_chain_consistency_loss(logits)
                loss = loss + args.chain_weight * chain
            if args.ablation_level >= 5:
                kg = kg_relation_loss(logits)
                loss = loss + args.kg_weight * kg
            (loss / args.grad_accum).backward()
            for key, val in [("loss", loss), ("ce", ce), ("kd", kd), ("r2s", r2s), ("chain", chain), ("kg", kg)]:
                totals[key] += float(val.detach().cpu())
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % args.log_every == 0:
                print(json.dumps({
                    "epoch": epoch,
                    "step": step,
                    **{k: round(v / step, 4) for k, v in totals.items()},
                }, ensure_ascii=False), flush=True)
        val_metrics, val_graph_metrics, _, _ = base.evaluate(model, val_loader)
        test_metrics, test_graph_metrics, records, timing = base.evaluate(model, test_loader, measure_timing=True)
        row = {
            "epoch": epoch,
            **{f"train_{k}": totals[k] / max(1, len(train_loader)) for k in totals},
            "val_mean_macro_f1": val_metrics["mean_macro_f1"],
            "val_A5_mean_macro_f1": val_graph_metrics["mean_macro_f1"],
            "test_mean_macro_f1": test_metrics["mean_macro_f1"],
            "test_A5_mean_macro_f1": test_graph_metrics["mean_macro_f1"],
            "test_mean_macro_auc": test_metrics["mean_macro_auc"],
            "test_sec_per_sample": timing["sec_per_sample"],
            "test_fps": timing["samples_per_sec"],
        }
        history.append(row)
        print(json.dumps({"event": "epoch_eval", **row}, ensure_ascii=False), flush=True)
        if val_metrics["mean_macro_f1"] > best:
            best = val_metrics["mean_macro_f1"]
            torch.save(base.trainable_state_dict(model), out_dir / "best_trainable_state.pt")
            base.write_metrics_csv(out_dir / "best_test_metrics_at_val_best.csv", test_metrics)
            base.write_metrics_csv(out_dir / "best_A5_graph_metrics.csv", test_graph_metrics)
            (out_dir / "best_test_metrics_at_val_best.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "best_A5_graph_metrics.json").write_text(json.dumps(test_graph_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "efficiency.json").write_text(json.dumps({
                "model_name": "InternVL3.5-2B-HF",
                "parameter_count_total": total_params,
                "parameter_count_trainable": trainable_params,
                "test_timing": timing,
                "best_epoch_by_val": epoch,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(json.dumps({"event": "done", "out_dir": str(out_dir), "best_val_mean_macro_f1": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
