"""ThyQC G2D-UOT main run.

This is intentionally a narrow successor to the clean GT-QDM pipeline: the
InternVL/LoRA model, ordered contact-sheet input, physician labels, teacher
probability bank, and evaluation are reused.  R2S is removed; chain/KG rules
only define the clinical transport geometry.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor

_CODE_ROOT = os.environ.get("THYQC_CODE_ROOT", str(Path(__file__).resolve().parent))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)
import train_internvl_clean_gt_qdm_student_20260809 as clean

legacy = clean.base

from g2d_uot_core import (
    RATIONALE_TOKENS,
    clinical_relation_cost_matrix,
    contextual_semantic_cost,
    flatten_task_probs,
    probability_conditioned_semantic_cost,
    semantic_cost_matrix,
    unbalanced_sinkhorn_objective,
    unbalanced_sinkhorn_transport_discrepancy,
)

TASKS = clean.TASKS


def load_teacher_rationale_bank(path):
    bank = {}
    for row in clean.read_jsonl(path):
        prediction = row.get("raw_prediction") or {
            task: max(row["probabilities"][task], key=row["probabilities"][task].get)
            for task in TASKS
        }
        clauses = []
        for task in TASKS:
            label = prediction[task]
            evidence = " ".join(sorted(RATIONALE_TOKENS[label]))
            clauses.append(f"{task}: state={label}; rationale={evidence}")
        text = "; ".join(clauses)
        for key in (row.get("sample_id"), row.get("segment_id"), row.get("review_id")):
            key = clean.norm_id(key)
            if key:
                bank[key] = text
    return bank


def rationale_lookup(row, bank, soft_targets):
    for key in (row.get("original_sample_id"), row.get("segment_id"), row.get("review_id"), row.get("sample_id")):
        value = bank.get(clean.norm_id(key))
        if value:
            return value
    clauses = []
    for task, labels in TASKS.items():
        label = labels[int(np.argmax(soft_targets[task]))]
        clauses.append(f"{task}: state={label}; rationale={' '.join(sorted(RATIONALE_TOKENS[label]))}")
    return "; ".join(clauses)


def load_manifest_with_raw_teacher(
    path,
    teacher_bank,
    rationale_bank,
    max_images=1,
    teacher_on_physician=False,
    prob_teacher_bank=None,
    prob_teacher_on_physician=False,
):
    """Build the legacy rows while keeping teacher mass before graph projection."""
    rows = []
    soft_hits = 0
    for row in clean.read_jsonl(path):
        labels = {task: vocab.index(row["student_label"][task]) for task, vocab in TASKS.items()}
        source = str(row.get("label_source", "")).lower()
        is_physician = "doctor" in source
        teacher_item = clean.teacher_lookup(row, teacher_bank)
        prob_teacher_item = clean.teacher_lookup(
            row, teacher_bank if prob_teacher_bank is None else prob_teacher_bank
        )
        # Physician-reviewed rows are supervision anchors.  Do not replace
        # their task-local/UOT source mass with a possibly conflicting teacher
        # prediction merely because the same sample also exists in the teacher
        # bank.  Teacher probabilities remain the source mass for the expanded
        # teacher-pseudo rows, matching the established ThyQC data contract.
        use_teacher = teacher_item is not None and (
            teacher_on_physician or not is_physician
        )
        if use_teacher:
            soft_hits += 1
            uot_targets = teacher_item["probs"]
        else:
            uot_targets = {
                task: clean.hard_soft(
                    task, labels[task], 0.98 if is_physician else 0.70
                )
                for task in TASKS
            }
        use_prob_teacher = prob_teacher_item is not None and (
            prob_teacher_on_physician or not is_physician
        )
        if is_physician and not use_prob_teacher:
            # L_prob is task-local fidelity and must not fight a gold anchor.
            soft_targets = {
                task: clean.hard_soft(task, labels[task], 0.98)
                for task in TASKS
            }
        elif use_prob_teacher:
            soft_targets = prob_teacher_item["probs"]
        else:
            soft_targets = {
                task: clean.hard_soft(task, labels[task], 0.70) for task in TASKS
            }
        paths = row.get("input_paths") or [row.get("input_path")]
        rows.append({
            "review_id": row.get("review_id", ""),
            "sample_id": row["sample_id"],
            "case_id": row.get("case_id", ""),
            "image_path": paths[:max_images][0],
            "labels": labels,
            "soft_targets": soft_targets,
            "uot_targets": uot_targets,
            # The threshold in anchor_ce is now exact: teacher confidence is
            # capped at .95 by the loader, while physician anchors are .98.
            "teacher_confidence": 0.98 if is_physician else (
                teacher_item["confidence"] if teacher_item is not None else 0.70
            ),
            "split": row.get("student_training", {}).get("split", row.get("split", "")),
            "label_source": row.get("label_source", ""),
            "soft_hit": use_teacher,
            "rationale_text": rationale_lookup(
                row, rationale_bank if use_teacher else {}, uot_targets
            ),
        })
    return rows, soft_hits


class G2DDataset(legacy.ThyroidDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["semantic_cost"] = torch.tensor(self.rows[idx]["semantic_cost"], dtype=torch.float32)
        item["uot_targets"] = {
            task: torch.tensor(
                self.rows[idx]["uot_targets"][task], dtype=torch.float32
            )
            for task in TASKS
        }
        return item


class G2DCollator(legacy.Collator):
    def __call__(self, batch):
        result = super().__call__(batch)
        result["semantic_cost"] = torch.stack([item["semantic_cost"] for item in batch])
        result["uot_targets"] = {
            task: torch.stack([item["uot_targets"][task] for item in batch])
            for task in TASKS
        }
        return result


@torch.no_grad()
def embed_texts(model, tokenizer, texts, batch_size=64):
    embedding = model.backbone.language_model.get_input_embeddings()
    device = next(model.parameters()).device
    outputs = []
    for start in range(0, len(texts), batch_size):
        toks = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                         max_length=160, return_tensors="pt")
        ids = toks["input_ids"].to(device)
        mask = toks["attention_mask"].to(device).unsqueeze(-1)
        vectors = embedding(ids).float()
        outputs.append(((vectors * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu())
    return torch.cat(outputs)


def attach_sample_semantic_costs(rows, model, tokenizer, context_weight, mode):
    prototype_texts = [
        f"{task}: state={label}; rationale={' '.join(sorted(RATIONALE_TOKENS[label]))}"
        for task, labels in TASKS.items() for label in labels
    ]
    prototypes = embed_texts(model, tokenizer, prototype_texts)
    if mode == "fixed_jaccard":
        costs = semantic_cost_matrix(TASKS).unsqueeze(0).expand(len(rows), -1, -1).clone()
    else:
        teacher_mass = torch.stack([
            torch.cat([torch.as_tensor(row["uot_targets"][task]) for task in TASKS])
            for row in rows
        ]).float()
        costs = probability_conditioned_semantic_cost(
            prototypes, teacher_mass, context_weight=context_weight
        )
    for row, matrix in zip(rows, costs):
        row["semantic_cost"] = matrix.numpy()
    return prototypes, costs


def anchor_ce(logits, labels, confidence, pseudo_weight):
    """Physician CE is the anchor; teacher pseudo-hard CE is down-weighted."""
    # The legacy collator does not carry label_source into the batch.  Its
    # doctor anchors are assigned confidence 0.98, while pseudo rows are <=
    # 0.95, so this threshold preserves the old source distinction exactly.
    weights = torch.where(confidence.to(next(iter(logits.values())).device) >= 0.97,
                          torch.ones_like(confidence),
                          torch.full_like(confidence, float(pseudo_weight)))
    losses = []
    for task in TASKS:
        y = labels[task].to(weights.device)
        # Average over examples, not over the sum of their weights.  Dividing by
        # weights.sum() cancels pseudo_weight completely when batch_size == 1.
        losses.append((F.cross_entropy(logits[task], y, reduction="none") * weights).mean())
    return sum(losses) / len(losses)


def local_probability_fidelity(logits, soft_targets, confidence, temperature=2.0):
    losses = []
    for task in TASKS:
        target = soft_targets[task].to(logits[task].device)
        logp = F.log_softmax(logits[task] / temperature, dim=-1)
        per_sample = F.kl_div(logp, target, reduction="none").sum(dim=-1) * (temperature**2)
        w = confidence.to(logits[task].device)
        losses.append((per_sample * w).sum() / w.sum().clamp_min(1e-8))
    return sum(losses) / len(losses)


def g2d_uot_loss(logits, soft_targets, cost, *, epsilon, rho, iterations, mode="full"):
    student = {task: F.softmax(logits[task], dim=-1) for task in TASKS}
    teacher = {task: soft_targets[task].to(next(iter(logits.values())).device) for task in TASKS}
    source = flatten_task_probs(teacher, list(TASKS))
    target = flatten_task_probs(student, list(TASKS))
    if mode == "transport":
        value = unbalanced_sinkhorn_transport_discrepancy(
            source, target, cost, epsilon=epsilon, rho=rho, iterations=iterations
        )
        zero = value.detach() * 0
        return value, {"transport": value.detach(), "source_kl": zero,
                       "target_kl": zero, "negative_entropy": zero}
    return unbalanced_sinkhorn_objective(
        source, target, cost, epsilon=epsilon, rho=rho,
        iterations=iterations, return_components=True
    )


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--teacher-jsonl", required=True)
    ap.add_argument(
        "--prob-teacher-jsonl",
        default="",
        help=(
            "Optional task-local L_prob teacher bank. UOT always uses the raw "
            "--teacher-jsonl bank."
        ),
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--init-state", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--samples-per-epoch", type=int, default=1600)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--class-weight-power", type=float, default=0.5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-num", type=int, default=6)
    ap.add_argument("--max-images", type=int, default=1)
    ap.add_argument("--pseudo-hard-weight", type=float, default=0.25)
    ap.add_argument(
        "--teacher-on-physician",
        action="store_true",
        help=(
            "Keep physician labels for L_cls while using available teacher "
            "probabilities/rationales as the L_prob and UOT source mass."
        ),
    )
    ap.add_argument(
        "--prob-teacher-on-physician",
        action="store_true",
        help="Use teacher probabilities for physician rows in L_prob as well.",
    )
    ap.add_argument("--lambda-prob", type=float, default=0.06)
    ap.add_argument("--lambda-g2d", type=float, default=0.06)
    ap.add_argument("--clinical-eta", type=float, default=1.0)
    ap.add_argument("--semantic-context-weight", type=float, default=2.0)
    ap.add_argument("--semantic-cost-mode", choices=("probability", "fixed_jaccard"),
                    default="probability")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--uot-epsilon", type=float, default=0.08)
    ap.add_argument("--uot-rho", type=float, default=0.50)
    ap.add_argument("--uot-iterations", type=int, default=30)
    ap.add_argument("--uot-loss-mode", choices=("full", "transport"), default="full")
    ap.add_argument("--log-every", type=int, default=40)
    ap.add_argument("--validation-only", action="store_true",
                    help="Tune on validation only; never evaluate or write test results.")
    ap.add_argument("--preserve-legacy-test-loader-rng", action="store_true",
                    help="Consume the DataLoader base-seed draw formerly caused by test evaluation, without reading test data.")
    args = ap.parse_args()

    legacy.set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bank = clean.load_teacher_probability_bank(args.teacher_jsonl)
    prob_bank = clean.load_teacher_probability_bank(
        args.prob_teacher_jsonl or args.teacher_jsonl
    )
    rationale_bank = load_teacher_rationale_bank(args.teacher_jsonl)
    rows, soft_hits = load_manifest_with_raw_teacher(
        args.manifest,
        bank,
        rationale_bank,
        max_images=args.max_images,
        teacher_on_physician=args.teacher_on_physician,
        prob_teacher_bank=prob_bank,
        prob_teacher_on_physician=args.prob_teacher_on_physician,
    )
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    print(json.dumps({"event": "loaded_rows", "method": "g2d_uot_thyqc", "seed": args.seed,
                      "train": len(split["train"]), "val": len(split["val"]), "test": len(split["test"]),
                      "teacher_bank": len(bank), "soft_hits": soft_hits, "out_dir": str(out)}), flush=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = legacy.InternVLQCClassifier(args.model_path, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha).cuda()
    init_report = None
    if args.init_state:
        init_report = legacy.load_trainable_state(model, args.init_state)
    prototypes, semantic_costs = attach_sample_semantic_costs(
        rows, model, processor.tokenizer, args.semantic_context_weight,
        args.semantic_cost_mode
    )
    semantic_bank = {}
    for row in rows:
        for key in (row.get("sample_id"), row.get("review_id")):
            key = clean.norm_id(key)
            if key:
                semantic_bank[key] = torch.tensor(row["semantic_cost"])
    torch.save({"costs": semantic_bank, "prototypes": prototypes,
                "context_weight": args.semantic_context_weight}, out / "semantic_cost_bank.pt")
    collator = G2DCollator(processor)
    train_ds = G2DDataset(split["train"], None, args.max_num)
    val_ds = G2DDataset(split["val"], None, args.max_num)
    test_ds = G2DDataset(split["test"], None, args.max_num)
    sampler = legacy.build_sampler(split["train"], args.class_weight_power)
    if sampler is not None and args.samples_per_epoch > 0:
        sampler.num_samples = args.samples_per_epoch
    loader = lambda ds, train=False: DataLoader(ds, batch_size=args.batch_size if train else 1,
        sampler=sampler if train else None, shuffle=(sampler is None and train), collate_fn=collator, num_workers=0)
    train_loader, val_loader, test_loader = loader(train_ds, True), loader(val_ds), loader(test_ds)
    clinical_cost = clinical_relation_cost_matrix(TASKS).cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    initial_val, _, _, _ = legacy.evaluate(model, val_loader)
    if args.validation_only and args.preserve_legacy_test_loader_rng:
        # Creating a DataLoader iterator consumes one global torch RNG draw for
        # its base seed, even with num_workers=0. Preserve the historical
        # WeightedRandomSampler trajectory without touching held-out examples.
        torch.empty((), dtype=torch.int64).random_()
    best = initial_val["mean_macro_f1"]
    history = [{"epoch": 0, "val_mean_macro_f1": best}]
    torch.save(legacy.trainable_state_dict(model), out / "best_trainable_state.pt")
    (out / "best_validation_metrics.json").write_text(json.dumps(initial_val, indent=2), encoding="utf-8")
    print(json.dumps({"event": "initial_checkpoint_eval", "load": init_report,
                      "val_mean_macro_f1": best, "test_evaluated": False}), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        totals = {k: 0.0 for k in (
            "loss", "cls", "prob", "g2d_uot", "uot_transport",
            "uot_source_kl", "uot_target_kl", "uot_negative_entropy"
        )}
        for step, batch in enumerate(train_loader, 1):
            logits = model(batch)
            labels = {task: batch["labels"][task].cuda() for task in TASKS}
            conf = batch["teacher_confidence"].cuda()
            # Collator preserves metadata in order; use the same rows to mark
            # doctor anchors versus teacher-generated pseudo-hard examples.
            anchor = anchor_ce(logits, labels, conf, args.pseudo_hard_weight)
            prob = local_probability_fidelity(logits, batch["soft_targets"], conf, args.temperature)
            sample_cost = batch["semantic_cost"].cuda() + args.clinical_eta * clinical_cost.unsqueeze(0)
            uot, uot_parts = g2d_uot_loss(logits, batch["uot_targets"], sample_cost, epsilon=args.uot_epsilon,
                                          rho=args.uot_rho, iterations=args.uot_iterations,
                                          mode=args.uot_loss_mode)
            loss = anchor + args.lambda_prob * prob + args.lambda_g2d * uot
            (loss / args.grad_accum).backward()
            for key, val in (("loss", loss), ("cls", anchor), ("prob", prob), ("g2d_uot", uot)):
                totals[key] += float(val.detach().cpu())
            for key, val in uot_parts.items():
                totals[f"uot_{key}"] += float(val.cpu())
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if step % args.log_every == 0:
                print(json.dumps({"event": "train_progress", "epoch": epoch, "step": step,
                                  **{k: round(v / step, 5) for k, v in totals.items()}}), flush=True)
        val_metrics, val_graph, _, _ = legacy.evaluate(model, val_loader)
        if args.validation_only and args.preserve_legacy_test_loader_rng:
            torch.empty((), dtype=torch.int64).random_()
        row = {"epoch": epoch, **{f"train_{k}": totals[k] / max(1, len(train_loader)) for k in totals},
               "val_mean_macro_f1": val_metrics["mean_macro_f1"], "val_mean_macro_auc": val_metrics["mean_macro_auc"]}
        history.append(row); print(json.dumps({"event": "epoch_eval", **row}), flush=True)
        if row["val_mean_macro_f1"] > best:
            best = row["val_mean_macro_f1"]
            torch.save(legacy.trainable_state_dict(model), out / "best_trainable_state.pt")
            (out / "best_validation_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({**vars(args), "method": "g2d_uot_thyqc", "rs_loss": False,
        "independent_chain_loss": False, "independent_kg_loss": False, "clinical_cost": "chain+kg mappings in C_clin",
        "task_count": len(TASKS), "teacher_soft_hits": soft_hits,
        "prob_teacher_source": args.prob_teacher_jsonl or args.teacher_jsonl,
        "teacher_source_mass": "raw probabilities before consistency-graph projection",
        "semantic_cost": args.semantic_cost_mode,
        "uot_objective": "transport + source generalized-KL + target generalized-KL - entropy"}, indent=2), encoding="utf-8")
    if not args.validation_only:
        legacy.load_trainable_state(model, str(out / "best_trainable_state.pt"))
        test_metrics, test_graph, records, timing = legacy.evaluate(model, test_loader, measure_timing=True)
        legacy.write_metrics_csv(out / "best_test_metrics_at_val_best.csv", test_metrics)
        legacy.write_metrics_csv(out / "best_A5_graph_metrics.csv", test_graph)
        (out / "best_test_metrics_at_val_best.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        (out / "best_test_records_at_val_best.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "best_A5_graph_metrics.json").write_text(json.dumps(test_graph, indent=2), encoding="utf-8")
        (out / "efficiency.json").write_text(json.dumps({"model_name": "InternVL3.5-2B-HF", "test_timing": timing,
            "parameter_count_total": total_params, "parameter_count_trainable": trainable_params,
            "best_epoch_by_val": max(history, key=lambda x: x["val_mean_macro_f1"])["epoch"]}, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "method": "g2d_uot_thyqc", "best_epoch": max(history, key=lambda x: x["val_mean_macro_f1"])["epoch"],
                      "best_val_mean_macro_f1": best, "test_evaluated": not args.validation_only,
                      "out_dir": str(out)}), flush=True)


if __name__ == "__main__":
    run()
