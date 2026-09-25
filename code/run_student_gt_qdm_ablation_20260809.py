import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TASKS = {
    "image_usability": ["acceptable", "limited", "unacceptable"],
    "structure_visibility": ["adequate", "partial", "absent", "uncertain"],
    "keyframe_status": ["yes", "candidate", "no"],
    "primary_defect": ["none", "severe_artifact", "thyroid_not_visible", "target_incomplete"],
    "recommended_action": ["hold", "repeat_acquisition", "reacquire_target", "adjust_angle_and_rescan"],
}
TASK_NAMES = list(TASKS)
PROB_DIM = sum(len(v) for v in TASKS.values())
STATE_NAMES = ["normal_hold", "incomplete_rescan", "absent_reacquire", "artifact_repeat"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def deduplicate_rows(rows):
    seen = set()
    out = []
    for row in rows:
        split = row.get("student_training", {}).get("split", "")
        key_id = (
            row.get("original_sample_id")
            or row.get("contact_sheet_source")
            or row.get("review_id")
            or row.get("segment_id")
            or row.get("sample_id")
        )
        key = (split, key_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def flatten_probs(prob_dict):
    vals = []
    for task in TASK_NAMES:
        vals.extend(prob_dict[task])
    return np.asarray(vals, dtype=np.float32)


def split_flat_probs(flat_batch):
    out = {}
    i = 0
    for task, labels in TASKS.items():
        n = len(labels)
        out[task] = flat_batch[:, i : i + n]
        i += n
    return out


def load_prob_cache(path):
    cache = {}
    for row in read_jsonl(path):
        cache[row["path"]] = row["probs"]
    return cache


def normalize_id(value):
    if value is None:
        return ""
    return str(value).replace("__doctor_repeat0", "").replace("__doctor_repeat1", "").replace("__doctor_repeat2", "").replace("__doctor_repeat3", "").replace("__doctor_repeat4", "")


def load_teacher_probs(path):
    out = {}
    if not path or not Path(path).exists():
        return out
    for row in read_jsonl(path):
        probs = row.get("probabilities") or {}
        if not probs:
            continue
        converted = {}
        ok = True
        for task, labels in TASKS.items():
            task_probs = probs.get(task, {})
            vals = [float(task_probs.get(lbl, 0.0)) for lbl in labels]
            s = sum(vals)
            if s <= 0:
                ok = False
                break
            converted[task] = [v / s for v in vals]
        if not ok:
            continue
        keys = {
            normalize_id(row.get("sample_id")),
            normalize_id(row.get("segment_id")),
            normalize_id(row.get("review_id")),
        }
        for key in keys:
            if key:
                out[key] = converted
    return out


def hard_distribution(label, task, smoothing=0.03):
    labels = TASKS[task]
    arr = np.full(len(labels), smoothing / max(1, len(labels) - 1), dtype=np.float32)
    arr[labels.index(label)] = 1.0 - smoothing
    return arr


def anchored_soft_distribution(row, teacher_probs, task, teacher_mix=0.35):
    hard = hard_distribution(row["student_label"][task], task)
    keys = [
        normalize_id(row.get("original_sample_id")),
        normalize_id(row.get("segment_id")),
        normalize_id(row.get("review_id")),
        normalize_id(row.get("sample_id")),
    ]
    soft = None
    for key in keys:
        if key in teacher_probs:
            soft = np.asarray(teacher_probs[key][task], dtype=np.float32)
            break
    if soft is None:
        return hard
    # The physician label remains the anchor; teacher probabilities only provide dark knowledge.
    mixed = (1.0 - teacher_mix) * hard + teacher_mix * soft
    mixed = mixed / np.clip(mixed.sum(), 1e-8, None)
    return mixed.astype(np.float32)


def rows_to_arrays(rows, image_prob_cache, mode, teacher_probs=None):
    xs = []
    ys = {task: [] for task in TASKS}
    soft = {task: [] for task in TASKS}
    meta = []
    teacher_hit = 0
    for row in rows:
        paths = row["input_paths"]
        if mode == "contact_sheet":
            seq_paths = [paths[0]]
        elif mode == "global_ordered":
            seq_paths = paths[:1] + paths[-4:]
        else:
            seq_paths = paths[-4:]
        xs.append(np.stack([flatten_probs(image_prob_cache[p]) for p in seq_paths], axis=0))
        hit = False
        for task in TASKS:
            ys[task].append(TASKS[task].index(row["student_label"][task]))
            dist = anchored_soft_distribution(row, teacher_probs or {}, task)
            soft[task].append(dist)
            if teacher_probs:
                for key in [normalize_id(row.get("original_sample_id")), normalize_id(row.get("segment_id")), normalize_id(row.get("review_id")), normalize_id(row.get("sample_id"))]:
                    if key in teacher_probs:
                        hit = True
                        break
        teacher_hit += int(hit)
        meta.append({"sample_id": row.get("sample_id"), "review_id": row.get("review_id"), "split": row.get("student_training", {}).get("split")})
    return (
        np.stack(xs, axis=0),
        {t: np.asarray(v, dtype=np.int64) for t, v in ys.items()},
        {t: np.stack(v, axis=0).astype(np.float32) for t, v in soft.items()},
        meta,
        teacher_hit,
    )


def binary_auc(y_true, scores):
    pairs = sorted(zip(scores, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def macro_auc(y_true, probs, n_classes):
    vals = []
    for c in range(n_classes):
        auc = binary_auc([1 if y == c else 0 for y in y_true], probs[:, c].tolist())
        if auc is not None:
            vals.append(auc)
    return float(sum(vals) / len(vals)) if vals else 0.0


def compute_metrics(y_true_by_task, probs_by_task):
    rows = []
    for task, labels in TASKS.items():
        y_true = y_true_by_task[task]
        probs = probs_by_task[task]
        y_pred = probs.argmax(axis=1)
        n = len(labels)
        cm = np.zeros((n, n), dtype=int)
        for yt, yp in zip(y_true, y_pred):
            cm[int(yt), int(yp)] += 1
        ps, rs, fs = [], [], []
        for i in range(n):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            ps.append(p)
            rs.append(r)
            fs.append(f1)
        rows.append({
            "task": task,
            "Acc": float((y_true == y_pred).mean() * 100.0),
            "Macro-P": float(np.mean(ps) * 100.0),
            "Macro-R": float(np.mean(rs) * 100.0),
            "Macro-F1": float(np.mean(fs) * 100.0),
            "Macro-AUC": float(macro_auc(y_true.tolist(), probs, n) * 100.0),
        })
    rows.append({
        "task": "five_task_mean",
        "Acc": "",
        "Macro-P": "",
        "Macro-R": "",
        "Macro-F1": float(np.mean([r["Macro-F1"] for r in rows])),
        "Macro-AUC": float(np.mean([r["Macro-AUC"] for r in rows])),
    })
    return rows


def probs_from_flat(flat_batch):
    task_probs = {}
    i = 0
    for task, labels in TASKS.items():
        n = len(labels)
        arr = flat_batch[:, i : i + n]
        arr = arr / np.clip(arr.sum(axis=1, keepdims=True), 1e-8, None)
        task_probs[task] = arr
        i += n
    return task_probs


def class_weights(y, n_classes):
    counts = Counter(y.tolist())
    vals = []
    for i in range(n_classes):
        vals.append((len(y) / max(1, counts[i])) ** 0.5)
    arr = np.asarray(vals, dtype=np.float32)
    arr = arr / np.clip(arr.mean(), 1e-8, None)
    return torch.tensor(arr, dtype=torch.float32)


class ContactHead(nn.Module):
    def __init__(self, input_dim=PROB_DIM, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, len(labels)) for task, labels in TASKS.items()})

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, 0, :]
        feat = self.net(x)
        return {task: head(feat) for task, head in self.heads.items()}


class TemporalHead(nn.Module):
    def __init__(self, input_dim=PROB_DIM, hidden=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, len(labels)) for task, labels in TASKS.items()})

    def forward(self, x):
        out, _ = self.gru(x)
        feat = out[:, -1, :]
        return {task: head(feat) for task, head in self.heads.items()}


def task_prob_dict_from_logits(logits):
    return {task: F.softmax(logits[task], dim=-1) for task in TASKS}


def reasoning_states(prob):
    iu = prob["image_usability"]
    sv = prob["structure_visibility"]
    kf = prob["keyframe_status"]
    defect = prob["primary_defect"]
    action = prob["recommended_action"]
    vals = torch.stack(
        [
            (iu[:, 0] + sv[:, 0] + kf[:, 0] + defect[:, 0] + action[:, 0]) / 5.0,
            (sv[:, 1] + kf[:, 1] + defect[:, 3] + action[:, 3]) / 4.0,
            (sv[:, 2] + kf[:, 2] + defect[:, 2] + action[:, 2]) / 4.0,
            (iu[:, 2] + kf[:, 2] + defect[:, 1] + action[:, 1]) / 4.0,
        ],
        dim=1,
    )
    return vals / torch.clamp(vals.sum(dim=1, keepdim=True), min=1e-6)


def soft_to_tensor(soft_np, device):
    return {task: torch.tensor(soft_np[task], dtype=torch.float32, device=device) for task in TASKS}


def decision_chain_loss(prob):
    loss = 0.0
    # Defect -> action relation: this is a strong but not sole action cue.
    defect_to_action = prob["primary_defect"]
    mapped_action = torch.stack(
        [
            defect_to_action[:, 0],
            defect_to_action[:, 1],
            defect_to_action[:, 2],
            defect_to_action[:, 3],
        ],
        dim=1,
    )
    loss = loss + F.kl_div(torch.log(torch.clamp(prob["recommended_action"], min=1e-6)), mapped_action.detach(), reduction="batchmean")
    # Visibility -> keyframe/defect relation.
    sv = prob["structure_visibility"]
    expected_no = sv[:, 2]
    expected_candidate = sv[:, 1] + 0.5 * sv[:, 3]
    expected_yes = sv[:, 0]
    kf_target = torch.stack([expected_yes, expected_candidate, expected_no], dim=1)
    kf_target = kf_target / torch.clamp(kf_target.sum(dim=1, keepdim=True), min=1e-6)
    loss = loss + F.kl_div(torch.log(torch.clamp(prob["keyframe_status"], min=1e-6)), kf_target.detach(), reduction="batchmean")
    return loss


def kg_relation_loss(prob):
    # Soft logical penalties over the clinical QC graph.
    iu = prob["image_usability"]
    sv = prob["structure_visibility"]
    kf = prob["keyframe_status"]
    defect = prob["primary_defect"]
    action = prob["recommended_action"]
    penalty = 0.0
    penalty = penalty + (sv[:, 2] * kf[:, 0]).mean()  # absent should not be yes keyframe
    penalty = penalty + (defect[:, 1] * action[:, 0]).mean()  # severe artifact should not hold
    penalty = penalty + (defect[:, 2] * action[:, 0]).mean()  # not visible should not hold
    penalty = penalty + (defect[:, 3] * action[:, 0]).mean()  # incomplete should not hold
    penalty = penalty + (defect[:, 0] * (1.0 - action[:, 0])).mean()  # none should map toward hold
    penalty = penalty + (iu[:, 2] * (kf[:, 0] + action[:, 0])).mean()  # unusable should not be hold/yes
    return penalty


def train_model(model, train_x, train_y, train_soft, val_x, val_y, variant, epochs, lr, seed, device):
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    weights = {task: class_weights(train_y[task], len(TASKS[task])).to(device) for task in TASKS}
    x_tensor = torch.tensor(train_x, dtype=torch.float32)
    soft_tensor = soft_to_tensor(train_soft, device)
    n = len(train_x)
    best_score = -1
    best_state = None
    stale = 0
    history = []
    cfg = {
        "hard": variant >= 1,
        "soft": variant >= 2,
        "r2s": variant >= 3,
        "chain": variant >= 4,
        "kg": variant >= 5,
    }
    loss_w = {"hard": 1.0, "soft": 0.35, "r2s": 0.20, "chain": 0.15, "kg": 0.10}
    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n)
        losses = []
        for start in range(0, n, 96):
            sel = idx[start : start + 96]
            sel_np = sel.numpy()
            xb = x_tensor[sel].to(device)
            logits = model(xb)
            prob = task_prob_dict_from_logits(logits)
            loss = 0.0
            if cfg["hard"]:
                hard_loss = 0.0
                for task in TASKS:
                    yb = torch.tensor(train_y[task][sel_np], dtype=torch.long, device=device)
                    hard_loss = hard_loss + F.cross_entropy(logits[task], yb, weight=weights[task])
                loss = loss + loss_w["hard"] * hard_loss / len(TASKS)
            if cfg["soft"]:
                soft_loss = 0.0
                for task in TASKS:
                    target = soft_tensor[task][sel].detach()
                    soft_loss = soft_loss + F.kl_div(F.log_softmax(logits[task], dim=-1), target, reduction="batchmean")
                loss = loss + loss_w["soft"] * soft_loss / len(TASKS)
            if cfg["r2s"]:
                target_soft = {task: soft_tensor[task][sel].detach() for task in TASKS}
                state_target = reasoning_states(target_soft).detach()
                state_pred = reasoning_states(prob)
                loss = loss + loss_w["r2s"] * F.kl_div(torch.log(torch.clamp(state_pred, min=1e-6)), state_target, reduction="batchmean")
            if cfg["chain"]:
                loss = loss + loss_w["chain"] * decision_chain_loss(prob)
            if cfg["kg"]:
                loss = loss + loss_w["kg"] * kg_relation_loss(prob)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics, _ = eval_model(model, val_x, val_y, device)
        val_score = next(r for r in val_metrics if r["task"] == "five_task_mean")["Macro-F1"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_mean_macro_f1": val_score})
        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 25 == 0:
            print(json.dumps({"event": "train", "variant": variant, "epoch": epoch, "val_mean_macro_f1": round(val_score, 2)}, ensure_ascii=False), flush=True)
        if stale >= 45:
            break
    model.load_state_dict(best_state)
    return model, history


def eval_model(model, x, y, device):
    model.eval()
    probs = {}
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        for task in TASKS:
            probs[task] = F.softmax(logits[task], dim=-1).detach().cpu().numpy()
    return compute_metrics(y, probs), probs


def write_task_metrics(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "task", "Acc", "Macro-P", "Macro-R", "Macro-F1", "Macro-AUC"])
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ["Acc", "Macro-P", "Macro-R", "Macro-F1", "Macro-AUC"]:
                if isinstance(out.get(key), float):
                    out[key] = f"{out[key]:.2f}"
            writer.writerow(out)


def write_summary(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "mean_macro_f1", "mean_macro_auc"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "variant": row["variant"],
                "mean_macro_f1": f"{row['mean_macro_f1']:.2f}",
                "mean_macro_auc": f"{row['mean_macro_auc']:.2f}",
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prob-cache", required=True)
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = read_jsonl(args.manifest)
    raw_row_count = len(rows)
    rows = [r for r in rows if str(r.get("label_source", "")).startswith("doctor_gold")]
    filtered_count = len(rows)
    rows = deduplicate_rows(rows)
    splits = {s: [r for r in rows if r.get("student_training", {}).get("split") == s] for s in ["train", "val", "test"]}
    cache = load_prob_cache(args.prob_cache)
    teacher_probs = load_teacher_probs(args.teacher_jsonl)

    missing_paths = sorted({p for r in rows for p in r["input_paths"] if p not in cache})
    if missing_paths:
        raise RuntimeError(f"Missing {len(missing_paths)} image probability rows, e.g. {missing_paths[:3]}")

    train_c, train_y, train_soft, _, train_hits = rows_to_arrays(splits["train"], cache, "contact_sheet", teacher_probs)
    val_c, val_y, val_soft, _, val_hits = rows_to_arrays(splits["val"], cache, "contact_sheet", teacher_probs)
    test_c, test_y, test_soft, test_meta, test_hits = rows_to_arrays(splits["test"], cache, "contact_sheet", teacher_probs)
    train_g, train_y_g, train_soft_g, _, train_hits_g = rows_to_arrays(splits["train"], cache, "global_ordered", teacher_probs)
    val_g, val_y_g, val_soft_g, _, val_hits_g = rows_to_arrays(splits["val"], cache, "global_ordered", teacher_probs)
    test_g, test_y_g, test_soft_g, _, test_hits_g = rows_to_arrays(splits["test"], cache, "global_ordered", teacher_probs)

    summary = []
    task_rows = []
    histories = {}

    def add_variant(name, metrics):
        mean = next(r for r in metrics if r["task"] == "five_task_mean")
        summary.append({"variant": name, "mean_macro_f1": mean["Macro-F1"], "mean_macro_auc": mean["Macro-AUC"]})
        for row in metrics:
            task_rows.append({"variant": name, **row})
        print(json.dumps({"event": "evaluated", "variant": name, "mean_macro_f1": round(mean["Macro-F1"], 2), "mean_macro_auc": round(mean["Macro-AUC"], 2)}, ensure_ascii=False), flush=True)

    # S0: frozen contact-sheet student probabilities.
    add_variant("S0_InternVL3.5-2B_contact_frozen", compute_metrics(test_y, probs_from_flat(test_c[:, 0, :])))

    variants = [
        ("S1_Hard_pseudo_label", 1, ContactHead(), train_c, train_y, train_soft, val_c, val_y, test_c, test_y, 1e-3),
        ("S2_Soft_KD", 2, ContactHead(), train_c, train_y, train_soft, val_c, val_y, test_c, test_y, 8e-4),
        ("S3_Soft_KD_R2S", 3, ContactHead(), train_c, train_y, train_soft, val_c, val_y, test_c, test_y, 8e-4),
        ("S4_Soft_KD_R2S_Chain", 4, ContactHead(), train_c, train_y, train_soft, val_c, val_y, test_c, test_y, 8e-4),
        ("S5_Soft_KD_R2S_Chain_KG", 5, ContactHead(), train_c, train_y, train_soft, val_c, val_y, test_c, test_y, 7e-4),
        ("S6_Full_ThyQC_GT_QDM", 5, TemporalHead(), train_g, train_y_g, train_soft_g, val_g, val_y_g, test_g, test_y_g, 8e-4),
    ]
    for idx, (name, level, model, tr_x, tr_y, tr_soft, va_x, va_y, te_x, te_y, lr) in enumerate(variants, start=1):
        trained, history = train_model(model, tr_x, tr_y, tr_soft, va_x, va_y, level, args.epochs, lr, args.seed + idx, device)
        metrics, probs = eval_model(trained, te_x, te_y, device)
        histories[name] = history
        add_variant(name, metrics)

    write_summary(out_dir / "gt_qdm_ablation_summary.csv", summary)
    write_task_metrics(out_dir / "gt_qdm_ablation_task_metrics.csv", task_rows)
    with open(out_dir / "gt_qdm_ablation_macro_f1_wide.csv", "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["variant", "mean_macro_f1"] + TASK_NAMES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        by_variant = {}
        for row in task_rows:
            by_variant.setdefault(row["variant"], {})[row["task"]] = row["Macro-F1"]
        for item in summary:
            row = {"variant": item["variant"], "mean_macro_f1": f"{item['mean_macro_f1']:.2f}"}
            for task in TASK_NAMES:
                row[task] = f"{by_variant[item['variant']][task]:.2f}"
            writer.writerow(row)
    report = {
        "raw_row_count": raw_row_count,
        "doctor_gold_row_count_before_dedup": filtered_count,
        "deduplicated_row_count": len(rows),
        "splits": {k: len(v) for k, v in splits.items()},
        "teacher_probability_coverage": {
            "train_contact": f"{train_hits}/{len(splits['train'])}",
            "val_contact": f"{val_hits}/{len(splits['val'])}",
            "test_contact": f"{test_hits}/{len(splits['test'])}",
            "train_global_ordered": f"{train_hits_g}/{len(splits['train'])}",
            "val_global_ordered": f"{val_hits_g}/{len(splits['val'])}",
            "test_global_ordered": f"{test_hits_g}/{len(splits['test'])}",
        },
        "summary": summary,
        "histories": histories,
        "note": "Physician labels are the hard supervision anchor; teacher probabilities are mixed as soft dark knowledge when available.",
    }
    with open(out_dir / "gt_qdm_ablation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / "gt_qdm_ablation_report.md", "w", encoding="utf-8") as f:
        f.write("# GT-QDM Student Multi-Level Distillation Ablation\n\n")
        f.write(f"- raw rows: {raw_row_count}\n")
        f.write(f"- doctor-gold rows before dedup: {filtered_count}\n")
        f.write(f"- deduplicated rows: {len(rows)}\n")
        f.write(f"- splits: {report['splits']}\n")
        f.write(f"- teacher probability coverage: {report['teacher_probability_coverage']}\n\n")
        f.write("| Variant | Mean Macro-F1 | Mean Macro-AUC |\n|---|---:|---:|\n")
        for row in summary:
            f.write(f"| {row['variant']} | {row['mean_macro_f1']:.2f} | {row['mean_macro_auc']:.2f} |\n")
    print(json.dumps({"event": "done", "out_dir": str(out_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
