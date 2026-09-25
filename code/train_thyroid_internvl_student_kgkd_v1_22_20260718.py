import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoProcessor


TASKS = {
    "image_usability": ["acceptable", "limited", "unacceptable"],
    "structure_visibility": ["adequate", "partial", "absent", "uncertain"],
    "keyframe_status": ["yes", "candidate", "no"],
    "primary_defect": ["none", "severe_artifact", "thyroid_not_visible", "target_incomplete"],
    "recommended_action": ["hold", "repeat_acquisition", "reacquire_target", "adjust_angle_and_rescan"],
}

DEFECT_GRAPH = {
    "none": {"image_usability": "acceptable", "keyframe_status": "yes", "recommended_action": "hold"},
    "severe_artifact": {"image_usability": "unacceptable", "keyframe_status": "no", "recommended_action": "repeat_acquisition"},
    "thyroid_not_visible": {"image_usability": "acceptable", "keyframe_status": "no", "recommended_action": "reacquire_target"},
    "target_incomplete": {"image_usability": "limited", "keyframe_status": "candidate", "recommended_action": "adjust_angle_and_rescan"},
}

STRUCTURE_TO_DEFECT = {
    "adequate": "none",
    "partial": "target_incomplete",
    "absent": "thyroid_not_visible",
    "uncertain": "target_incomplete",
}

STRUCTURE_TO_ACTION = {
    "adequate": "hold",
    "partial": "adjust_angle_and_rescan",
    "absent": "reacquire_target",
    "uncertain": "adjust_angle_and_rescan",
}

USABILITY_TO_ACTION = {
    "acceptable": "hold",
    "limited": "adjust_angle_and_rescan",
    "unacceptable": "repeat_acquisition",
}

KEYFRAME_TO_ACTION = {
    "yes": "hold",
    "candidate": "adjust_angle_and_rescan",
    "no": "repeat_acquisition",
}

PROMPT = (
    "Thyroid ultrasound QC only. Inspect the contact sheet visually. "
    "First judge image usability, then thyroid structure visibility, then keyframe status, "
    "main QC defect, and acquisition action. Do not infer diagnosis or TI-RADS."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_manifest(path, max_images=1):
    rows = []
    for row in read_jsonl(path):
        labels = {}
        for task, vocab in TASKS.items():
            value = row["student_label"][task]
            if value not in vocab:
                raise ValueError(f"Illegal label for {task}: {value}")
            labels[task] = vocab.index(value)
        conf = teacher_confidence(row)
        soft_targets = {task: make_soft_target(task, labels[task], conf) for task in TASKS}
        paths = row.get("input_paths") or [row.get("input_path")]
        rows.append(
            {
                "review_id": row.get("review_id", ""),
                "sample_id": row["sample_id"],
                "case_id": row.get("case_id", ""),
                "image_path": paths[:max_images][0],
                "labels": labels,
                "soft_targets": soft_targets,
                "teacher_confidence": conf,
                "split": row.get("student_training", {}).get("split", row.get("split", "")),
                "label_source": row.get("label_source", ""),
            }
        )
    return rows


def teacher_confidence(row):
    if row.get("doctor_review_applied") is True:
        return 0.98
    source = str(row.get("label_source", "")).lower()
    priority = str(row.get("teacher_review_priority", "")).lower()
    if "doctor" in source:
        base = 0.88
    elif "teacher" in source:
        base = 0.78
    else:
        base = 0.70
    if priority == "high":
        base -= 0.08
    elif priority == "medium":
        base -= 0.04
    elif priority == "low":
        base += 0.04
    return float(min(0.98, max(0.55, base)))


def make_soft_target(task, label_idx, confidence):
    n = len(TASKS[task])
    eps = max(0.02, 1.0 - confidence)
    vals = [eps / max(1, n - 1) for _ in range(n)]
    vals[label_idx] = 1.0 - eps
    return vals


def build_transform(input_size):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        images.append(resized_img.crop(box))
    if use_thumbnail and len(images) != 1:
        images.append(image.resize((image_size, image_size)))
    return images


class ThyroidDataset(Dataset):
    def __init__(self, rows, transform=None, max_num=6):
        self.rows = rows
        self.transform = transform
        self.max_num = max_num

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        labels = {task: torch.tensor(value, dtype=torch.long) for task, value in row["labels"].items()}
        soft_targets = {
            task: torch.tensor(row["soft_targets"][task], dtype=torch.float32)
            for task in TASKS
        }
        return {
            "review_id": row["review_id"],
            "sample_id": row["sample_id"],
            "case_id": row["case_id"],
            "image": image,
            "labels": labels,
            "soft_targets": soft_targets,
            "teacher_confidence": torch.tensor(row["teacher_confidence"], dtype=torch.float32),
        }


def make_query(tokenizer, model, question, num_patches):
    if "<image>" not in question:
        question = "<image>\n" + question
    template = model.conv_template.copy()
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
    query = query.replace("<image>", image_tokens, 1)
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    return query, eos_token_id


class Collator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"

    def __call__(self, batch):
        queries = [f"{self.processor.image_token}\n{PROMPT}" for _ in batch]
        images = [item["image"] for item in batch]
        labels = {task: [] for task in TASKS}
        soft_targets = {task: [] for task in TASKS}
        teacher_confidence = []
        meta = []
        for item in batch:
            for task in TASKS:
                labels[task].append(item["labels"][task])
                soft_targets[task].append(item["soft_targets"][task])
            teacher_confidence.append(item["teacher_confidence"])
            meta.append({k: item[k] for k in ["review_id", "sample_id", "case_id"]})
        toks = self.processor(images=images, text=queries, return_tensors="pt", padding=True)
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "pixel_values": toks["pixel_values"],
            "labels": {task: torch.stack(vals) for task, vals in labels.items()},
            "soft_targets": {task: torch.stack(vals) for task, vals in soft_targets.items()},
            "teacher_confidence": torch.stack(teacher_confidence),
            "meta": meta,
        }


class InternVLQCClassifier(nn.Module):
    def __init__(self, model_path, lora_rank=8, lora_alpha=16):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.backbone.config.use_cache = False
        self.backbone.img_context_token_id = None
        vision_module = getattr(self.backbone, "vision_model", None) or getattr(self.backbone, "vision_tower", None)
        for param in vision_module.parameters():
            param.requires_grad = False
        hidden = self.backbone.language_model.config.hidden_size
        lora = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        # InternVLChatModel has a custom multimodal forward signature. LoRA must be
        # attached to the inner language model so the outer visual-token injection
        # path remains unchanged.
        self.backbone.language_model = get_peft_model(self.backbone.language_model, lora)
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, len(labels)) for task, labels in TASKS.items()})

    def forward(self, batch):
        device = next(self.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
        outputs = self.backbone(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs.hidden_states[-1]
        pooled = hidden[:, -1]
        return {task: head(pooled.float()) for task, head in self.heads.items()}


def build_sampler(rows, class_weight):
    if class_weight <= 0:
        return None
    combo_counts = Counter(tuple(row["labels"][task] for task in TASKS) for row in rows)
    weights = []
    for row in rows:
        combo = tuple(row["labels"][task] for task in TASKS)
        weights.append((1.0 / combo_counts[combo]) ** class_weight)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def rank_auc_binary(y_true, scores):
    pos = [s for y, s in zip(y_true, scores) if y == 1]
    neg = [s for y, s in zip(y_true, scores) if y == 0]
    if not pos or not neg:
        return None
    values = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j][0] == values[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(v[1] for v in values[i:j])
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def macro_auc(y_true, probs, n_classes):
    aucs = []
    for cls in range(n_classes):
        y_bin = [1 if y == cls else 0 for y in y_true]
        auc = rank_auc_binary(y_bin, [row[cls] for row in probs])
        if auc is not None:
            aucs.append(auc)
    return float(sum(aucs) / len(aucs)) if aucs else 0.0


def compute_metrics(y_true_by_task, y_pred_by_task, probs_by_task):
    out = {}
    for task, labels in TASKS.items():
        n = len(labels)
        cm = np.zeros((n, n), dtype=int)
        y_true = y_true_by_task[task]
        y_pred = y_pred_by_task[task]
        for y, p in zip(y_true, y_pred):
            cm[y, p] += 1
        ps, rs, fs = [], [], []
        for i in range(n):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            ps.append(p)
            rs.append(r)
            fs.append(f1)
        out[task] = {
            "acc": float((np.array(y_true) == np.array(y_pred)).mean()),
            "macro_p": float(np.mean(ps)),
            "macro_r": float(np.mean(rs)),
            "macro_f1": float(np.mean(fs)),
            "macro_auc": macro_auc(y_true, probs_by_task[task], n),
            "confusion_matrix": cm.tolist(),
        }
    out["mean_macro_f1"] = float(np.mean([out[t]["macro_f1"] for t in TASKS]))
    out["mean_macro_auc"] = float(np.mean([out[t]["macro_auc"] for t in TASKS]))
    return out


def project_probs_from_defect(probs_by_task):
    out = {task: [list(row) for row in rows] for task, rows in probs_by_task.items()}
    defect_probs = probs_by_task["primary_defect"]
    for i, defect_row in enumerate(defect_probs):
        for task in ["image_usability", "keyframe_status", "recommended_action"]:
            arr = [0.0 for _ in TASKS[task]]
            for defect_idx, prob in enumerate(defect_row):
                defect_label = TASKS["primary_defect"][defect_idx]
                mapped = DEFECT_GRAPH[defect_label][task]
                arr[TASKS[task].index(mapped)] += prob
            out[task][i] = arr
    return out


@torch.no_grad()
def evaluate(model, loader, measure_timing=False):
    model.eval()
    y_true = defaultdict(list)
    y_pred = defaultdict(list)
    probs = defaultdict(list)
    records = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if measure_timing:
            torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for batch in loader:
        logits = model(batch)
        for task in TASKS:
            labels = batch["labels"][task].tolist()
            p = F.softmax(logits[task], dim=-1).detach().cpu().float().numpy().tolist()
            pred = np.argmax(np.array(p), axis=1).tolist()
            y_true[task].extend(labels)
            y_pred[task].extend(pred)
            probs[task].extend(p)
        for row_idx, meta in enumerate(batch["meta"]):
            rec = dict(meta)
            rec["true"] = {task: TASKS[task][y_true[task][-len(batch["meta"]) + row_idx]] for task in TASKS}
            rec["pred"] = {task: TASKS[task][y_pred[task][-len(batch["meta"]) + row_idx]] for task in TASKS}
            records.append(rec)
    metrics = compute_metrics(y_true, y_pred, probs)
    graph_probs = project_probs_from_defect(probs)
    graph_pred = {task: np.argmax(np.array(graph_probs[task]), axis=1).tolist() for task in TASKS}
    graph_metrics = compute_metrics(y_true, graph_pred, graph_probs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_sec = time.perf_counter() - start
    timing = {
        "wall_sec": wall_sec,
        "num_samples": len(records),
        "sec_per_sample": wall_sec / max(len(records), 1),
        "samples_per_sec": len(records) / wall_sec if wall_sec > 0 else 0.0,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0,
        "peak_vram_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else 0.0,
    }
    return metrics, graph_metrics, records, timing


def write_metrics_csv(path, metrics):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "Acc", "Macro-P", "Macro-R", "Macro-F1", "Macro-AUC"])
        writer.writeheader()
        for task in TASKS:
            m = metrics[task]
            writer.writerow(
                {
                    "task": task,
                    "Acc": f"{m['acc'] * 100:.2f}",
                    "Macro-P": f"{m['macro_p'] * 100:.2f}",
                    "Macro-R": f"{m['macro_r'] * 100:.2f}",
                    "Macro-F1": f"{m['macro_f1'] * 100:.2f}",
                    "Macro-AUC": f"{m['macro_auc'] * 100:.2f}",
                }
            )
        writer.writerow(
            {
                "task": "five_task_mean",
                "Acc": "",
                "Macro-P": "",
                "Macro-R": "",
                "Macro-F1": f"{metrics['mean_macro_f1'] * 100:.2f}",
                "Macro-AUC": f"{metrics['mean_macro_auc'] * 100:.2f}",
            }
        )


def trainable_state_dict(model):
    # Save every actually trainable tensor. InternVL keeps the multi-modal
    # projector trainable in addition to LoRA and classifier heads; omitting it
    # makes reloaded checkpoints fail even when in-memory evaluation is strong.
    state = model.state_dict()
    keep = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            keep[name] = state[name].detach().cpu()
    return keep


def load_trainable_state(model, path):
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {"missing": len(missing), "unexpected": len(unexpected)}


def set_heads_only_trainable(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("heads.")
    return {
        "trainable_params_after_freeze": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params_after_freeze": sum(p.numel() for p in model.parameters()),
    }


def task_one_hot(labels, task):
    return F.one_hot(labels, num_classes=len(TASKS[task])).float()


def map_distribution(source_probs, source_task, target_task, mapping):
    device = source_probs.device
    mat = torch.zeros(len(TASKS[source_task]), len(TASKS[target_task]), device=device)
    for src_label, dst_label in mapping.items():
        mat[TASKS[source_task].index(src_label), TASKS[target_task].index(dst_label)] = 1.0
    return source_probs @ mat


def normalized_mix(parts):
    out = None
    for weight, tensor in parts:
        out = tensor * weight if out is None else out + tensor * weight
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def kd_loss_for_task(logits, soft_targets, teacher_confidence, temperature):
    logp = F.log_softmax(logits / temperature, dim=-1)
    target = soft_targets.to(logits.device)
    sample_loss = F.kl_div(logp, target, reduction="none").sum(dim=-1) * (temperature ** 2)
    return (sample_loss * teacher_confidence.to(logits.device)).mean()


def confidence_weighted_ce(logits, labels, teacher_confidence):
    sample_loss = F.cross_entropy(logits, labels, reduction="none")
    weights = teacher_confidence.to(logits.device)
    return (sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)


def clinical_chain_consistency_loss(logits):
    probs = {task: F.softmax(logits[task], dim=-1) for task in TASKS}
    action_from_defect = map_distribution(probs["primary_defect"], "primary_defect", "recommended_action", {
        defect: DEFECT_GRAPH[defect]["recommended_action"] for defect in DEFECT_GRAPH
    })
    action_from_structure = map_distribution(probs["structure_visibility"], "structure_visibility", "recommended_action", STRUCTURE_TO_ACTION)
    action_from_usability = map_distribution(probs["image_usability"], "image_usability", "recommended_action", USABILITY_TO_ACTION)
    action_from_keyframe = map_distribution(probs["keyframe_status"], "keyframe_status", "recommended_action", KEYFRAME_TO_ACTION)
    action_target = normalized_mix(
        [
            (0.40, action_from_defect),
            (0.25, action_from_structure),
            (0.20, action_from_usability),
            (0.15, action_from_keyframe),
        ]
    ).detach()
    action_loss = F.kl_div(F.log_softmax(logits["recommended_action"], dim=-1), action_target, reduction="batchmean")

    defect_from_structure = map_distribution(probs["structure_visibility"], "structure_visibility", "primary_defect", STRUCTURE_TO_DEFECT)
    defect_target = normalized_mix([(0.70, defect_from_structure), (0.30, probs["primary_defect"])]).detach()
    defect_loss = F.kl_div(F.log_softmax(logits["primary_defect"], dim=-1), defect_target, reduction="batchmean")

    return action_loss + 0.5 * defect_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
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
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--log-every", type=int, default=40)
    parser.add_argument("--resume-trainable", default="")
    parser.add_argument("--kd-weight", type=float, default=0.25)
    parser.add_argument("--kg-consistency-weight", type=float, default=0.10)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--train-heads-only", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.manifest, max_images=args.max_images)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    print(
        json.dumps(
            {"event": "loaded_rows", "train": len(train_rows), "val": len(val_rows), "test": len(test_rows), "out_dir": str(out_dir)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = InternVLQCClassifier(args.model_path, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha).cuda()
    if args.resume_trainable:
        load_info = load_trainable_state(model, args.resume_trainable)
        print(json.dumps({"event": "loaded_trainable_state", "path": args.resume_trainable, **load_info}, ensure_ascii=False), flush=True)
    if args.train_heads_only:
        freeze_info = set_heads_only_trainable(model)
        print(json.dumps({"event": "train_heads_only", **freeze_info}, ensure_ascii=False), flush=True)
    collator = Collator(processor)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_ds = ThyroidDataset(train_rows, None, args.max_num)
    val_ds = ThyroidDataset(val_rows, None, args.max_num)
    test_ds = ThyroidDataset(test_rows, None, args.max_num)
    sampler = build_sampler(train_rows, args.class_weight_power)
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
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_kg = 0.0
        for step, batch in enumerate(train_loader, 1):
            logits = model(batch)
            teacher_conf = batch["teacher_confidence"].cuda()
            ce_loss = sum(
                confidence_weighted_ce(logits[task], batch["labels"][task].cuda(), teacher_conf)
                for task in TASKS
            ) / len(TASKS)
            kd_loss = sum(
                kd_loss_for_task(logits[task], batch["soft_targets"][task], teacher_conf, args.kd_temperature)
                for task in TASKS
            ) / len(TASKS)
            kg_loss = clinical_chain_consistency_loss(logits)
            loss = ce_loss + args.kd_weight * kd_loss + args.kg_consistency_weight * kg_loss
            (loss / args.grad_accum).backward()
            total_loss += float(loss.detach().cpu())
            total_ce += float(ce_loss.detach().cpu())
            total_kd += float(kd_loss.detach().cpu())
            total_kg += float(kg_loss.detach().cpu())
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "loss": round(total_loss / step, 4),
                            "ce": round(total_ce / step, 4),
                            "kd": round(total_kd / step, 4),
                            "kg": round(total_kg / step, 4),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        val_metrics, val_graph_metrics, _, _ = evaluate(model, val_loader)
        metrics, graph_metrics, records, timing = evaluate(model, test_loader, measure_timing=True)
        history_row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_loader)),
            "train_ce": total_ce / max(1, len(train_loader)),
            "train_kd": total_kd / max(1, len(train_loader)),
            "train_kg": total_kg / max(1, len(train_loader)),
            "val_mean_macro_f1": val_metrics["mean_macro_f1"],
            "val_A5_mean_macro_f1": val_graph_metrics["mean_macro_f1"],
            "test_mean_macro_f1": metrics["mean_macro_f1"],
            "test_A5_mean_macro_f1": graph_metrics["mean_macro_f1"],
            "test_mean_macro_auc": metrics["mean_macro_auc"],
            "test_sec_per_sample": timing["sec_per_sample"],
            "test_fps": timing["samples_per_sec"],
        }
        history.append(history_row)
        print(
            json.dumps(
                {
                    "event": "epoch_eval",
                    "epoch": epoch,
                    "train_loss": round(history_row["train_loss"], 4),
                    "train_ce": round(history_row["train_ce"], 4),
                    "train_kd": round(history_row["train_kd"], 4),
                    "train_kg": round(history_row["train_kg"], 4),
                    "val_mean_macro_f1": history_row["val_mean_macro_f1"],
                    "test_mean_macro_f1": history_row["test_mean_macro_f1"],
                    "test_mean_macro_auc": history_row["test_mean_macro_auc"],
                    "test_sec_per_sample": history_row["test_sec_per_sample"],
                    "test_fps": history_row["test_fps"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if val_metrics["mean_macro_f1"] > best:
            best = val_metrics["mean_macro_f1"]
            torch.save(trainable_state_dict(model), out_dir / "best_trainable_state.pt")
            (out_dir / "best_test_metrics_at_val_best.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "best_A5_graph_metrics.json").write_text(json.dumps(graph_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            write_metrics_csv(out_dir / "best_test_metrics_at_val_best.csv", metrics)
            write_metrics_csv(out_dir / "best_A5_graph_metrics.csv", graph_metrics)
            (out_dir / "efficiency.json").write_text(
                json.dumps(
                    {
                        "model_name": "InternVL3.5-2B-HF",
                        "parameter_count_total": total_params,
                        "parameter_count_trainable": trainable_params,
                        "test_timing": timing,
                        "best_epoch_by_val": epoch,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            with (out_dir / "best_test_predictions.jsonl").open("w", encoding="utf-8") as handle:
                for rec in records:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    last_metrics, _, last_records, last_timing = evaluate(model, test_loader, measure_timing=True)
    write_metrics_csv(out_dir / "last_test_metrics.csv", last_metrics)
    (out_dir / "last_test_metrics.json").write_text(json.dumps(last_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "last_efficiency.json").write_text(json.dumps({"test_timing": last_timing}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "last_test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for rec in last_records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
