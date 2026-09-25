"""Build image-probability features for the GT-QDM stage."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_student_gt_qdm_ablation_20260809 as temporal
import train_thyroid_internvl_student_kgkd_v1_22_20260718 as base


@torch.no_grad()
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base.set_seed(args.seed)
    rows = temporal.deduplicate_rows([
        row for row in temporal.read_jsonl(args.manifest)
        if str(row.get("label_source", "")).startswith("doctor_gold")
    ])
    image_paths = sorted({path for row in rows for path in row["input_paths"]})
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    model = base.InternVLQCClassifier(
        args.model_path, lora_rank=16, lora_alpha=32
    ).cuda().eval()
    load_report = base.load_trainable_state(model, args.checkpoint)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(image_paths), args.batch_size):
            paths = image_paths[start:start + args.batch_size]
            images = [Image.open(path).convert("RGB") for path in paths]
            queries = [f"{processor.image_token}\n{base.PROMPT}" for _ in paths]
            tokens = processor(
                images=images, text=queries, return_tensors="pt", padding=True
            )
            logits = model({
                key: tokens[key]
                for key in ("input_ids", "attention_mask", "pixel_values")
            })
            probabilities = {
                task: F.softmax(logits[task], dim=-1).cpu().float()
                for task in base.TASKS
            }
            for index, path in enumerate(paths):
                record = {
                    "path": path,
                    "probs": {
                        task: probabilities[task][index].tolist()
                        for task in base.TASKS
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if start == 0 or (start // args.batch_size + 1) % 20 == 0:
                print(json.dumps({
                    "event": "cache_progress",
                    "done": start + len(paths),
                    "total": len(image_paths),
                    "load": load_report,
                }), flush=True)

    print(json.dumps({
        "event": "done",
        "images": len(image_paths),
        "seconds": time.perf_counter() - started,
        "out": str(output),
        "load": load_report,
    }), flush=True)


if __name__ == "__main__":
    run()
