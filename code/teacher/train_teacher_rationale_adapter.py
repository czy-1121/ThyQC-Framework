"""Continue the seed-42 QC teacher LoRA with short rationale generation SFT.

This writes a separate adapter. The original classification LoRA, heads, and
probability files remain unchanged. Loss is computed only on assistant tokens.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patch_processor(processor, config):
    if getattr(processor, "patch_size", None) is None and hasattr(config, "vision_config"):
        processor.patch_size = getattr(config.vision_config, "patch_size", None)
    processor.vision_feature_select_strategy = getattr(
        config, "vision_feature_select_strategy", "default"
    )
    processor.num_additional_image_tokens = 1
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def split_rows(rows, seed, validation_fraction=0.10):
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    n_validation = max(1, round(len(ordered) * validation_fraction))
    return ordered[n_validation:], ordered[:n_validation]


def chat_text(processor, user_prompt, target_json=None, include_image=True):
    user_content = []
    if include_image:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": user_prompt})
    messages = [{
        "role": "user",
        "content": user_content,
    }]
    if target_json is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": target_json}],
        })
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=target_json is None,
    )


def assistant_only_labels(full_input_ids, full_attention_mask, prompt_token_count):
    labels = full_input_ids.clone()
    labels[full_attention_mask == 0] = -100
    labels[:, :prompt_token_count] = -100
    return labels


class RationaleDataset(Dataset):
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class Collator:
    def __init__(self, processor, text_only_training=False):
        self.processor = processor
        self.text_only_training = text_only_training

    def __call__(self, batch):
        if len(batch) != 1:
            raise ValueError("Rationale SFT uses batch_size=1 for exact assistant masking")
        row = batch[0]
        include_image = not self.text_only_training
        prompt_text = chat_text(
            self.processor, row["user_prompt"], None, include_image=include_image
        )
        full_text = chat_text(
            self.processor,
            row["user_prompt"],
            row["target_json"],
            include_image=include_image,
        )
        if self.text_only_training:
            prompt_inputs = self.processor.tokenizer(
                [prompt_text], padding=False, return_tensors="pt"
            )
            full_inputs = self.processor.tokenizer(
                [full_text], padding=False, return_tensors="pt"
            )
        else:
            image = Image.open(row["image_path"]).convert("RGB")
            prompt_inputs = self.processor(
                text=[prompt_text], images=[image], padding=False, return_tensors="pt"
            )
            full_inputs = self.processor(
                text=[full_text], images=[image], padding=False, return_tensors="pt"
            )
        prompt_count = int(prompt_inputs["attention_mask"].sum().item())
        full_inputs["labels"] = assistant_only_labels(
            full_inputs["input_ids"], full_inputs["attention_mask"], prompt_count
        )
        return full_inputs


@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    losses = []
    for batch in loader:
        batch = {key: value.to("cuda:0") for key, value in batch.items()}
        outputs = model(**batch, use_cache=False)
        losses.append(float(outputs.loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--initial-adapter", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--text-only-training",
        action="store_true",
        help="Freeze visual adapters and train only rationale language organization.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )

    rows = read_jsonl(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    train_rows, validation_rows = split_rows(
        rows, args.seed, args.validation_fraction
    )
    (out_dir / "split_report.json").write_text(json.dumps({
        "train": len(train_rows),
        "validation": len(validation_rows),
        "seed": args.seed,
    }, indent=2), encoding="utf-8")

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    processor = patch_processor(
        AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True), config
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_dir,
        quantization_config=quantization,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model = PeftModel.from_pretrained(
        model, args.initial_adapter, is_trainable=True
    )
    model.enable_input_require_grads()
    if args.text_only_training:
        for name, parameter in model.named_parameters():
            if "vision_tower" in name:
                parameter.requires_grad = False
    model.print_trainable_parameters()
    trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or not any("lora_" in name for name in trainable):
        raise RuntimeError("No trainable LoRA parameters found in rationale adapter")

    collator = Collator(processor, text_only_training=args.text_only_training)
    train_loader = DataLoader(
        RationaleDataset(train_rows),
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        RationaleDataset(validation_rows),
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_validation = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        started = time.time()
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to("cuda:0") for key, value in batch.items()}
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            if epoch == 1 and step == 1:
                gradient_names = [
                    name for name, parameter in model.named_parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if not gradient_names or not any("lora_" in name for name in gradient_names):
                    raise RuntimeError("First rationale-SFT backward pass produced no LoRA gradients")
                print(json.dumps({
                    "event": "gradient_gate_passed",
                    "trainable_parameters": len(trainable),
                    "parameters_with_gradients": len(gradient_names),
                }), flush=True)
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(outputs.loss.detach().cpu())
            if step % args.log_every == 0:
                print(json.dumps({
                    "event": "train",
                    "epoch": epoch,
                    "step": step,
                    "steps": len(train_loader),
                    "loss": running_loss / step,
                    "elapsed_sec": round(time.time() - started, 1),
                }), flush=True)

        validation_loss = evaluate_loss(model, validation_loader)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            "validation_loss": validation_loss,
            "elapsed_sec": round(time.time() - started, 1),
        }
        history.append(epoch_record)
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        model.save_pretrained(out_dir / f"epoch_{epoch}_adapter")
        if validation_loss < best_validation:
            best_validation = validation_loss
            model.save_pretrained(out_dir / "best_rationale_adapter")
            (out_dir / "best_metrics.json").write_text(
                json.dumps(epoch_record, indent=2), encoding="utf-8"
            )
        print(json.dumps({"event": "epoch", **epoch_record}), flush=True)

    print(json.dumps({
        "event": "done",
        "best_validation_loss": best_validation,
        "best_adapter": str(out_dir / "best_rationale_adapter"),
    }), flush=True)


if __name__ == "__main__":
    main()
