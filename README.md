# ThyQC

Official implementation of **ThyQC**, a multi-task quality-control framework for thyroid ultrasound. ThyQC transfers task probabilities and structured semantic knowledge from a generative teacher to a compact discriminative student through G2D-UOT, followed by GT-QDM temporal refinement.

## Installation

```bash
git clone https://github.com/czy-1121/ThyQC-Reproducible.git
cd ThyQC-Reproducible
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The student backbone is [OpenGVLab/InternVL3_5-2B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF).

## Data preparation

The training manifest is a JSONL file containing the sample identifier, input image paths, five task labels, data split, and label source. The teacher package is a JSONL file containing task probabilities and short task-wise rationales. Path templates and training settings are provided in `configs/`.

The clinical dataset is available under the corresponding institutional data-use requirements. The images in `data/anonymized_examples/` are de-identified examples illustrating the input format.

## Training

### 1. G2D-UOT student training

```bash
python code/train_g2d_uot_thyqc_20260919.py \
  --model-path OpenGVLab/InternVL3_5-2B-HF \
  --manifest <manifest.jsonl> \
  --teacher-jsonl <teacher_package.jsonl> \
  --out-dir outputs/seed42/stage1 \
  --seed 42 \
  --validation-only
```

### 2. Probability cache

```bash
python code/build_g2d_gt_qdm_prob_cache_20260919.py \
  --model-path OpenGVLab/InternVL3_5-2B-HF \
  --checkpoint outputs/seed42/stage1/best_trainable_state.pt \
  --manifest <manifest.jsonl> \
  --out outputs/seed42/student_image_prob_cache.jsonl \
  --seed 42
```

### 3. GT-QDM training

```bash
python code/train_g2d_uot_gt_qdm_seed42_20260919.py \
  --manifest <manifest.jsonl> \
  --prob-cache outputs/seed42/student_image_prob_cache.jsonl \
  --teacher-jsonl <teacher_package.jsonl> \
  --out-dir outputs/seed42/gt_qdm \
  --seed 42 \
  --validation-only
```

Use seeds `41`, `42`, and `43` for the multi-seed experiment.

## Evaluation

Evaluate the validation-selected checkpoint without further optimization:

```bash
python code/train_g2d_uot_gt_qdm_seed42_20260919.py \
  --manifest <manifest.jsonl> \
  --prob-cache outputs/seed42/student_image_prob_cache.jsonl \
  --teacher-jsonl <teacher_package.jsonl> \
  --out-dir outputs/seed42/evaluation \
  --init-state outputs/seed42/gt_qdm/best_gt_qdm_state.pt \
  --seed 42 \
  --epochs 0
```

The task-level metrics are written to `best_test_metrics_at_val_best.json`. Reference aggregate metrics are provided in `results/metrics_summary.csv`. A reproduction within **2 percentage points** of the reference result is considered consistent across supported environments.

## Model weights

See `weights/README.md` for backbone and ThyQC checkpoint instructions.

## Citation

Citation information will be added with the paper release.

## License

This repository is released under the MIT License and is intended for research use. Third-party models and datasets remain subject to their original licenses and access conditions.
