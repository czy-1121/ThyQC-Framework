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

## Reproduction

The repository includes de-identified evaluation images, labels, three ThyQC checkpoints, and the cached student probabilities required by GT-QDM. Run the bundled evaluation with:

```bash
python code/evaluate_public.py --all-seeds
```

Predictions and task-level metrics are written to `results/reproduced/`. Small numerical differences across supported PyTorch environments are expected; values within 2 percentage points are considered consistent.

## Data format

The released evaluation data are in `data/anonymized_test/`. Training manifests use JSONL records containing input paths, five task labels, data split, and label source. Teacher knowledge packages contain five-task probabilities and short task-wise rationales.

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

## Model weights

The released ThyQC weights are stored with Git LFS under `weights/thyqc/`. Teacher training and structured-rationale code is provided in `code/teacher/`; teacher weights and the private retrieval corpus are not distributed.

## Citation

Citation information will be added with the paper release.

## License

This repository is released under the MIT License and is intended for research use. Third-party models and datasets remain subject to their original licenses and access conditions.
