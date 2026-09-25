# ThyQC

ThyQC is a thyroid-ultrasound quality-control framework that transfers structured teacher knowledge to a compact student model. This public release contains the current G2D-UOT + GT-QDM implementation and the files needed to adapt it to an approved dataset.

## Repository contents

- `code/`: core modules, training scripts, inference scripts, GT-QDM, and G2D-UOT utilities.
- `configs/`: seed-specific templates with private paths replaced by placeholders.
- `data/anonymized_examples/`: a small collection of de-identified four-frame contact sheets.
- `results/metrics_summary.csv`: aggregate metric summary for the formal multi-seed run.
- `weights/README.md`: instructions for obtaining and placing compatible trainable weights.
- `run_formal_inference.sh`: an inference-only replay template.

No raw clinical dataset, patient identifiers, private server information, per-sample predictions, or credentials are included.

## Method overview

The teacher supplies task probabilities and short structured task semantics. Probabilities provide transport mass, semantic task relations define transport geometry, and clinical chain and knowledge-graph relations constrain transfers. The student combines global contact-sheet evidence with ordered sparse-frame probabilities and GT-QDM temporal refinement.

The training objective is:

```text
L_total = L_cls + lambda_prob * L_prob + lambda_G2D * L_G2D-UOT
```

The released templates use `lambda_prob=0.06` and `lambda_G2D=1.8`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Obtain the selected backbone and approved clinical data under their original licenses. Replace the path placeholders in `configs/` and set `THYQC_CODE_ROOT` to the local `code/` directory.

## Inference template

```bash
python code/train_g2d_uot_gt_qdm_seed42_20260919.py \
  --manifest <student_manifest.jsonl> \
  --prob-cache <student_probability_cache.jsonl> \
  --teacher-jsonl <teacher_labels.jsonl> \
  --semantic-cost-mode fixed_jaccard \
  --out-dir <output_dir> \
  --init-state <trainable_gt_qdm_state.pt> \
  --seed <seed> \
  --epochs 0 \
  --lambda-prob 0.06 \
  --lambda-g2d 1.8 \
  --uot-loss-mode transport
```

For training, use the same script with the approved manifest, teacher package, and a positive epoch count. The command-line templates are intentionally path-agnostic so that no private machine layout is exposed.

## Data and privacy

The included images are de-identified examples only. Any clinical release requires institutional approval, de-identification review, and redistribution permission. Do not commit raw images, original case IDs, timestamps, local filesystem paths, free-text clinical notes, or credentials.

## Weights

The large base backbone and trainable checkpoints are intentionally not redistributed. See `weights/README.md` for the expected layout and license checks.

## License

The repository code is released under the MIT License. Third-party models, datasets, and example images remain under their original licenses.
