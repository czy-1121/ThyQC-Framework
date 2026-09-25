# ThyQC G2D-UOT + GT-QDM formal multi-seed reproduction

This package contains the formal ThyQC G2D-UOT + GT-QDM multi-seed reproduction used for the reported results.

This repository is the public code release for ThyQC. It includes the formal training/evaluation implementation, reproducibility metadata, locked student-head weights, metrics, predictions, and a small set of anonymized four-frame examples. Raw clinical data, private server paths, credentials, and the full internal test set are intentionally excluded.

## Formal protocol

- Seeds: 41, 42, 43
- Backbone: InternVL3.5-2B-HF with the existing ThyQC probability cache
- Student head: legacy GT-QDM `TemporalHead`
- Objective: `CE + 0.06 * L_prob + 1.8 * L_G2D-UOT`
- UOT mode: `transport`, epsilon `0.08`, rho `0.50`, 30 Sinkhorn iterations
- Semantic cost: task-state semantic relation matrix
- Clinical cost: chain/KG relations in `C_clin`
- Test protocol: select checkpoint on validation, then evaluate the locked checkpoint once on test

## Reproduction command

Run from a Python environment with the required dependencies after making the listed data paths available:

```bash
python train_g2d_uot_gt_qdm_seed42_20260919.py \
  --manifest <student_global_ordered_k4_manifest.jsonl> \
  --prob-cache <seed_specific_student_image_prob_cache.jsonl> \
  --teacher-jsonl <llava_med_teacher_labels.jsonl> \
  --semantic-cost-mode fixed_jaccard \
  --out-dir <output_dir> \
  --init-state <seed_specific_best_gt_qdm_state.pt> \
  --seed <41|42|43> --epochs 0 \
  --lambda-prob 0.06 --lambda-g2d 1.8 \
  --uot-loss-mode transport
```

`--epochs 0` performs inference-only evaluation of the supplied locked checkpoint.

## Formal test recheck

The independent inference recheck reproduced the stored formal metrics exactly at the JSON-file level (matching SHA-256):

| Seed | Mean Macro-F1 | Mean Macro-AUC |
|---:|---:|---:|
| 41 | 88.1822457 | 96.3080258 |
| 42 | 86.1058911 | 95.5597494 |
| 43 | 87.4724716 | 97.2624206 |

The three-seed summary is in `formal_mean_sd_summary.csv`.

## Code/model/data separation

The package stores the exact training/evaluation code, GT-QDM state files, configurations, predictions, and metrics. The large backbone and dataset files remain external and are listed in `formal_reproduction_manifest.json` with their original remote paths.

The implementation follows the paper's knowledge-transfer design: teacher task probabilities provide the transport mass, structured task semantics define the semantic geometry, clinical chain/KG relations define clinically compatible transport, and GT-QDM performs temporal refinement in the student model.

## Data and licensing

The included example contact sheets are for illustration and smoke testing only. Users must obtain the InternVL3.5-2B-HF backbone and any clinical dataset under their respective licenses. The internal test set should only be redistributed after institutional de-identification and data-sharing approval.
