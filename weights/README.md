# Model weights

## Backbone

Download the InternVL3.5-2B-HF backbone from Hugging Face:

- [OpenGVLab/InternVL3_5-2B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF)

The Hugging Face repository ID can be passed directly to `--model-path`.

## ThyQC checkpoints

Evaluation-only checkpoints use the following layout:

```text
weights/
  seed41_best_gt_qdm_state.pt
  seed42_best_gt_qdm_state.pt
  seed43_best_gt_qdm_state.pt
```

The checkpoints can also be regenerated with the training commands in the main README. When released separately, place the downloaded files in this directory without renaming them.
