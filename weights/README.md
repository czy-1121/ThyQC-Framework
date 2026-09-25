# Model weights

## Backbone

Download the InternVL3.5-2B-HF backbone from Hugging Face:

- [OpenGVLab/InternVL3_5-2B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF)

The Hugging Face repository ID can be passed directly to `--model-path`.

## ThyQC checkpoints

The repository stores the three released student runs with Git LFS:

```text
weights/
  thyqc/
    seed41_backbone_state.pt
    seed41_gt_qdm_state.pt
    seed42_backbone_state.pt
    seed42_gt_qdm_state.pt
    seed43_backbone_state.pt
    seed43_gt_qdm_state.pt
```

Run `git lfs pull` after cloning if the checkpoint files were not downloaded automatically.
