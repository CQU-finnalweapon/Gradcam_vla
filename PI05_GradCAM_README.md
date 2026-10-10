# Three-camera PI0.5 Grad-CAM

Offline visualization against the original mozbrain code, with no robot control.
The source repository is owned by root and is not writable by this user; the
adapter is therefore kept here and imports that repository directly.

## Run

```bash
bash /mnt/vepfs01/output/klay.zhou/gradcam_vla/run.sh \
  --device cuda:3 --episode 0 --frame 150 \
  --action-step 0 --action-dim 13 --sign 1 --verify \
  --out /mnt/vepfs01/output/klay.zhou/gradcam_vla/results/new_run
```

The output directory must be empty. Use `--checkpoint`, `--dataset`, `--task`,
`--tokenizer`, `--seed`, and `--layer` to choose a different experiment. The
checkpoint must contain a complete `model.pt`, `model.safetensors`, or
`model.safetensors.index.json` with all indexed safetensors shards, its
configuration, normalization statistics, and a compatible tokenizer. Raw
distributed training checkpoints must first be exported with `merge_bcp.py`.
Only PI0.5 and exactly these three camera inputs are supported by this adapter.

Camera names are mapped as follows:

| Output Name | Accepted Policy/Dataset Names |
| --- | --- |
| `cam_high` | `cam_high` |
| `left_wrist_high` | `left_wrist_high`, `cam_left_wrist`, `cam_left_wrist_high` |
| `right_wrist_high` | `right_wrist_high`, `cam_right_wrist`, `cam_right_wrist_high` |

## What Is Explained

The target is `sign * normalized_actions[0, action_step, action_dim]`, after all
10 Euler denoising steps for the sample checkpoint. For that checkpoint dimension
13 is the right gripper command; 0-5 are left-arm Cartesian pose, 6 left gripper,
7-12 right-arm pose, and 14-19 torso pose. Inspect `target.label` in metadata for
other configurations. The target is a regression value, not a class probability,
object detection score, training loss, or overall attention score.

By default the hook captures `layer_norm1` in the last SigLIP encoder block,
separately for each camera. Gradients flow from the final action through every
denoising step and the differentiable prefix KV cache to these patch activations.
Channel weights are the mean target gradient over patch positions. Their weighted
activation sum is the signed map; ReLU gives standard positive Grad-CAM. The
256 patches form a 16 by 16 grid, upsampled to the input size. Preprocessing's
top/left padding or center crop is inverted before overlaying the original frame.

Weights are frozen temporarily to reduce memory usage; autograd starts at the
chosen feature layer. Hooks, training modes, and gradient flags are restored.
No training or weight updates occur. This run conditions on the dataset's task
and robot state, with fixed Gaussian noise. No previous action chunk is supplied,
so inference does not include RTC guidance from a previous chunk.

## Outputs

- `comparison.png`: three columns for the cameras; rows for originals, positive
  Grad-CAM, and negative response diagnostics.
- `*_gradcam.png`: standard positive Grad-CAM on each original camera frame.
- `*_negative.png`: negative signed response, provided separately as a diagnostic.
- `*_original.png` and `*_model_input.png`: original frames and actual model inputs.
- `gradcam_raw.npz`: signed/positive patch grids and unnormalized projected maps.
- `normalized_actions.npy`: complete padded normalized action chunk.
- `actions.npy`: unpadded action chunk after the checkpoint's output normalization
  is inverted. These are offline predictions and are not sent to hardware.
- `metadata.json`: weights, target, camera mapping, gradients, state, runtime,
  and optional verification measurements.

Maps are normalized independently for display. Colors cannot compare camera
importance or different runs. A blank positive map stays blank; negative values
are never silently converted to positive Grad-CAM. The result explains local
model sensitivity to the selected action target and does not guarantee object
localization or causal necessity.

`--verify` compares the differentiable sampler with the original sampler under
the same inputs/noise, then occludes 10% of high-CAM, low-CAM, and random patches
in each camera. It records the target's changes; it does not enforce a ranking.
Occlusion introduces out-of-distribution inputs and is supporting evidence only.

## Environment And Tests

`run.sh` uses `.venvs/mozbrain-gradcam` with system packages and sets
`PYTHONNOUSERSITE=1` to select the compatible system PyTorch/torchvision pair
(2.9.1+cu128 / 0.24.1), rather than the incompatible user-installed PyTorch.
`run.py` imports the repository's patched Transformers 4.53.2. Dependencies were
installed only in this virtual environment, not into the training environment.

```bash
PYTHONNOUSERSITE=1 /mnt/vepfs01/output/klay.zhou/.venvs/mozbrain-gradcam/bin/python \
  -m pytest /mnt/vepfs01/output/klay.zhou/gradcam_vla/test_gradcam.py -q
```

Tests cover the analytical CAM formula, independent gradients from a shared
camera encoder, cleanup on failures, padding/crop coordinates, and blank-map
rendering. Actual model verification is recorded in each run's metadata.
