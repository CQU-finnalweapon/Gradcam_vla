#!/usr/bin/env python3
"""Offline three-camera Grad-CAM for mozbrain PI0.5 checkpoints."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from gradcam import explain

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_REPO = WORKSPACE / 'trainning/mozbrain-main/mozbrain'
CAMERA_ALIASES = {
    'cam_high': ('cam_high',),
    'left_wrist_high': ('left_wrist_high', 'cam_left_wrist', 'cam_left_wrist_high'),
    'right_wrist_high': ('right_wrist_high', 'cam_right_wrist', 'cam_right_wrist_high'),
}


def setup_repo(repo):
    sys.path[:0] = [str(repo), str(repo / 'thirdparty/transformers/src')]
    from transformers.models.gemma import modeling_gemma
    if not hasattr(modeling_gemma, '_gated_residual'):
        raise RuntimeError('PI0.5 needs this repository\'s patched Transformers')


def load_sharded_safetensors(index_path):
    from safetensors.torch import load_file

    weight_map = json.loads(index_path.read_text())['weight_map']
    if not weight_map:
        raise ValueError('Empty safetensors weight index')
    state = {}
    for filename in sorted(set(weight_map.values())):
        print(f'Reading checkpoint shard: {filename}', flush=True)
        shard = load_file(str(index_path.parent / filename), device='cpu')
        expected = {key for key, value in weight_map.items() if value == filename}
        if set(shard) != expected:
            raise RuntimeError(f'Shard/index key mismatch: {filename}')
        state.update(shard)
    return state


def load_policy(checkpoint, device, tokenizer=None):
    from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.configs.policies import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    if config.type != 'pi05':
        raise ValueError('This adapter supports pi05 checkpoints only')
    config.device = str(device)
    candidates = [Path(tokenizer)] if tokenizer else [
        checkpoint / 'tokenizer/sentencepiece_tokenizer.model',
        Path(config.tokenizer_path or '/nonexistent'),
    ]
    config.tokenizer_path = str(next((p for p in candidates if p.is_file()), ''))
    if not config.tokenizer_path:
        raise FileNotFoundError('Supply --tokenizer with the checkpoint\'s SentencePiece model')
    with torch.device(device):
        print(f'Constructing PI0.5 on {device}...', flush=True)
        policy = PI05Policy(config)
    weight_path = checkpoint / 'model.safetensors'
    if weight_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(weight_path), device='cpu')
    elif (checkpoint / 'model.safetensors.index.json').exists():
        weight_path = checkpoint / 'model.safetensors.index.json'
        state = load_sharded_safetensors(weight_path)
    else:
        weight_path = checkpoint / 'model.pt'
        print(f'Reading checkpoint: {weight_path}', flush=True)
        state = torch.load(weight_path, map_location='cpu', weights_only=True, mmap=True)
    for wrapper in ('state_dict', 'policy', 'model'):
        if wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]
            break
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    expected = policy.state_dict()
    # Neither text vocabulary head is used by PI0.5 action inference. Some
    # exported FSDP checkpoints contain a flattened shard for these heads.
    unused = {k for k in expected if '.lm_head.' in k}
    missing = set(expected) - set(state) - unused
    extra = set(state) - set(expected)
    mismatches = [k for k in expected.keys() & state.keys() - unused
                  if expected[k].shape != state[k].shape]
    if missing or extra or mismatches:
        raise RuntimeError(f'Incomplete checkpoint: missing={sorted(missing)}, '
                           f'extra={sorted(extra)}, shape_mismatch={mismatches}')
    filtered = {k: v for k, v in state.items() if k not in unused}
    print('Copying verified checkpoint tensors to the model...', flush=True)
    policy.load_state_dict(filtered, strict=False)
    for module in (policy.normalize_inputs, policy.normalize_targets, policy.unnormalize_outputs):
        for key, tensor in module.state_dict().items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f'Invalid normalization statistics: {key}')
    policy.eval()
    policy.model.gradient_checkpointing_disable()
    report = {'weights': str(weight_path.resolve()), 'loaded_tensors': len(filtered),
              'unused_language_heads': sorted(unused), 'missing_used_tensors': []}
    print(f'Loaded {len(filtered)} tensors; all action/vision weights and normalization statistics verified.', flush=True)
    return policy, report


def camera_mapping(policy):
    keys = list(policy.config.image_features)
    mapping = {}
    for label, aliases in CAMERA_ALIASES.items():
        matches = [key for key in keys if key.removeprefix('observation.images.') in aliases]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one configured camera for {label}: {keys}')
        mapping[matches[0]] = label
    if set(mapping) != set(keys):
        raise ValueError('This run expects exactly the three requested cameras; do not silently drop other views')
    return mapping


def load_observation(root, episode, frame, mapping, task_override):
    import pyarrow.parquet as pq

    info = json.loads((root / 'meta/info.json').read_text())
    context = {'episode_chunk': episode // info.get('chunks_size', 1000), 'episode_index': episode}
    data_path = root / info['data_path'].format(**context)
    table = pq.read_table(data_path)
    if not 0 <= frame < len(table):
        raise ValueError(f'Frame {frame} is outside episode {episode} ({len(table)} frames)')
    row = table.slice(frame, 1).to_pylist()[0]
    if int(row['frame_index']) != frame or int(row['episode_index']) != episode:
        raise ValueError('Parquet row index does not match the requested frame/episode')
    tasks = [json.loads(line) for line in (root / 'meta/tasks.jsonl').read_text().splitlines() if line.strip()]
    task = task_override or next(x['task'] for x in tasks if x['task_index'] == int(row['task_index']))
    raw = {k: np.asarray(v, dtype=np.float32) if isinstance(v, list) else v for k, v in row.items()}
    originals, sources = {}, {}
    for key, label in mapping.items():
        short = key.removeprefix('observation.images.')
        candidates = (short,) + CAMERA_ALIASES[label]
        video_key = next((k for name in candidates for k in (name, 'observation.images.' + name)
                          if k in info['features'] and info['features'][k]['dtype'] == 'video'), None)
        if video_key is None:
            raise KeyError(f'No dataset video for {label}')
        path = root / info['video_path'].format(**context, video_key=video_key)
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open {path}')
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot read frame {frame} from {path}')
            actual_frame = round(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if actual_frame != frame:
                raise RuntimeError(f'Video seek mismatch: requested {frame}, got {actual_frame}')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
        raw[key] = rgb
        # Older ObservationStandardizer versions read unprefixed camera names.
        raw[short] = rgb
        originals[key] = rgb
        sources[label] = {'video': str(path.resolve()), 'dataset_camera': video_key,
                          'policy_camera': key, 'original_shape': list(rgb.shape)}
    raw['task'] = task
    return raw, originals, sources


def prepare(policy, raw, device):
    from lerobot.common.robot_devices.transform_utils import ObservationStandardizer

    standardizer = ObservationStandardizer(policy.config.observation_config)
    # Only the observation is needed: avoid computing demonstration actions.
    obs = standardizer.standardize_camera_features(raw)
    obs['observation.state'] = standardizer.pack_state(raw, get_mask=False)
    if tuple(obs['observation.state'].shape) != tuple(policy.config.input_features['observation.state'].shape):
        raise ValueError('Packed robot state does not match checkpoint')
    batch = {k: v.unsqueeze(0).to(device) for k, v in obs.items()}
    batch['task'] = [raw['task']]
    batch = policy.normalize_inputs(batch)
    if not torch.isfinite(batch['observation.state']).all():
        raise ValueError('Normalized robot state contains non-finite values')
    images, masks = policy.prepare_images(batch)
    tokens, token_masks = policy.prepare_language(batch)
    state = policy.prepare_state(batch)
    return (images, masks, tokens, token_masks, state), batch['observation.state'].cpu().tolist()


def action_label(config, dim):
    from lerobot.common.robot_devices.transform_utils import IndexConfig
    obs = config.observation_config
    extra = {'use_stop_signal': True} if getattr(obs, 'use_stop_signal', False) else {}
    components = IndexConfig.get_component_info(
        obs.robot_type, obs.moz1_structure, obs.action_space, obs.gripper_first, **extra)
    for component, details in components.items():
        start, end = details['range']
        if start <= dim < end:
            return f'{component}[{dim - start}]'
    return f'action[{dim}]'


def upsample(grid, size):
    return F.interpolate(torch.as_tensor(grid)[None, None].float(), size=size,
                         mode='bilinear', align_corners=False)[0, 0].numpy()


def project_to_original(grid, original_shape, image_size, mode):
    """Invert ObservationStandardizer's top/left padding or center crop."""
    h, w = original_shape[:2]
    th, tw = image_size
    model_map = upsample(grid, (th, tw))
    if mode == 'resize_with_pad':
        if th / tw > h / w:
            nw, nh = tw, int(h * tw / w)
        else:
            nh, nw = th, int(w * th / h)
        return upsample(model_map[th - nh:, tw - nw:], (h, w))
    if mode == 'resize':
        return upsample(model_map, (h, w))
    if mode == 'resize_with_crop':
        result = np.zeros((h, w), dtype=np.float32)
        if th / tw > h / w:
            nw = int(h * tw / th)
            start = (w - nw) // 2
            result[:, start:start + nw] = upsample(model_map, (h, nw))
        else:
            nh = int(w * th / tw)
            start = (h - nh) // 2
            result[start:start + nh] = upsample(model_map, (nh, w))
        return result
    raise ValueError(f'Unsupported resize mode {mode}')


def overlay(rgb, heat):
    maximum = float(heat.max())
    normalized = heat / maximum if maximum > 0 else np.zeros_like(heat)
    colored = cv2.applyColorMap(np.uint8(normalized.clip(0, 1) * 255), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    # Zero heat leaves the input unchanged, including completely blank CAMs.
    alpha = (0.55 * normalized)[..., None]
    return np.uint8(np.clip(rgb * (1 - alpha) + colored * alpha, 0, 255))


def render(out, mapping, originals, prepared, maps, metadata, config):
    rows = [[], [], []]
    arrays = {}
    for key, (signed, positive) in zip(mapping, maps, strict=True):
        label = mapping[key]
        rgb = originals[key]
        negative = np.maximum(-signed, 0)
        images = [rgb]
        Image.fromarray(rgb).save(out / f'{label}_original.png')
        for kind, grid in [('gradcam', positive), ('negative', negative)]:
            projected = project_to_original(grid, rgb.shape, config.observation_config.image_size,
                                            config.observation_config.image_resize_mode)
            combined = overlay(rgb, projected)
            Image.fromarray(combined).save(out / f'{label}_{kind}.png')
            images.append(combined)
            arrays[f'{label}_{kind}_original'] = projected
        for row, image in zip(rows, images, strict=True):
            panel = Image.fromarray(image)
            panel.thumbnail((480, 320))
            tile = Image.new('RGB', (480, 355), 'white')
            tile.paste(panel, ((480-panel.width)//2, 30))
            ImageDraw.Draw(tile).text((12, 9), label, fill='black')
            row.append(tile)
        arrays[f'{label}_signed_grid'] = signed
        arrays[f'{label}_positive_grid'] = positive
    for key, image in zip(mapping, prepared[0], strict=True):
        rgb = ((image[0].detach().cpu().float() + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        Image.fromarray(rgb).save(out / f'{mapping[key]}_model_input.png')
    canvas = Image.new('RGB', (1440, 3 * 385 + 85), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), f"PI0.5 Grad-CAM | episode {metadata['episode']} frame {metadata['frame']} | "
              f"{metadata['target']['sign']:+d} * action[{metadata['target']['action_step']},"
              f"{metadata['target']['action_dim']}] ({metadata['target']['label']})", fill='black')
    # PIL's built-in font may not support a non-English task; it is preserved in metadata.
    draw.text((15, 33), metadata['task'].encode('ascii', errors='replace').decode(), fill='black')
    draw.text((15, 54), 'Full denoising trajectory, fixed seed. Each heatmap scaled independently.', fill='black')
    titles = ['Original camera frames', 'Grad-CAM: positive evidence for the signed target',
              'Negative evidence (diagnostic; not standard positive Grad-CAM)']
    for r, row in enumerate(rows):
        y = 85 + r * 385
        draw.text((15, y), titles[r], fill='black')
        for col, tile in enumerate(row):
            canvas.paste(tile, (480 * col, y + 25))
    canvas.save(out / 'comparison.png')
    np.savez_compressed(out / 'gradcam_raw.npz', **arrays)


def verify(policy, prepared, noise, actions, maps, target):
    with torch.no_grad():
        reference = policy.model.sample_actions(*prepared, noise=noise)
        delta = (actions - reference).abs().max().item()
        if not torch.allclose(actions, reference, atol=1e-5, rtol=1e-5):
            raise RuntimeError(f'Differentiable sampler differs from original by {delta}')
        result = {'sampler_max_abs_error': delta, 'occlusion': []}
        step, dim = target['action_step'], target['action_dim']
        baseline = reference[0, step, dim].item()
        for camera_idx, (_, positive) in enumerate(maps):
            count = max(1, int(positive.size * 0.1))
            order = np.argsort(positive.reshape(-1), kind='stable')
            rng = np.random.default_rng(42)
            indices = {'high_cam': order[-count:], 'low_cam': order[:count],
                       'random': rng.choice(positive.size, count, replace=False)}
            item = {'camera_index': camera_idx, 'occluded_patch_count': count,
                    'blank_positive_cam': bool(positive.max() == 0)}
            for kind, selected in indices.items():
                mask = np.zeros(positive.size, dtype=np.float32)
                mask[selected] = 1
                mask = torch.from_numpy(mask.reshape(positive.shape))[None, None].to(noise.device)
                mask = F.interpolate(mask, size=prepared[0][camera_idx].shape[-2:], mode='nearest').bool()
                images = list(prepared[0])
                images[camera_idx] = images[camera_idx].masked_fill(mask, 0)  # gray after [-1, 1] scaling
                perturbed = policy.model.sample_actions(images, *prepared[1:], noise=noise)
                value = perturbed[0, step, dim].item()
                item[kind] = {'value': value, 'absolute_change': abs(value-baseline),
                              'signed_target_change': target['sign'] * (value-baseline)}
            result['occlusion'].append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_REPO / 'outputs/Basketball_trtc_test/checkpoints/005000/pretrained_model')
    parser.add_argument('--dataset', type=Path, default=WORKSPACE / 'datasets/20260330_basketball_NewGripper_Moz1WB')
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--frame', type=int, default=150)
    parser.add_argument('--task', help='Override the task from dataset metadata')
    parser.add_argument('--action-step', type=int, default=0)
    parser.add_argument('--action-dim', type=int, default=13, help='Real normalized action dimension (default right gripper for sample checkpoint)')
    parser.add_argument('--sign', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--layer', type=int, default=-1, help='SigLIP encoder block index, hook at layer_norm1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--verify', action='store_true', help='Compare original sampler and run high/low/random occlusion')
    parser.add_argument('--out', type=Path, default=WORKSPACE / 'gradcam_vla/results/basketball_ep0_frame150')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {args.out}')
    setup_repo(args.repo.resolve())
    torch.set_num_threads(8)
    device = torch.device(args.device)
    start = time.perf_counter()
    policy, weight_report = load_policy(args.checkpoint, device, args.tokenizer)
    mapping = camera_mapping(policy)
    # Retain the model's actual camera order for both hooks and rendering.
    mapping = {k: mapping[k] for k in policy.config.image_features}
    raw, originals, sources = load_observation(args.dataset, args.episode, args.frame, mapping, args.task)
    prepared, normalized_state = prepare(policy, raw, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn((1, policy.config.n_action_steps, policy.config.max_action_dim),
                        generator=generator, device=device)
    target = {'action_step': args.action_step, 'action_dim': args.action_dim,
              'sign': args.sign, 'label': action_label(policy.config, args.action_dim),
              'units': 'normalized action output; gradient through all Euler steps'}
    print(f"Task: {raw['task']}; target: {target}; cameras: {mapping}", flush=True)
    autocast = (torch.autocast(device.type, dtype=torch.bfloat16)
                if policy.config.dtype == 'bfloat16' and device.type in ('cuda', 'cpu')
                else nullcontext())
    with autocast:
        actions, maps, diagnostics = explain(policy, prepared, noise, args.action_step,
                                              args.action_dim, args.sign, args.layer)
        print(f'Grad-CAM completed. Camera gradients: {diagnostics}', flush=True)
        verification = verify(policy, prepared, noise, actions, maps, target) if args.verify else None
    metadata = {
        'repo': str(args.repo.resolve()), 'checkpoint': str(args.checkpoint.resolve()),
        'dataset': str(args.dataset.resolve()), 'episode': args.episode, 'frame': args.frame,
        'task': raw['task'], 'target': target, 'seed': args.seed,
        'target_value': args.sign * actions[0, args.action_step, args.action_dim].item(),
        'num_denoising_steps': policy.config.num_steps, 'rtc_previous_actions': None,
        'layer': f'vision_tower.vision_model.encoder.layers[{args.layer}].layer_norm1',
        'camera_sources': sources, 'camera_order': list(mapping.values()),
        'diagnostics': dict(zip(mapping.values(), diagnostics, strict=True)),
        'weight_loading': weight_report, 'verification': verification,
        'normalized_state': normalized_state, 'seconds': time.perf_counter()-start,
        'torch_version': torch.__version__, 'command': sys.argv,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    render(args.out, mapping, originals, prepared, maps, metadata, policy.config)
    np.save(args.out / 'normalized_actions.npy', actions[0].float().cpu().numpy())
    with torch.no_grad():
        physical = policy.unnormalize_outputs({'action': actions[:, :, :policy.config.action_feature.shape[0]]})['action']
    np.save(args.out / 'actions.npy', physical[0].float().cpu().numpy())
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    print(f'Saved three-camera comparison: {args.out / "comparison.png"}', flush=True)


if __name__ == '__main__':
    main()
