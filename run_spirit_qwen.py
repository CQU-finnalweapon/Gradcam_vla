#!/usr/bin/env python3
"""Action-conditioned Grad-CAM using the Spirit checkpoint's own repository."""
import argparse
import faulthandler
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = Path('/mnt/vepfs02/output/anthony.cao/mozbrain_spirit_new_topic')
SOURCE = REPO / 'outputs/0928_ft_22303_pourcorn_uDAS_20h_new_topic_base_v1_6_0630stage2'
DATA = Path('/mnt/vepfs02/output/anthony.cao/datasets/22303_pourcorn_udas/dataset_0_57_ae6e0e92b5fa7267')


def sample_differentiable(model, state, embed, noise):
    """Same Euler solver, with out-of-place updates and padded-action masking."""
    embed = model.proj_vlm_output(embed)
    dt = torch.tensor(-1.0 / model.config.num_steps, device=state.device)
    time = torch.tensor(1.0, device=state.device)
    x = noise
    dim = model.config.action_feature.shape[0]
    while time >= -dt / 2:
        velocity = model.denoise_step(state.clone(), None, embed, x, time.expand(state.shape[0]))
        x = x + dt * velocity
        time = time + dt
        # Native x_t aliases noise and is updated in place. Its padding RHS
        # therefore uses the updated x_t, not the original initial noise.
        x = torch.cat((x[..., :dim], x[..., dim:] * time), dim=-1)
    return x


def unmerge(values, grid, merge):
    """Undo Qwen's (merged row, merged col, inner row, inner col) patch order."""
    t, h, w = map(int, grid)
    return values.reshape(t, h // merge, w // merge, merge, merge).permute(0, 1, 3, 2, 4).reshape(t, h, w).mean(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--frame', type=int, default=150)
    parser.add_argument('--action-step', type=int, default=10)
    parser.add_argument('--action-dim', type=int, default=9)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=HERE / 'results/pourcorn_step20000_ep0_frame150_right_z')
    args = parser.parse_args()
    faulthandler.enable()
    start = time.time()
    torch.set_num_threads(1)
    sys.path.insert(0, str(REPO))
    from mozinfer.spirit_qwen_dit_cotrain_policy import SpiritQwenDiTCotrainPolicy, pad_and_cat
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.common.robot_devices.transform_utils import ObservationStandardizer
    from transformers import AutoProcessor, AutoImageProcessor
    from transformers.modeling_utils import no_init_weights
    from run import load_sharded_safetensors, load_observation, camera_mapping, action_label, project_to_original, overlay

    checkpoint = HERE / 'checkpoints/pourcorn_step20000'
    assets = HERE / 'checkpoints/Qwen3-VL-4B-Instruct-assets'
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    assert config.type == 'spirit_qwen_dit_cotrain'
    original_backbone = config.backbone
    config.backbone = str(assets)
    config.tokenizer_path = str(checkpoint / 'tokenizer')
    config.pretrained_path = str(checkpoint)
    config.vlm_use_sac = False
    config.attention_implementation = 'eager'
    config.device = 'cuda'
    # Only the image processor is used by this repository's QwenWithDiT class.
    # Keep tokenizer loading explicitly tied to the checkpoint.
    from types import SimpleNamespace
    processor_loader = AutoProcessor.from_pretrained
    AutoProcessor.from_pretrained = lambda *a, **kw: SimpleNamespace(image_processor=AutoImageProcessor.from_pretrained(str(assets), local_files_only=True))
    print('Constructing checkpoint architecture from', REPO, flush=True)
    try:
        # Construct without random initialization; every persistent tensor
        # must pass the strict checkpoint key/shape check below.
        with torch.device('cpu'), no_init_weights():
            policy = SpiritQwenDiTCotrainPolicy(config)
    finally:
        AutoProcessor.from_pretrained = processor_loader
    state_dict = load_sharded_safetensors(checkpoint / 'model.safetensors.index.json')
    expected = policy.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    extra = sorted(set(state_dict) - set(expected))
    mismatch = [k for k in set(expected) & set(state_dict) if expected[k].shape != state_dict[k].shape]
    if missing or extra or mismatch:
        raise RuntimeError({'missing': missing, 'extra': extra, 'shape_mismatch': mismatch})
    print('Detaching safetensors mmap into local CPU RAM...', flush=True)
    state_dict = {k: v.clone() for k, v in state_dict.items()}
    print('Checkpoint keys/shapes verified. Copying weights...', flush=True)
    policy.load_state_dict(state_dict, strict=True, assign=True)
    print('Assigned checkpoint tensors; moving model to CUDA...', flush=True)
    policy = policy.to('cuda')
    del state_dict, expected
    policy.eval().requires_grad_(False)
    policy.model.paligemma_with_expert.qwen.gradient_checkpointing_disable()
    for module in (policy.normalize_inputs, policy.normalize_targets, policy.unnormalize_outputs):
        assert all(torch.isfinite(v).all() for v in module.state_dict().values())
    print('All checkpoint keys/shapes loaded strictly; normalization finite.', flush=True)

    mapping = camera_mapping(policy)
    raw, originals, sources = load_observation(DATA, args.episode, args.frame, mapping, None)
    standardizer = ObservationStandardizer(config.observation_config)
    obs = standardizer.standardize_camera_features(raw)
    obs['observation.state'] = standardizer.pack_state(raw, get_mask=False)
    batch = {k: v[None].to('cuda') for k, v in obs.items()}
    batch['task'] = [raw['task']]
    batch = policy.normalize_inputs(batch)
    state = policy.prepare_state(batch)
    pixels, grids, ids, positions, _ = policy.preprocess_rb_batch(batch)
    pixels = torch.cat(pixels).contiguous()
    grids = torch.cat(grids).contiguous()
    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=policy.language_tokenizer.pad_token_id).to('cuda')
    positions = pad_and_cat(positions).to('cuda')
    mask = ids.ne(policy.language_tokenizer.pad_token_id)
    model = policy.model
    visual = model.paligemma_with_expert.qwen.model.visual
    captured = {}
    def hook(module, inputs, output):
        output.requires_grad_(True)
        captured['activation'] = output
        return output
    handle = visual.blocks[-1].register_forward_hook(hook)
    torch.manual_seed(args.seed)
    noise = model.sample_noise((1, config.n_action_steps, config.max_action_dim), 'cuda')
    print('Forward: image grids', grids.tolist(), 'target', action_label(config, args.action_dim), flush=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        embed = model.paligemma_with_expert.qwen.model(pixel_values=pixels, image_grid_thw=grids, input_ids=ids,
            position_ids=positions, attention_mask=mask, use_cache=False).last_hidden_state
        actions = sample_differentiable(model, state.clone(), embed, noise.clone())
        # Repository Unnormalize.forward is decorated with no_grad. Invoke
        # its unchanged underlying function to retain the action's graph.
        real_actions = type(policy.unnormalize_outputs).forward.__wrapped__(
            policy.unnormalize_outputs, {'action': actions[..., :config.action_feature.shape[0]]})['action']
        target = real_actions[0, args.action_step, args.action_dim]
    assert torch.isfinite(real_actions).all()
    print('Backward from action scalar', target.item(), flush=True)
    activation = captured['activation']
    grad, = torch.autograd.grad(target, activation)
    assert torch.isfinite(grad).all() and grad.abs().max() > 0
    handle.remove()
    # Essential check: the differentiable solver must reproduce native inference.
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        reference = model.sample_actions_unified(state.clone(), embed.detach(), noise=noise.clone())
    error = float((reference - actions.detach()).abs().max())
    if not torch.allclose(reference, actions.detach(), atol=1e-5, rtol=1e-5):
        raise RuntimeError(f'Sampler mismatch: {error}')
    print('Native/differentiable sampler max error:', error, flush=True)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    merge = visual.spatial_merge_size
    maps = {}
    rows = [[], [], []]
    offset = 0
    stats = {}
    for key, grid in zip(mapping, grids.tolist(), strict=True):
        label = mapping[key]
        count = int(np.prod(grid))
        a = activation[offset:offset+count].detach().float()
        g = grad[offset:offset+count].float()
        signed = unmerge((a * g.mean(0)).sum(-1), grid, merge).cpu().numpy()
        offset += count
        maps[label + '_signed_grid'] = signed
        stats[label] = {'grid_thw': grid, 'gradient_abs_max': g.abs().max().item(), 'signed_min': float(signed.min()), 'signed_max': float(signed.max())}
        rgb = originals[key]
        images = [rgb]
        Image.fromarray(rgb).save(out / f'{label}_original.png')
        Image.fromarray((obs[key].permute(1, 2, 0).numpy() * 255).clip(0,255).astype('uint8')).save(out / f'{label}_model_input.png')
        size = config.observation_config.wrist_image_size if 'wrist' in key else config.observation_config.image_size
        for kind, heat in [('gradcam', np.maximum(signed, 0)), ('negative', np.maximum(-signed, 0))]:
            projected = project_to_original(heat, rgb.shape, size, config.observation_config.image_resize_mode)
            maps[f'{label}_{kind}_original'] = projected
            rendered = overlay(rgb, projected)
            Image.fromarray(rendered).save(out / f'{label}_{kind}.png')
            images.append(rendered)
        for row, img in zip(rows, images):
            panel = Image.fromarray(img)
            panel.thumbnail((480, 320))
            tile = Image.new('RGB', (480, 350), 'white')
            tile.paste(panel, ((480-panel.width)//2, 27))
            ImageDraw.Draw(tile).text((12, 8), label, fill='black')
            row.append(tile)
    assert offset == activation.shape[0]
    canvas = Image.new('RGB', (1440, 1230), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), f'Spirit Qwen3-VL | checkpoint 20000 | episode {args.episode} frame {args.frame} | action[{args.action_step},{args.action_dim}]', fill='black')
    draw.text((15, 32), raw['task'], fill='black')
    draw.text((15, 52), f'visual.blocks.23 output | full 10-step sampling | seed {args.seed} | each heatmap scaled independently', fill='black')
    for r, title in enumerate(['Original camera frames', 'Positive Grad-CAM for the selected action scalar', 'Negative signed CAM (diagnostic)']):
        y = 80 + r * 380
        draw.text((15, y), title, fill='black')
        for c, tile in enumerate(rows[r]):
            canvas.paste(tile, (480*c, y+25))
    canvas.save(out / 'comparison.png')
    np.savez_compressed(out / 'gradcam_raw.npz', **maps)
    np.save(out / 'predicted_actions.npy', real_actions.detach().float().cpu().numpy())
    metadata = {'checkpoint_step': 20000, 'source_run': str(SOURCE), 'merged_checkpoint': str(checkpoint), 'architecture_repo': str(REPO),
        'dataset': str(DATA), 'episode': args.episode, 'frame': args.frame, 'task': raw['task'], 'cameras': sources,
        'layer': 'model.paligemma_with_expert.qwen.model.visual.blocks.23', 'activation_shape': list(activation.shape),
        'target': {'action_step_zero_based': args.action_step, 'action_dim_zero_based': args.action_dim, 'label': action_label(config, args.action_dim), 'unnormalized_value': target.item()},
        'seed': args.seed, 'sampling_steps': config.num_steps, 'sampler_max_abs_error': error, 'maps': stats,
        'original_backbone': original_backbone, 'local_backbone_assets': str(assets), 'attention_implementation': 'eager', 'autocast': 'bfloat16',
        'visual_token_order': 'unmerged from (t,h//2,w//2,2,2) into (t,h,w)', 'normalization': 'each positive/negative camera map independently scaled for display',
        'seconds': time.time()-start}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    files = ['mozinfer/spirit_qwen_dit_cotrain_policy.py', 'lerobot/common/policies/qwen_dit/qwen_with_dit.py', 'lerobot/common/policies/qwen_dit/qwen3_action_expert.py', 'lerobot/common/robot_devices/transform_utils.py']
    provenance = {'source_run': str(SOURCE), 'architecture_repo': str(REPO), 'strict_state_dict_load': True,
        'source_sha256': {f: hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in files},
        'weights_index': json.loads((checkpoint/'model.safetensors.index.json').read_text()),
        'processor_note': 'Qwen3-VL processor configuration reused from local Qwen3-VL-8B assets (same image processing family); 4B model config extracted from this run training log.'}
    (out/'checkpoint_provenance.json').write_text(json.dumps(provenance, indent=2))
    print('Saved', out/'comparison.png', flush=True)


if __name__ == '__main__':
    main()
