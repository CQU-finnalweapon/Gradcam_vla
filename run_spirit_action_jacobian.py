#!/usr/bin/env python3
"""Exact action-Jacobian norm maps for the Spirit Qwen3-VL policy.

For every model-input pixel p this computes
  sqrt(sum_{action_step, action_dim, rgb_channel} (d action / d pixel)^2).
The VJPs are evaluated in chunks, so the full 1200-output Jacobian is never
materialized at once.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = Path('/mnt/vepfs02/output/anthony.cao/mozbrain_spirit_new_topic')
DATA = Path('/mnt/vepfs02/output/anthony.cao/datasets/22303_pourcorn_udas/dataset_0_57_ae6e0e92b5fa7267')
CHECKPOINT = HERE / 'checkpoints/pourcorn_step20000'
ASSETS = HERE / 'checkpoints/Qwen3-VL-4B-Instruct-assets'


def sample_differentiable(model, state, embed, noise):
    embed = model.proj_vlm_output(embed)
    dt = torch.tensor(-1.0 / model.config.num_steps, device=state.device)
    tm = torch.tensor(1.0, device=state.device)
    x = noise
    dim = model.config.action_feature.shape[0]
    while tm >= -dt / 2:
        velocity = model.denoise_step(state.clone(), None, embed, x, tm.expand(state.shape[0]))
        x = x + dt * velocity
        tm = tm + dt
        # Reproduce the repository's native in-place padding behavior.
        x = torch.cat((x[..., :dim], x[..., dim:] * tm), dim=-1)
    return x


def unpack_rgb(pixels, grid, processor):
    """Invert the processor's patch permutation into unique [0,1] RGB pixels."""
    t, h, w = map(int, grid)
    if t != 1:
        raise ValueError('This adapter expects static camera images')
    p, m, temporal = processor.patch_size, processor.merge_size, processor.temporal_patch_size
    x = pixels.reshape(h // m, w // m, m, m, 3, temporal, p, p)
    x = x.permute(4, 5, 0, 2, 6, 1, 3, 7).reshape(3, temporal, h*p, w*p)
    torch.testing.assert_close(x[:, 0], x[:, 1], atol=0, rtol=0)
    mean = torch.tensor(processor.image_mean, device=x.device)[:, None, None]
    std = torch.tensor(processor.image_std, device=x.device)[:, None, None]
    return (x[:, 0] * std + mean).detach().requires_grad_(True)


def pack_rgb(image, processor):
    """Differentiable processor packing; duplicate temporal pixels stay tied."""
    p, m, temporal = processor.patch_size, processor.merge_size, processor.temporal_patch_size
    h, w = image.shape[-2] // p, image.shape[-1] // p
    mean = torch.tensor(processor.image_mean, device=image.device)[:, None, None]
    std = torch.tensor(processor.image_std, device=image.device)[:, None, None]
    normalized = (image - mean) / std
    x = normalized[None].expand(temporal, -1, -1, -1)
    x = x.reshape(temporal, 3, h//m, m, p, w//m, m, p)
    return x.permute(2, 5, 3, 6, 1, 0, 4, 7).reshape(h*w, 3*temporal*p*p)


def parse_tile_size(value):
    if value == 'full':
        return None
    try:
        h, w = map(int, value.lower().split('x'))
        if h > 0 and w > 0:
            return h, w
    except ValueError:
        pass
    raise argparse.ArgumentTypeError('image-tile-size must be full or positive HxW, e.g. 128x160')


def tile_images(images, tile_size=None):
    """Independent input leaves, stitched back into unchanged full camera images."""
    tiles, locations, stitched = [], [], []
    for camera, image in enumerate(images):
        h, w = image.shape[-2:]
        th, tw = tile_size or (h, w)
        rows = []
        for y in range(0, h, th):
            row = []
            for x in range(0, w, tw):
                tile = image[:, y:y+th, x:x+tw].detach().clone().requires_grad_(True)
                tiles.append(tile)
                locations.append((camera, y, x, tile.shape[-2], tile.shape[-1]))
                row.append(tile)
            rows.append(torch.cat(row, dim=-1))
        stitched.append(torch.cat(rows, dim=-2))
    return tiles, locations, stitched


def jacobian_norm(outputs, inputs, chunk=1, progress=None, output_start=0,
                  output_end=None, input_chunk=0, squared=False):
    """Stream exact squared Jacobian entries over disjoint output/input groups.

    Input groups share one full-context forward graph. Small input groups bound
    returned gradient buffers but repeat upstream backward work.
    """
    all_flat = outputs.reshape(-1)
    output_end = all_flat.numel() if output_end is None else output_end
    if chunk < 1 or input_chunk < 0 or not inputs:
        raise ValueError('invalid Jacobian chunk sizes or empty inputs')
    if not 0 <= output_start < output_end <= all_flat.numel():
        raise ValueError('invalid action output range')
    flat = all_flat[output_start:output_end]
    input_chunk = input_chunk or len(inputs)
    sums = [torch.zeros(x.shape[-2:], device=x.device, dtype=torch.float32) for x in inputs]
    valid = torch.ones((), device=flat.device, dtype=torch.bool)
    for start in range(0, flat.numel(), chunk):
        end = min(start + chunk, flat.numel())
        basis = torch.zeros((end-start, flat.numel()), device=flat.device, dtype=flat.dtype)
        basis[torch.arange(end-start, device=flat.device), torch.arange(start, end, device=flat.device)] = 1
        for first in range(0, len(inputs), input_chunk):
            last = min(first+input_chunk, len(inputs))
            retain = end < flat.numel() or last < len(inputs)
            if chunk == 1:
                grads = torch.autograd.grad(flat, inputs[first:last], grad_outputs=basis[0], retain_graph=retain)
                grads = tuple(g[None] for g in grads)
            else:
                grads = torch.autograd.grad(flat, inputs[first:last], grad_outputs=basis,
                                            is_grads_batched=True, retain_graph=retain)
            for acc, g in zip(sums[first:last], grads, strict=True):
                valid.logical_and_(torch.isfinite(g).all())
                acc.add_(g.float().square().sum(dim=(0, 1)))
            del grads
        if progress:
            progress(end, flat.numel())
        del basis
    if not valid.item() or not all(torch.isfinite(x).all().item() for x in sums):
        raise RuntimeError('Non-finite Jacobian or accumulated square sum')
    return [x.detach() if squared else x.sqrt().detach() for x in sums]


def stitch_tile_sums(tile_sums, locations, images):
    sums = [torch.zeros(x.shape[-2:], device=x.device, dtype=torch.float32) for x in images]
    for value, (camera, y, x, h, w) in zip(tile_sums, locations, strict=True):
        sums[camera][y:y+h, x:x+w] = value
    return sums


def overlay(rgb, heat, lo, hi):
    z = ((heat - lo) / max(hi - lo, 1e-12)).clip(0, 1)
    color = cv2.applyColorMap(np.uint8(z * 255), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    alpha = (0.58 * z)[..., None]
    return np.uint8(np.clip(rgb * (1 - alpha) + color * alpha, 0, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--frame', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--vjp-chunk', type=int, default=1)
    ap.add_argument('--image-tile-size', type=parse_tile_size, default=None, help='RGB input tiles: full or HxW; preserves full-image context')
    ap.add_argument('--tiles-per-backward', type=int, default=0, help='Number of input tiles per backward (0=all); smaller repeats backward and lowers returned-gradient memory')
    ap.add_argument('--activation-offload', action='store_true', help='Store autograd saved tensors on CPU; slower, reduces GPU activation memory')
    ap.add_argument('--verify-tiling', action='store_true', help='Compare full-image vs tiled gradients on three action outputs within the same graph')
    ap.add_argument('--action-start', type=int, default=0, help='First flattened action output (inclusive) for this worker')
    ap.add_argument('--action-end', type=int, default=None, help='Last flattened action output (exclusive) for this worker')
    ap.add_argument('--partial-output', type=Path, default=None, help='Save only this worker partial NPZ and exit')
    ap.add_argument('--output', type=Path, default=HERE / 'results/pourcorn_step20000_ep0_frame150_action_jacobian')
    args = ap.parse_args()
    if args.partial_output is None and (args.action_start != 0 or args.action_end is not None):
        ap.error('action ranges require --partial-output to avoid labeling partial maps as full')
    if args.vjp_chunk < 1 or args.tiles_per_backward < 0:
        ap.error('--vjp-chunk must be positive')
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(1)
    sys.path.insert(0, str(REPO))

    from mozinfer.spirit_qwen_dit_cotrain_policy import SpiritQwenDiTCotrainPolicy, pad_and_cat
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.common.robot_devices.transform_utils import ObservationStandardizer
    from transformers import AutoProcessor, AutoImageProcessor
    from transformers.modeling_utils import no_init_weights
    from run import load_sharded_safetensors, load_observation, camera_mapping

    config = PreTrainedConfig.from_pretrained(str(CHECKPOINT))
    config.backbone = str(ASSETS)
    config.tokenizer_path = str(CHECKPOINT / 'tokenizer')
    config.pretrained_path = str(CHECKPOINT)
    config.vlm_use_sac = False
    config.attention_implementation = 'eager'
    config.device = 'cuda'

    old_loader = AutoProcessor.from_pretrained
    AutoProcessor.from_pretrained = lambda *a, **kw: type('P', (), {
        'image_processor': AutoImageProcessor.from_pretrained(str(ASSETS), local_files_only=True)
    })()
    try:
        with torch.device('cpu'), no_init_weights():
            policy = SpiritQwenDiTCotrainPolicy(config)
    finally:
        AutoProcessor.from_pretrained = old_loader

    state_dict = load_sharded_safetensors(CHECKPOINT / 'model.safetensors.index.json')
    expected = policy.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    extra = sorted(set(state_dict) - set(expected))
    mismatch = [k for k in set(expected) & set(state_dict) if expected[k].shape != state_dict[k].shape]
    if missing or extra or mismatch:
        raise RuntimeError({'missing': missing, 'extra': extra, 'shape_mismatch': mismatch})
    state_dict = {k: v.clone() for k, v in state_dict.items()}
    policy.load_state_dict(state_dict, strict=True, assign=True)
    policy = policy.to('cuda').eval().requires_grad_(False)
    del state_dict, expected
    policy.model.paligemma_with_expert.qwen.gradient_checkpointing_disable()

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
    baseline_pixels = torch.cat(pixels).contiguous()
    grids = torch.cat(grids).contiguous()
    processor = policy.model.paligemma_with_expert.image_processor
    image_inputs = []
    offset = 0
    for grid in grids.tolist():
        count = int(np.prod(grid))
        image_inputs.append(unpack_rgb(baseline_pixels[offset:offset+count], grid, processor))
        offset += count
    tile_inputs, tile_locations, stitched = tile_images(image_inputs, args.image_tile_size)
    print(f'Image tiles: {len(tile_inputs)}, tiles per backward: {args.tiles_per_backward or len(tile_inputs)}', flush=True)
    pixels = torch.cat([pack_rgb(x, processor) for x in stitched])
    torch.testing.assert_close(pixels.detach(), baseline_pixels, atol=1e-6, rtol=1e-6)
    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=policy.language_tokenizer.pad_token_id).to('cuda')
    positions = pad_and_cat(positions).to('cuda')
    attention_mask = ids.ne(policy.language_tokenizer.pad_token_id)
    model = policy.model
    torch.manual_seed(args.seed)
    noise = model.sample_noise((1, config.n_action_steps, config.max_action_dim), 'cuda')

    print('Image grids:', grids.tolist(), flush=True)
    saved_context = torch.autograd.graph.save_on_cpu(pin_memory=True) if args.activation_offload else nullcontext()
    with saved_context, torch.autocast('cuda', dtype=torch.bfloat16):
        embed = model.paligemma_with_expert.qwen.model(
            pixel_values=pixels, image_grid_thw=grids, input_ids=ids,
            position_ids=positions, attention_mask=attention_mask, use_cache=False,
        ).last_hidden_state
        actions = sample_differentiable(model, state.clone(), embed, noise.clone())
        # Keep the repository's action unnormalization formula but remove only
        # its no_grad wrapper so the Jacobian reaches the image tensor.
        real_actions = type(policy.unnormalize_outputs).forward.__wrapped__(
            policy.unnormalize_outputs, {'action': actions[..., :config.action_feature.shape[0]]})['action']
    assert real_actions.shape == (1, config.n_action_steps, config.action_feature.shape[0])
    flat_outputs = real_actions.reshape(-1)
    if args.action_end is None:
        args.action_end = flat_outputs.numel()
    if not (0 <= args.action_start < args.action_end <= flat_outputs.numel()):
        ap.error(f'action range must be within [0,{flat_outputs.numel()}), got [{args.action_start},{args.action_end})')
    print(f'Action output dimension: {flat_outputs.numel()}, worker range [{args.action_start},{args.action_end})', flush=True)

    # Verify against repository-native inference before spending 1200 VJPs.
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        reference = model.sample_actions_unified(state.clone(), embed.detach(), noise=noise.clone())
    error = float((reference-actions.detach()).abs().max())
    torch.testing.assert_close(reference, actions.detach(), atol=1e-5, rtol=1e-5)
    print('Native sampler max error:', error, flush=True)
    tiling_check = []
    if args.verify_tiling:
        for index in sorted({args.action_start, (args.action_start+args.action_end)//2, args.action_end-1}):
            direct = torch.autograd.grad(flat_outputs[index], stitched, retain_graph=True)
            repeated = torch.autograd.grad(flat_outputs[index], stitched, retain_graph=True)
            routed = torch.autograd.grad(stitched, tile_inputs, grad_outputs=direct, retain_graph=True)
            for piece,(camera,y,x,h,w) in zip(routed,tile_locations):
                torch.testing.assert_close(piece,direct[camera][:,y:y+h,x:x+w],rtol=0,atol=0)
            pieces = torch.autograd.grad(flat_outputs[index], tile_inputs, retain_graph=True)
            restored = [torch.zeros_like(x) for x in stitched]
            for piece,(camera,y,x,h,w) in zip(pieces,tile_locations):
                restored[camera][:,y:y+h,x:x+w] = piece
            delta = max(float((x-y).abs().max()) for x,y in zip(direct,restored))
            denom = sum(float(x.square().sum()) for x in direct)**.5
            relative = sum(float((x-y).square().sum()) for x,y in zip(direct,restored))**.5/max(denom,1e-30)
            repeat_relative = sum(float((x-y).square().sum()) for x,y in zip(direct,repeated))**.5/max(denom,1e-30)
            if relative > max(.01, 3*repeat_relative):
                raise RuntimeError(f'Tiled gradient relative error {relative} exceeds repeat noise {repeat_relative}')
            tiling_check.append({'action_index':index,'max_abs_gradient_error':delta,
                                 'relative_l2_error':relative, 'repeated_untiled_relative_l2_error':repeat_relative,
                                 'stitch_routing_max_error':0.0})
        del direct, repeated, routed, pieces, restored
        print('Full-image/tiled gradient checks:',tiling_check,flush=True)
    def progress(end, total):
        if end == total or end % 20 == 0:
            print(f'Jacobian VJP {end}/{total}, elapsed {time.time()-t0:.1f}s', flush=True)
    torch.cuda.synchronize()
    backward_start = time.perf_counter()
    tile_sums = jacobian_norm(real_actions, tile_inputs, args.vjp_chunk, progress,
                             args.action_start, args.action_end, args.tiles_per_backward, squared=True)
    squared_maps = stitch_tile_sums(tile_sums, tile_locations, image_inputs)
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - backward_start
    run_details = {
        'action_range': [args.action_start, args.action_end],
        'total_action_outputs': flat_outputs.numel(), 'vjp_chunk': args.vjp_chunk,
        'image_tile_size': args.image_tile_size, 'tile_count': len(tile_inputs),
        'tiles_per_backward': args.tiles_per_backward, 'activation_offload': args.activation_offload,
        'backward_seconds': backward_seconds,
        'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
        'sampler_max_abs_error': error, 'seed': args.seed, 'episode': args.episode, 'frame': args.frame,
        'checkpoint': str(CHECKPOINT), 'architecture_repo': str(REPO),
        'checkpoint_index_sha256': hashlib.sha256((CHECKPOINT/'model.safetensors.index.json').read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256(baseline_pixels.cpu().numpy().tobytes()+state.cpu().numpy().tobytes()+ids.cpu().numpy().tobytes()).hexdigest(),
        'task': raw['task'], 'seconds': time.time()-t0,
        'image_size': list(config.observation_config.image_size),
        'wrist_image_size': list(config.observation_config.wrist_image_size),
        'image_resize_mode': config.observation_config.image_resize_mode,
        'tiling_gradient_verification': tiling_check,
        'labels': list(mapping.values()), 'cameras': sources,
    }
    if args.partial_output is not None:
        # Only native-grid squared sums are merged. Interpolation and sqrt do
        # not commute: never merge separately interpolated worker norm maps.
        args.partial_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.partial_output,
            **{label+'_squared': x.cpu().numpy() for label, x in zip(mapping.values(), squared_maps)},
            **{label+'_original': originals[key] for key,label in mapping.items()},
            **{label+'_input': image.detach().cpu().numpy() for label,image in zip(mapping.values(), image_inputs)},
            predicted_actions=real_actions.detach().float().cpu().numpy())
        args.partial_output.with_suffix('.json').write_text(json.dumps(run_details, indent=2))
        print('Saved partial:', args.partial_output, flush=True)
        return
    heatmaps = [x.sqrt() for x in squared_maps]

    # Convert model patch pixels to spatial heatmaps, one map per camera.
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    norm_maps = {}
    camera_stats = {}
    for key, grid, heat_tensor, image_tensor in zip(mapping, grids.tolist(), heatmaps, image_inputs, strict=True):
        label = mapping[key]
        heat = heat_tensor.cpu().numpy()
        norm_maps[label + '_model_input'] = heat
        Image.fromarray((image_tensor.detach().permute(1,2,0).cpu().numpy()*255).clip(0,255).astype('uint8')).save(out/f'{label}_model_input.png')
        rgb = originals[key]
        size = config.observation_config.wrist_image_size if 'wrist' in key else config.observation_config.image_size
        projected = torch.nn.functional.interpolate(
            torch.from_numpy(heat)[None, None], size=tuple(size), mode='bilinear', align_corners=False
        )[0, 0].numpy()
        # project_to_original handles the resize/padding used by the robot input.
        from run import project_to_original
        original_heat = project_to_original(projected, rgb.shape, size, config.observation_config.image_resize_mode)
        norm_maps[label + '_original'] = original_heat
        Image.fromarray(rgb).save(out / f'{label}_original.png')
        camera_stats[label] = {
            'grid_thw': grid, 'model_heat_shape': list(heat.shape),
            'original_shape': list(original_heat.shape), 'sum': float(original_heat.sum()),
            'max': float(original_heat.max()), 'mean': float(original_heat.mean()),
        }

    # One common display scale makes camera-to-camera intensities comparable.
    all_values = np.concatenate([v.reshape(-1) for k, v in norm_maps.items() if k.endswith('_original')])
    lo, hi = 0.0, float(np.percentile(all_values, 99.7))
    rows = [[], []]
    for key, label in mapping.items():
        rgb = originals[key]
        heat = norm_maps[label + '_original']
        rendered = overlay(rgb, heat, lo, hi)
        Image.fromarray(rendered).save(out / f'{label}_action_jacobian_norm.png')
        for row, image in zip(rows, [rgb, rendered]):
            panel = Image.fromarray(image)
            panel.thumbnail((480, 320))
            tile = Image.new('RGB', (480, 350), 'white')
            tile.paste(panel, ((480 - panel.width) // 2, 27))
            ImageDraw.Draw(tile).text((12, 8), label, fill='black')
            row.append(tile)
    canvas = Image.new('RGB', (1440, 820), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), f'Spirit Qwen3-VL | checkpoint 20000 | episode {args.episode} frame {args.frame}', fill='black')
    draw.text((15, 32), raw['task'], fill='black')
    draw.text((15, 52), 'Per-pixel L2 norm of the full action Jacobian | 60 steps x 20 action dims | common display scale', fill='black')
    for r, title in enumerate(['Original camera frames', 'Action-Jacobian pixel-gradient norm']):
        y = 80 + r * 370
        draw.text((15, y), title, fill='black')
        for c, tile in enumerate(rows[r]):
            canvas.paste(tile, (480 * c, y + 25))
    canvas.save(out / 'comparison.png')
    np.savez_compressed(out / 'action_jacobian_norm.npz', **norm_maps)
    np.save(out / 'predicted_actions.npy', real_actions.detach().float().cpu().numpy())
    metadata = {
        'method': 'exact action Jacobian pixel-gradient L2 norm',
        'formula': 'sqrt(sum_{t,d,c}(d action[t,d] / d model_pixel[c])^2)',
        'checkpoint_step': 20000, 'checkpoint': str(CHECKPOINT), 'architecture_repo': str(REPO),
        'dataset': str(DATA), 'episode': args.episode, 'frame': args.frame, 'task': raw['task'],
        'cameras': sources, 'action_shape': list(real_actions.shape), 'vjp_chunk': args.vjp_chunk,
        'seed': args.seed, 'model_pixel_definition': 'unique resized RGB pixels in [0,1], before Qwen normalization/temporal duplication/patch packing',
        'sampler_max_abs_error': error, 'action_units': 'checkpoint unnormalized physical output units; dimensions have mixed units',
        'patch_size': 16, 'temporal_patch_size': 2, 'display_percentile_hi': 99.7,
        'camera_stats': camera_stats, 'seconds': time.time() - t0, 'execution': run_details,
        'note': 'Exact full-output Jacobian norm at effective model-input pixels. Original-frame heatmaps are display interpolations, not derivatives with respect to original full-resolution video pixels.'
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print('Saved:', out / 'comparison.png', flush=True)


if __name__ == '__main__':
    main()
