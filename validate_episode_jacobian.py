#!/usr/bin/env python3
"""Stream-check a complete saved episode without loading a model or using a GPU."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from episode_jacobian import atomic_json


def validate(root):
    start = time.perf_counter()
    manifest = json.loads((root / 'manifest.json').read_text())
    paths = sorted((root / 'frames').glob('frame_*.npz'))
    expected_frames = list(range(manifest['episode_frames']))
    if [int(p.stem.split('_')[-1]) for p in paths] != expected_frames:
        raise ValueError('Expected exactly one NPZ for every episode frame')
    noise_hashes, shapes, extrema, durations = set(), {}, {}, []
    max_sampler_error = 0.0
    total_bytes = 0
    reference_labels = None
    previous_timestamp = None
    for frame, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            for key, value in [('run_id', manifest['run_id']), ('frame', frame),
                               ('episode', manifest['episode']), ('action_step', manifest['action_step']),
                               ('seed', manifest['seed'])]:
                if meta[key] != value:
                    raise ValueError(f'{path.name}: inconsistent {key}')
            if not np.isfinite(meta['timestamp']) or (previous_timestamp is not None and meta['timestamp'] <= previous_timestamp):
                raise ValueError(f'{path.name}: timestamps must increase')
            previous_timestamp = meta['timestamp']
            labels = (data['state_labels'].tolist(), data['action_labels'].tolist(), meta['labels'])
            if reference_labels is None:
                reference_labels = labels
            elif labels != reference_labels:
                raise ValueError(f'{path.name}: labels changed')
            state_dim, action_dim = len(labels[0]), len(labels[1])
            for key in ['physical_state', 'selected_action', 'predicted_actions']:
                if not np.isfinite(data[key]).all():
                    raise ValueError(f'{path.name}: nonfinite {key}')
            if data['physical_state'].shape != (state_dim,) or data['selected_action'].shape != (action_dim,):
                raise ValueError(f'{path.name}: vector shape mismatch')
            np.testing.assert_array_equal(data['selected_action'], data['predicted_actions'][0, manifest['action_step']])
            keys = ['jac_state'] + ['jac_image_' + label for label in meta['labels']]
            for key in keys:
                jac = data[key]
                if key == 'jac_state':
                    expected = (state_dim, action_dim)
                    masked = meta['state_masked_indices']
                    if masked and np.any(jac[masked] != 0):
                        raise ValueError(f'{path.name}: masked state derivatives are nonzero')
                else:
                    label = key.removeprefix('jac_image_')
                    image = data['image_' + label]
                    original = data['original_' + label]
                    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                        raise ValueError(f'{path.name}: malformed model image')
                    if original.ndim != 3 or original.shape[2] != 3 or original.dtype != np.uint8:
                        raise ValueError(f'{path.name}: malformed original image')
                    expected = (action_dim, 3, *image.shape[:2])
                if jac.shape != expected or jac.dtype != np.float32 or not np.isfinite(jac).all():
                    raise ValueError(f'{path.name}: invalid {key}')
                if key in shapes and shapes[key] != list(jac.shape):
                    raise ValueError(f'{path.name}: shape changed for {key}')
                shapes[key] = list(jac.shape)
                bounds = extrema.setdefault(key, [float('inf'), float('-inf')])
                bounds[0] = min(bounds[0], float(jac.min()))
                bounds[1] = max(bounds[1], float(jac.max()))
            noise_hashes.add(meta['noise_sha256'])
            error = meta['sampler_max_abs_error']
            if not np.isfinite(error):
                raise ValueError(f'{path.name}: nonfinite sampler error')
            max_sampler_error = max(max_sampler_error, error)
            durations.append(meta['seconds_compute'])
        total_bytes += path.stat().st_size
        if frame % 50 == 0:
            print(f'Validated {frame + 1}/{len(paths)}', flush=True)
    if len(noise_hashes) != 1:
        raise ValueError('Sampling noise differs across frames')
    return dict(valid=True, run_id=manifest['run_id'], episode_complete=True,
                frames=len(paths), action_step=manifest['action_step'],
                jacobian_shapes=shapes, signed_value_ranges=extrema,
                noise_sha256=next(iter(noise_hashes)),
                sampler_max_abs_error=max_sampler_error,
                masked_state_derivatives_zero=True, npz_bytes=total_bytes,
                seconds_compute_per_frame=dict(mean=float(np.mean(durations)),
                    median=float(np.median(durations)), min=min(durations), max=max(durations)),
                validation_seconds=time.perf_counter()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.input)
    atomic_json(args.input / 'validation.json', report)
    print(json.dumps(report, indent=2), flush=True)
