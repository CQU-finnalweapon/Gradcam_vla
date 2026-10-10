#!/usr/bin/env python3
"""Independent GPU workers partition action rows; merge native pixel square sums."""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from run_spirit_action_jacobian import parse_tile_size, overlay
from run import project_to_original

HERE = Path(__file__).resolve().parent


def partition_outputs(total, workers):
    if workers < 1 or total < workers:
        raise ValueError('Each GPU must receive at least one output')
    return [(total*i//workers, total*(i+1)//workers) for i in range(workers)]


def merge_partials(paths, total):
    """Reject incomplete/overlapping/mismatched jobs before producing any map."""
    cursor, sums, first, records, assets = 0, {}, None, [], {}
    invariants = ('total_action_outputs', 'seed', 'episode', 'frame', 'checkpoint_index_sha256',
                  'input_sha256', 'task', 'labels', 'image_size', 'wrist_image_size', 'image_resize_mode')
    for path in paths:
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['action_range'][0] != cursor or meta['action_range'][1] <= cursor:
            raise ValueError('Action ranges overlap or leave a gap')
        cursor = meta['action_range'][1]
        if first is None:
            first = meta
        elif any(meta[k] != first[k] for k in invariants):
            raise ValueError('Workers used different inputs, checkpoint, or configuration')
        with np.load(path, allow_pickle=False) as data:
            actions = data['predicted_actions']
            if not np.isfinite(actions).all():
                raise ValueError('Invalid predicted actions')
            if not records:
                assets = {k: data[k].copy() for k in data.files if not k.endswith('_squared')}
            else:
                np.testing.assert_allclose(actions, assets['predicted_actions'], rtol=1e-5, atol=1e-5)
            for label in first['labels']:
                key = label+'_squared'
                value = data[key]
                if not np.isfinite(value).all() or np.any(value < 0):
                    raise ValueError('Invalid squared Jacobian map')
                if key not in sums:
                    sums[key] = np.zeros_like(value, dtype=np.float64)
                if sums[key].shape != value.shape:
                    raise ValueError('Worker map shape mismatch')
                sums[key] += value
        records.append(meta)
    if cursor != total or first['total_action_outputs'] != total:
        raise ValueError('Partial results do not cover the complete action vector')
    maps = {k.removesuffix('_squared')+'_model_input': np.sqrt(v).astype(np.float32) for k,v in sums.items()}
    return maps, assets, records


def render(out, maps, assets, meta):
    rows = [[], []]
    for label in meta['labels']:
        rgb = assets[label+'_original']
        size = meta['wrist_image_size'] if 'wrist' in label else meta['image_size']
        maps[label+'_original'] = project_to_original(maps[label+'_model_input'], rgb.shape, size, meta['image_resize_mode'])
        Image.fromarray(rgb).save(out/f'{label}_original.png')
        im = assets[label+'_input'].transpose(1,2,0)
        Image.fromarray((im*255).clip(0,255).astype('uint8')).save(out/f'{label}_model_input.png')
    hi = float(np.percentile(np.concatenate([maps[x+'_original'].ravel() for x in meta['labels']]), 99.7))
    for label in meta['labels']:
        rgb = assets[label+'_original']
        rendered = overlay(rgb, maps[label+'_original'], 0., hi)
        Image.fromarray(rendered).save(out/f'{label}_action_jacobian_norm.png')
        for row, img in zip(rows, [rgb, rendered]):
            panel = Image.fromarray(img); panel.thumbnail((480,320))
            tile = Image.new('RGB', (480,350), 'white')
            tile.paste(panel, ((480-panel.width)//2,27))
            ImageDraw.Draw(tile).text((12,8), label, fill='black')
            row.append(tile)
    canvas = Image.new('RGB',(1440,820),'white'); draw = ImageDraw.Draw(canvas)
    draw.text((15,12), f"Full action Jacobian | checkpoint 20000 | episode {meta['episode']} frame {meta['frame']}", fill='black')
    draw.text((15,32), meta['task'], fill='black')
    draw.text((15,52), 'Multi-GPU output partition | full image context | common display scale', fill='black')
    for r,title in enumerate(['Original camera frames','Full action Jacobian pixel norm']):
        y=80+r*370; draw.text((15,y),title,fill='black')
        for c,tile in enumerate(rows[r]): canvas.paste(tile,(480*c,y+25))
    canvas.save(out/'comparison.png')
    np.savez_compressed(out/'action_jacobian_norm.npz',**maps)
    np.save(out/'predicted_actions.npy',assets['predicted_actions'])
    return hi


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpus', required=True, help='Explicit CUDA IDs/UUIDs, e.g. 3,6')
    ap.add_argument('--vjp-chunk', type=int, default=1)
    ap.add_argument('--image-tile-size', type=parse_tile_size, default=None, help='full or HxW in effective input pixels')
    ap.add_argument('--tiles-per-backward', type=int, default=0, help='0=all; smaller input groups repeat backward')
    ap.add_argument('--activation-offload', action='store_true')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--frame', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--merge-only', type=Path, help='Merge saved worker NPZ/JSON in this job directory without recomputing')
    ap.add_argument('--reference-relative-l2', type=float, default=0.002, help='Explicit BF16 numerical comparison tolerance, default 0.2 percent relative L2')
    ap.add_argument('--reference-max-abs', type=float, default=1e-4)
    ap.add_argument('--reference', type=Path, help='Previous full single-GPU result directory; verify all heatmaps/actions')
    ap.add_argument('--output', type=Path, default=HERE/'results/pourcorn_step20000_ep0_frame150_action_jacobian_multigpu')
    args=ap.parse_args()
    gpus=[s.strip() for s in args.gpus.split(',') if s.strip()]
    if not gpus or len(set(gpus)) != len(gpus) or args.vjp_chunk<1 or args.tiles_per_backward<0 or args.reference_relative_l2<0 or args.reference_max_abs<0:
        ap.error('GPUs must be nonempty/unique and chunk sizes valid')
    # Read actual policy dimensions; no hard-coded output count or silent subset.
    config=json.loads((HERE/'checkpoints/pourcorn_step20000/config.json').read_text())
    total=config['n_action_steps']*config['output_features']['action']['shape'][0]
    ranges=partition_outputs(total,len(gpus))
    args.output.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    if args.merge_only:
        paths=[args.merge_only/f'worker{i}.npz' for i in range(len(gpus))]
        timing_path=args.merge_only/'job_timing.json'
        worker_seconds=json.loads(timing_path.read_text())['worker_wall_seconds'] if timing_path.exists() else None
    else:
        job=args.output/'partials'/uuid.uuid4().hex[:12]; job.mkdir(parents=True)
        procs, paths, handles = [], [], []
        t0=time.perf_counter()
        try:
            for rank,(gpu,(start,end)) in enumerate(zip(gpus,ranges)):
                path=job/f'worker{rank}.npz'; paths.append(path)
                fh=path.with_suffix('.log').open('w'); handles.append(fh)
                env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                tile='full' if args.image_tile_size is None else 'x'.join(map(str,args.image_tile_size))
                cmd=[sys.executable,'-u',str(HERE/'run_spirit_action_jacobian.py'),
                     '--action-start',str(start),'--action-end',str(end),'--partial-output',str(path),
                     '--episode',str(args.episode),'--frame',str(args.frame),'--seed',str(args.seed),
                     '--vjp-chunk',str(args.vjp_chunk),'--image-tile-size',tile,
                     '--tiles-per-backward',str(args.tiles_per_backward)]
                if args.activation_offload: cmd.append('--activation-offload')
                procs.append(subprocess.Popen(cmd,env=env,stdout=fh,stderr=subprocess.STDOUT))
                print(f'GPU {gpu}: actions [{start},{end}), log {path.with_suffix(".log")}',flush=True)
            report=time.perf_counter()
            while True:
                codes=[p.poll() for p in procs]
                for rank,code in enumerate(codes):
                    if code is not None and code!=0:
                        log=paths[rank].with_suffix('.log')
                        raise RuntimeError(f'GPU {gpus[rank]} exited {code}: {log}\n'+log.read_text()[-3000:])
                if all(code is not None for code in codes): break
                if time.perf_counter()-report>=30:
                    for gpu,path in zip(gpus,paths):
                        lines=path.with_suffix('.log').read_text().splitlines()
                        print(f'GPU {gpu}: {lines[-1] if lines else "starting"}',flush=True)
                    report=time.perf_counter()
                time.sleep(1)
        finally:
            for p in procs:
                if p.poll() is None: p.terminate()
            for p in procs:
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
            for fh in handles: fh.close()
        worker_seconds=time.perf_counter()-t0
        (job/'job_timing.json').write_text(json.dumps({'worker_wall_seconds':worker_seconds,'gpus':gpus}))
    maps,assets,records=merge_partials(paths,total)
    hi=render(args.output,maps,assets,records[0])
    verification={}
    validation_errors=[]
    if args.reference:
        reference_meta=json.loads((args.reference/'metadata.json').read_text())
        for k in ('episode','frame','seed'):
            if reference_meta[k]!=getattr(args,k): raise ValueError('Reference sample mismatch')
        with np.load(args.reference/'action_jacobian_norm.npz') as ref:
            for k,value in maps.items():
                abs_error=float(np.max(np.abs(value-ref[k])))
                rel_l2=float(np.linalg.norm(value-ref[k])/np.linalg.norm(ref[k]))
                passed=abs_error <= args.reference_max_abs and rel_l2 <= args.reference_relative_l2
                if not passed:
                    validation_errors.append(k)
                verification[k]={'max_abs_error':abs_error, 'relative_l2_error':rel_l2, 'passed':passed,
                    'strict_previous_allclose':bool(np.allclose(value,ref[k],atol=1e-7,rtol=2e-4))}
        np.testing.assert_allclose(assets['predicted_actions'],np.load(args.reference/'predicted_actions.npy'),atol=1e-5,rtol=1e-5)
    meta={'method':'full action Jacobian norm, action rows partitioned across GPUs',
          'gpus':gpus,'workers':records,'worker_wall_seconds':worker_seconds,
          'max_worker_seconds':max(r['seconds'] for r in records),
          'seconds':time.perf_counter()-t0,'merge_only':bool(args.merge_only),
          'reference_thresholds':{'max_abs':args.reference_max_abs,'relative_l2':args.reference_relative_l2},
          'reference_passed':not validation_errors if args.reference else None,
          'reduction':'sum raw squared gradients on native pixel grid, sqrt once, then interpolate',
          'display_percentile':99.7,'display_upper_value':hi,'reference_verification':verification,
          'reference':str(args.reference) if args.reference else None,
          'image_tile_note':'Real input leaves stitched into full images; batching tiles reduces returned gradient buffers, not the global Transformer graph.'}
    (args.output/'multigpu_metadata.json').write_text(json.dumps(meta,indent=2))
    if validation_errors:
        raise ValueError(f'Reference comparison failed for {validation_errors}; see multigpu_metadata.json')
    print(f'Saved {args.output/"comparison.png"}; this invocation {meta["seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    main()
