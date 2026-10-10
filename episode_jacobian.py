#!/usr/bin/env python3
"""Signed image/physical-state Jacobians for one action time step, per episode frame."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import zipfile

import cv2
import numpy as np
import torch

from run_spirit_action_jacobian import (
    HERE, REPO, DATA, CHECKPOINT, ASSETS, sample_differentiable,
    unpack_rgb, pack_rgb, tile_images, parse_tile_size,
)
from run import CAMERA_ALIASES, camera_mapping, load_sharded_safetensors

SCHEMA = 1


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(temp, path)


def atomic_npz(path, arrays, compressed=False):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with temp.open('wb') as stream:
            (np.savez_compressed if compressed else np.savez)(stream, **arrays)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def differentiable_normalize(module, values):
    """Use the repository's actual transform, bypassing only its no_grad wrapper."""
    method = type(module).forward
    return getattr(method, '__wrapped__', method)(module, values)


class EpisodeReader:
    """One parquet episode and three synchronized video streams (LeRobot v2 layout)."""
    def __init__(self, root, episode, mapping):
        import pyarrow.parquet as pq
        self.root, self.episode, self.mapping = Path(root), episode, mapping
        self.info = json.loads((self.root/'meta/info.json').read_text())
        context = {'episode_chunk': episode//self.info.get('chunks_size',1000), 'episode_index':episode}
        self.data_path = self.root/self.info['data_path'].format(**context)
        self.rows = pq.read_table(self.data_path).to_pylist()
        if not self.rows:
            raise ValueError('Empty episode')
        for i, row in enumerate(self.rows):
            if int(row['frame_index']) != i or int(row['episode_index']) != episode:
                raise ValueError('Expected contiguous episode-local frame_index starting at zero')
        self.tasks = {int(r['task_index']):r['task'] for r in
                      (json.loads(line) for line in (self.root/'meta/tasks.jsonl').read_text().splitlines() if line.strip())}
        self.fps = float(self.info['fps'])
        self.paths, self.caps = {}, {}
        self.next_frame = None
        for key, label in mapping.items():
            candidates = (key.removeprefix('observation.images.'),) + CAMERA_ALIASES[label]
            video_key = next((k for name in candidates for k in (name,'observation.images.'+name)
                              if k in self.info['features'] and self.info['features'][k]['dtype']=='video'), None)
            if video_key is None:
                raise ValueError(f'Missing video camera {label}; map dataset cameras to checkpoint inputs')
            self.paths[label] = str((self.root/self.info['video_path'].format(**context,video_key=video_key)).resolve())

    def read(self, frame):
        row = self.rows[frame]
        raw = {k:np.asarray(v,dtype=np.float32) if isinstance(v,list) else v for k,v in row.items()}
        originals = {}
        for key,label in self.mapping.items():
            if label not in self.caps:
                self.caps[label] = cv2.VideoCapture(self.paths[label])
            cap = self.caps[label]
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open {self.paths[label]}')
            if self.next_frame != frame:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame):
                    raise RuntimeError(f'Video seek failed for {label}, frame {frame}')
            ok, bgr = cap.read()
            if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES))-1 != frame:
                raise RuntimeError(f'Video/frame alignment failure {label}, frame {frame}')
            rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            raw[key] = rgb
            raw[key.removeprefix('observation.images.')] = rgb
            originals[label] = rgb
        self.next_frame = frame+1
        raw['task'] = self.tasks[int(row['task_index'])]
        return raw, originals

    def close(self):
        for cap in self.caps.values(): cap.release()


def vector_labels(config, state=False):
    from lerobot.common.robot_devices.transform_utils import IndexConfig
    obs=config.observation_config
    dim=config.input_features['observation.state'].shape[0] if state else config.action_feature.shape[0]
    labels=[f'{"state" if state else "action"}[{i}]' for i in range(dim)]
    components=IndexConfig.get_component_info(obs.robot_type,obs.moz1_structure,
        obs.state_space if state else obs.action_space,obs.gripper_first,
        False if state else getattr(obs,'use_stop_signal',False))
    for name,item in components.items():
        for index in range(*item['range']):
            if index<dim:
                label=name.replace('_cmd_', '_state_') if state else name
                labels[index]=f'{label}[{index-item["range"][0]}]'
    return labels


class SpiritEngine:
    def __init__(self,args):
        sys.path.insert(0,str(args.repo))
        from mozinfer.spirit_qwen_dit_cotrain_policy import SpiritQwenDiTCotrainPolicy, pad_and_cat
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.common.robot_devices.transform_utils import ObservationStandardizer
        from transformers import AutoProcessor, AutoImageProcessor
        from transformers.modeling_utils import no_init_weights
        from types import SimpleNamespace
        config=PreTrainedConfig.from_pretrained(str(args.checkpoint))
        if config.type != 'spirit_qwen_dit_cotrain':
            raise ValueError('This adapter requires a Spirit Qwen3-VL co-training policy')
        if not 0<=args.action_step<config.n_action_steps:
            raise ValueError('action-step outside checkpoint action chunk')
        config.backbone=str(args.assets)
        config.tokenizer_path=str(args.checkpoint/'tokenizer')
        config.pretrained_path=str(args.checkpoint)
        config.vlm_use_sac=False
        config.attention_implementation='eager'
        config.device='cuda'
        loader=AutoProcessor.from_pretrained
        AutoProcessor.from_pretrained=lambda *a,**kw:SimpleNamespace(image_processor=AutoImageProcessor.from_pretrained(str(args.assets),local_files_only=True))
        try:
            with torch.device('cpu'),no_init_weights():
                policy=SpiritQwenDiTCotrainPolicy(config)
        finally:
            AutoProcessor.from_pretrained=loader
        index=args.checkpoint/'model.safetensors.index.json'
        if index.exists():
            weights=load_sharded_safetensors(index)
        else:
            from safetensors.torch import load_file
            weights=load_file(str(args.checkpoint/'model.safetensors'))
        expected=policy.state_dict()
        if set(expected)!=set(weights) or any(expected[k].shape!=weights[k].shape for k in expected):
            raise ValueError('Checkpoint keys/shapes do not match the selected architecture')
        weights={k:v.clone() for k,v in weights.items()}
        policy.load_state_dict(weights,strict=True,assign=True)
        self.policy=policy.cuda().eval().requires_grad_(False)
        del weights,expected
        for module in (policy.normalize_inputs,policy.unnormalize_outputs):
            if not all(torch.isfinite(x).all().item() for x in module.state_dict().values()):
                raise ValueError('Nonfinite checkpoint normalization statistics')
        self.model=policy.model
        self.model.paligemma_with_expert.qwen.gradient_checkpointing_disable()
        self.standardizer=ObservationStandardizer(config.observation_config)
        self.mapping=camera_mapping(policy)
        self.config=config
        self.pad_and_cat=pad_and_cat
        self.processor=self.model.paligemma_with_expert.image_processor
        # Reinitialize the same generator on each worker, only once, never advance per frame.
        torch.manual_seed(args.seed)
        self.noise=self.model.sample_noise((1,config.n_action_steps,config.max_action_dim),'cuda')
        self.noise_sha=tensor_hash(self.noise)
        self.state_labels=vector_labels(config,True)
        self.action_labels=vector_labels(config,False)

    def calculate(self,raw,originals,args):
        start=time.perf_counter()
        policy=self.policy
        obs=self.standardizer.standardize_camera_features(raw)
        if 'observation.state' in raw:
            packed=torch.as_tensor(raw['observation.state'],dtype=torch.float32)
        else:
            packed=self.standardizer.pack_state(raw,get_mask=False)
        if tuple(packed.shape)!=tuple(self.config.input_features['observation.state'].shape):
            raise ValueError('Dataset physical state shape does not match checkpoint')
        physical=packed.to('cuda').detach().requires_grad_(True)
        if not torch.isfinite(physical).all(): raise ValueError('Nonfinite physical state')
        batch={k:v[None].cuda() for k,v in obs.items()}
        batch['observation.state']=physical[None]
        batch['task']=[raw['task']]
        batch=differentiable_normalize(policy.normalize_inputs,batch)
        state=policy.prepare_state(batch)
        pixels,grids,ids,positions,_=policy.preprocess_rb_batch(batch)
        reference_pixels=torch.cat(pixels).contiguous()
        grids=torch.cat(grids).contiguous()
        inputs=[]; offset=0
        for grid in grids.tolist():
            count=int(np.prod(grid))
            inputs.append(unpack_rgb(reference_pixels[offset:offset+count],grid,self.processor))
            offset+=count
        tiles,locations,full=tile_images(inputs,args.image_tile_size)
        pixels=torch.cat([pack_rgb(x,self.processor) for x in full])
        torch.testing.assert_close(pixels.detach(),reference_pixels,rtol=1e-6,atol=1e-6)
        ids=torch.nn.utils.rnn.pad_sequence(ids,batch_first=True,padding_value=policy.language_tokenizer.pad_token_id).cuda()
        positions=self.pad_and_cat(positions).cuda()
        mask=ids.ne(policy.language_tokenizer.pad_token_id)
        ctx=torch.autograd.graph.save_on_cpu(pin_memory=True) if args.activation_offload else nullcontext()
        with ctx,torch.autocast('cuda',dtype=torch.bfloat16):
            embed=self.model.paligemma_with_expert.qwen.model(pixel_values=pixels,image_grid_thw=grids,
                input_ids=ids,position_ids=positions,attention_mask=mask,use_cache=False).last_hidden_state
            actions=sample_differentiable(self.model,state.clone(),embed,self.noise.clone())
            real=differentiable_normalize(policy.unnormalize_outputs,{'action':actions[...,:self.config.action_feature.shape[0]]})['action']
            selected=real[0,args.action_step]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            native=self.model.sample_actions_unified(state.detach().clone(),embed.detach(),noise=self.noise.clone())
        torch.testing.assert_close(actions.detach(),native,atol=1e-5,rtol=1e-5)
        error=float((actions.detach()-native).abs().max())
        arrays=raw_jacobian(selected,tiles,physical,locations,inputs,args.vjp_chunk,args.tiles_per_backward)
        image_jacs,state_jac=arrays
        result={}
        for (key,label),image,jac in zip(self.mapping.items(),inputs,image_jacs):
            result['jac_image_'+label]=jac
            result['image_'+label]=(image.detach().cpu().numpy().transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
            result['original_'+label]=originals[label]
        result.update(jac_state=state_jac,physical_state=physical.detach().cpu().numpy(),
            selected_action=selected.detach().float().cpu().numpy(),predicted_actions=real.detach().float().cpu().numpy(),
            state_labels=np.asarray(self.state_labels),action_labels=np.asarray(self.action_labels))
        return result,{'sampler_max_abs_error':error,'seconds_compute':time.perf_counter()-start,
            'noise_sha256':self.noise_sha,'state_masked_indices':[2,9] if self.config.use_state else [],
            'image_grids':grids.tolist(),'task':raw['task'],
            'image_size':list(self.config.observation_config.image_size),
            'wrist_image_size':list(self.config.observation_config.wrist_image_size),
            'image_resize_mode':self.config.observation_config.image_resize_mode}


def raw_jacobian(outputs,tiles,physical,locations,images,chunk=1,input_chunk=0):
    """Keep all signed d action / d image and d action / d physical state."""
    count=outputs.numel()
    image_jacs=[np.empty((count,*image.shape),dtype=np.float32) for image in images]
    state_jac=np.empty((physical.numel(),count),dtype=np.float32)
    leaves=list(tiles)+[physical]
    # Include state as one additional input group. It must not be detached.
    width=input_chunk or len(leaves)
    for first in range(0,count,chunk):
        last=min(first+chunk,count)
        basis=torch.zeros((last-first,count),device=outputs.device,dtype=outputs.dtype)
        basis[torch.arange(last-first,device=outputs.device),torch.arange(first,last,device=outputs.device)]=1
        for lo in range(0,len(leaves),width):
            hi=min(lo+width,len(leaves))
            retain=last<count or hi<len(leaves)
            grads=torch.autograd.grad(outputs,leaves[lo:hi],grad_outputs=basis if chunk>1 else basis[0],
                is_grads_batched=chunk>1,retain_graph=retain,allow_unused=True)
            for k,grad in enumerate(grads,lo):
                if grad is None:
                    grad=torch.zeros((last-first,*leaves[k].shape),device=outputs.device)
                elif chunk==1: grad=grad[None]
                value=grad.detach().float().cpu().numpy()
                if not np.isfinite(value).all(): raise RuntimeError('Nonfinite Jacobian')
                if k==len(tiles): state_jac[:,first:last]=value.reshape(last-first,-1).T
                else:
                    camera,y,x,h,w=locations[k]
                    image_jacs[camera][first:last,:,y:y+h,x:x+w]=value
            del grads
    return image_jacs,state_jac


def valid_frame(path,run_id,frame,deep=False):
    if not path.exists(): return False
    try:
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            if meta['run_id']!=run_id or meta['frame']!=frame or meta['schema']!=SCHEMA:return False
            if deep:
                for key in data.files:
                    if key.startswith('jac_'):
                        x=data[key]
                        if x.dtype!=np.float32 or not np.isfinite(x).all(): return False
            return 'jac_state' in data.files and all('jac_image_'+label in data.files for label in meta['labels'])
    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile): return False


def worker(args):
    torch.set_num_threads(1)
    manifest=json.loads((args.output/'manifest.json').read_text())
    frames=json.loads(args.assignment.read_text())
    todo=[f for f in frames if not valid_frame(args.output/'frames'/f'frame_{f:06d}.npz',manifest['run_id'],f,True)]
    if not todo:
        print('All assigned frames already valid',flush=True); return
    engine=SpiritEngine(args)
    reader=EpisodeReader(args.dataset,args.episode,engine.mapping)
    try:
        for frame in todo:
            raw,originals=reader.read(frame)
            arrays,meta=engine.calculate(raw,originals,args)
            meta.update(schema=SCHEMA,run_id=manifest['run_id'],frame=frame,episode=args.episode,
                action_step=args.action_step,seed=args.seed,timestamp=float(reader.rows[frame]['timestamp']),
                labels=list(engine.mapping.values()),fps=reader.fps,
                image_jacobian_axes=['action','rgb','height','width'],state_jacobian_axes=['physical_state','action'],
                image_coordinates='effective model RGB pixels [0,1] before normalization',
                state_coordinates='packed physical state before normalization; model masks preserved',
                action_coordinates='unnormalized policy output, including delta representation from checkpoint')
            arrays['metadata']=np.asarray(json.dumps(meta,ensure_ascii=False))
            path=args.output/'frames'/f'frame_{frame:06d}.npz'
            atomic_npz(path,arrays,args.compress)
            print(f'Frame {frame} saved ({meta["seconds_compute"]:.2f}s compute, {path.stat().st_size/1e6:.1f} MB)',flush=True)
            del arrays,raw,originals
    finally: reader.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name,default in [('repo',REPO),('checkpoint',CHECKPOINT),('assets',ASSETS),('dataset',DATA)]:
        ap.add_argument('--'+name,type=Path,default=default)
    ap.add_argument('--episode',type=int,default=0)
    ap.add_argument('--action-step',type=int,default=10)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--gpus',default='6',help='GPU IDs, comma separated; contiguous frame ranges per persistent worker')
    ap.add_argument('--frames',help='Optional comma-separated frame IDs for sample/testing; default whole episode')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--vjp-chunk',type=int,default=1)
    ap.add_argument('--image-tile-size',type=parse_tile_size,default=None)
    ap.add_argument('--tiles-per-backward',type=int,default=0)
    ap.add_argument('--activation-offload',action='store_true')
    ap.add_argument('--compress',action='store_true',help='Lossless ZIP compression: lower disk space, extra CPU time')
    ap.add_argument('--assignment',type=Path,help=argparse.SUPPRESS)
    args=ap.parse_args()
    for name in ('repo','checkpoint','assets','dataset','output'):
        setattr(args,name,getattr(args,name).resolve())
    if args.vjp_chunk<1 or args.tiles_per_backward<0:ap.error('Invalid chunk sizes')
    if args.assignment:
        worker(args); return
    config=json.loads((args.checkpoint/'config.json').read_text())
    if not 0<=args.action_step<config['n_action_steps']:ap.error('action-step out of range')
    # Resolve cameras and schema before spawning expensive model processes.
    class C: pass
    policy=C(); policy.config=C()
    policy.config.image_features={k:v for k,v in config['input_features'].items() if v['type']=='VISUAL'}
    reader=EpisodeReader(args.dataset,args.episode,camera_mapping(policy))
    frames=sorted(set(map(int,args.frames.split(',')))) if args.frames else list(range(len(reader.rows)))
    if not frames or min(frames)<0 or max(frames)>=len(reader.rows):ap.error('frame selection outside episode')
    gpus=[g.strip() for g in args.gpus.split(',') if g.strip()]
    if not gpus or len(set(gpus))!=len(gpus):ap.error('GPUs must be unique/nonempty')
    identity={'schema':SCHEMA,'repo':str(args.repo),'checkpoint':str(args.checkpoint),'assets':str(args.assets),
        'dataset':str(args.dataset),'episode':args.episode,'action_step':args.action_step,'seed':args.seed,
        'config_sha256':hash_file(args.checkpoint/'config.json'),'parquet_sha256':hash_file(reader.data_path),
        'model_sources':{p:hash_file(args.repo/p) for p in ['mozinfer/spirit_qwen_dit_cotrain_policy.py',
            'lerobot/common/policies/qwen_dit/qwen_with_dit.py','lerobot/common/policies/qwen_dit/qwen3_action_expert.py',
            'lerobot/common/robot_devices/transform_utils.py','lerobot/common/policies/normalize.py']},
        'assets_sha256':{p:hash_file(args.assets/p) for p in ('config.json','preprocessor_config.json')},
        'weight_files':{p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
                        for p in sorted(args.checkpoint.glob('*.safetensors'))}}
    run_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest=dict(identity,run_id=run_id,episode_frames=len(reader.rows),fps=reader.fps,video_paths=reader.paths,
                  jacobian_dtype='float32',fixed_noise=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'frames').mkdir(exist_ok=True)
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())['run_id']!=run_id:
        raise ValueError('Output belongs to different inputs/configuration; choose a new output directory')
    atomic_json(path,manifest)
    job=args.output/'jobs'/uuid.uuid4().hex[:12];job.mkdir(parents=True)
    atomic_json(job/'request.json',dict(frames=frames,gpus=gpus,settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
    procs=[]; handles=[]; logs=[]; start=time.perf_counter()
    try:
        for rank,gpu in enumerate(gpus[:len(frames)]):
            n=min(len(gpus),len(frames)); assigned=frames[len(frames)*rank//n:len(frames)*(rank+1)//n]
            assignment=job/f'worker{rank}.json';atomic_json(assignment,assigned)
            log=job/f'worker{rank}.log';logs.append(log);fh=log.open('w');handles.append(fh)
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--assignment',str(assignment)]
            for key in ('repo','checkpoint','assets','dataset','episode','action_step','seed','output','vjp_chunk','tiles_per_backward'):
                cmd+=['--'+key.replace('_','-'),str(getattr(args,key))]
            if args.image_tile_size:cmd+=['--image-tile-size','x'.join(map(str,args.image_tile_size))]
            if args.activation_offload:cmd+=['--activation-offload']
            if args.compress:cmd+=['--compress']
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            procs.append(subprocess.Popen(cmd,env=env,stdout=fh,stderr=subprocess.STDOUT))
            print(f'GPU {gpu}: {len(assigned)} frames, log {log}',flush=True)
        last=0
        while any(p.poll() is None for p in procs):
            for i,p in enumerate(procs):
                if p.poll() not in (None,0):raise RuntimeError(f'Worker failed: {logs[i]}\n{logs[i].read_text()[-2500:]}')
            if time.perf_counter()-last>30:
                for log in logs:
                    lines=log.read_text().splitlines();print(log.name+': '+(lines[-1] if lines else 'starting'),flush=True)
                last=time.perf_counter()
            time.sleep(1)
        if any(p.returncode!=0 for p in procs):raise RuntimeError('Worker failed; inspect job logs')
    finally:
        for p in procs:
            if p.poll() is None:p.terminate()
        for p in procs:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for fh in handles:fh.close()
    complete=[f for f in range(len(reader.rows)) if valid_frame(args.output/'frames'/f'frame_{f:06d}.npz',run_id,f)]
    if not set(frames).issubset(complete):raise RuntimeError('Missing or invalid requested frames')
    atomic_json(args.output/'status.json',dict(complete_frames=complete,episode_frames=len(reader.rows),
        episode_complete=len(complete)==len(reader.rows),last_job_seconds=time.perf_counter()-start))
    print(f'Complete {len(complete)}/{len(reader.rows)} episode frames. {args.output}',flush=True)


if __name__=='__main__':main()
