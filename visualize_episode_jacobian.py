#!/usr/bin/env python3
"""Render stored signed Jacobians as camera video or state/action matrices, no model needed."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

from run import project_to_original
from run_spirit_action_jacobian import overlay


def load_meta(path):
    with np.load(path,allow_pickle=False) as data:return json.loads(str(data['metadata']))


def image_norm(jac,actions=None):
    selected=jac if actions is None else jac[actions]
    return np.sqrt(np.sum(np.square(selected,dtype=np.float64),axis=(0,1))).astype(np.float32)


def state_matrix(jacs,mode='rms'):
    values=np.asarray(jacs,dtype=np.float64)
    if mode=='frame':return np.abs(values[0])
    if mode=='rms':return np.sqrt(np.mean(values**2,axis=0))
    raise ValueError(mode)


def select_files(root,frames,allow_partial):
    manifest=json.loads((root/'manifest.json').read_text())
    paths=sorted((root/'frames').glob('frame_*.npz'))
    by_frame={int(p.stem.split('_')[-1]):p for p in paths}
    wanted=sorted(set(map(int,frames.split(',')))) if frames else list(range(manifest['episode_frames']))
    missing=[f for f in wanted if f not in by_frame]
    if missing and not allow_partial:raise ValueError(f'Missing {len(missing)} frames; use explicit --frames or --allow-partial')
    result=[by_frame[f] for f in wanted if f in by_frame]
    if not result:raise ValueError('No frames selected')
    for path in result:
        meta=load_meta(path)
        if meta['run_id']!=manifest['run_id']:raise ValueError('Mixed run IDs in result directory')
    return manifest,result


def camera_panel(data,meta,labels,actions,upper=None,percentile=99.7,diagnostics=None):
    """Original two-row linear overlay; None upper means per-frame relative scale."""
    canvas=Image.new('RGB',(480*len(labels),740),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,8),f"Frame {meta['frame']} | action step {meta['action_step']} | image Jacobian norm",fill='black')
    heats={label:image_norm(data['jac_image_'+label],actions) for label in labels}
    relative=upper is None
    if relative:
        upper=max(float(np.percentile(np.concatenate([h.ravel() for h in heats.values()]),percentile)),1e-12)
    scale='Frame-relative' if relative else 'Global'
    draw.text((10,724),f'{scale} 0 .. {upper:.6g}, shared across cameras; '+
              ('colors not comparable across frames' if relative else 'display clipping only'),fill='black')
    for c,label in enumerate(labels):
        rgb=data['original_'+label]
        heat=heats[label]
        size=meta['wrist_image_size'] if 'wrist' in label else meta['image_size']
        projected=project_to_original(heat,rgb.shape,size,meta['image_resize_mode'])
        rendered=overlay(rgb,projected,0,upper)
        panels=[(rgb,' original'),(rendered,' '+scale.lower()+' sensitivity')]
        for r,(img,caption) in enumerate(panels):
            panel=Image.fromarray(img);panel.thumbnail((480,320))
            y=50+r*345
            canvas.paste(panel,(c*480+(480-panel.width)//2,y))
            draw.text((c*480+10,y-18),label+caption,fill='black')
        if diagnostics is not None:
            q=np.percentile(heat,[50,99,99.7])
            diagnostics.append(dict(frame=meta['frame'],camera=label,median=float(q[0]),
                p99=float(q[1]),p997=float(q[2]),maximum=float(heat.max()),
                nonzero_fraction=float(np.mean(heat>0)),color_upper=upper,normalization='relative' if relative else 'global'))
    return np.asarray(canvas)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--mode',choices=['camera','state-frame','state-rms','state-time'],required=True)
    ap.add_argument('--frames',help='Explicit comma-separated video frame IDs; default all episode frames')
    ap.add_argument('--allow-partial',action='store_true',help='Use available frames only; recorded as partial output')
    ap.add_argument('--cameras',default='all',help='all or camera aliases separated by commas')
    ap.add_argument('--actions',help='Optional comma-separated output action columns')
    ap.add_argument('--states',help='Optional comma-separated input state rows')
    ap.add_argument('--state-index',type=int,default=0,help='Input state row for state-time mode')
    ap.add_argument('--percentile',type=float,default=99.7)
    ap.add_argument('--vmax',type=float,help='Fixed color upper limit; camera mode requires --normalization global')
    ap.add_argument('--normalization',choices=['relative','global'],default='relative',help='Camera scale: per-frame percentile (default) or one percentile across all selected frames')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if not 0<args.percentile<=100 or (args.vmax is not None and args.vmax<=0):ap.error('Invalid color scale')
    if args.mode=='camera' and args.normalization=='relative' and args.vmax is not None:
        ap.error('--vmax fixes a global scale; choose --normalization global or omit --vmax')
    manifest,paths=select_files(args.input,args.frames,args.allow_partial)
    first=load_meta(paths[0])
    actions=list(map(int,args.actions.split(','))) if args.actions else None
    states=list(map(int,args.states.split(','))) if args.states else None
    with np.load(paths[0],allow_pickle=False) as d:
        action_labels=d['action_labels'].tolist();state_labels=d['state_labels'].tolist()
    if actions is not None and (not actions or min(actions)<0 or max(actions)>=len(action_labels)):ap.error('Action index out of range')
    if states is not None and (not states or min(states)<0 or max(states)>=len(state_labels)):ap.error('State index out of range')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.mode=='camera':
        diagnostics=[]
        panel_options=dict(percentile=args.percentile,diagnostics=diagnostics)
        labels=first['labels'] if args.cameras=='all' else args.cameras.split(',')
        if any(label not in first['labels'] for label in labels):ap.error('Unknown camera alias')
        # Deterministic bounded samples of spatial values; one scale across time/cameras.
        samples=[]
        if args.normalization=='relative':
            upper=None
        elif args.vmax is None:
            for path in paths:
                with np.load(path,allow_pickle=False) as data:
                    for label in labels:
                        value=image_norm(data['jac_image_'+label],actions).ravel()
                        samples.append(value[np.linspace(0,value.size-1,min(2048,value.size),dtype=int)])
            upper=max(float(np.percentile(np.concatenate(samples),args.percentile)),1e-12)
        else:upper=args.vmax
        if args.output.suffix.lower()=='.png':
            if len(paths)!=1:ap.error('PNG camera output requires exactly one selected frame')
            with np.load(paths[0],allow_pickle=False) as d:
                Image.fromarray(camera_panel(d,first,labels,actions,upper,**panel_options)).save(args.output)
        else:
            if args.output.suffix.lower()!='.mp4':ap.error('Camera output must be .png or .mp4')
            numbers=[load_meta(p)['frame'] for p in paths]
            if any(b!=a+1 for a,b in zip(numbers,numbers[1:])):
                ap.error('Video requires consecutive frames so playback retains episode timing')
            writer=cv2.VideoWriter(str(args.output),cv2.VideoWriter_fourcc(*'mp4v'),manifest['fps'],(480*len(labels),740))
            if not writer.isOpened():raise RuntimeError('Cannot open MP4 writer')
            try:
                for index,path in enumerate(paths):
                    with np.load(path,allow_pickle=False) as d:
                        rgb=camera_panel(d,json.loads(str(d['metadata'])),labels,actions,upper,**panel_options)
                    writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                    if index%30==0:print(f'Rendered {index+1}/{len(paths)}',flush=True)
            finally:writer.release()
        args.output.with_suffix('.diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        values=[];numbers=[];timestamps=[]
        for path in paths:
            with np.load(path,allow_pickle=False) as data:
                values.append(data['jac_state']);meta=json.loads(str(data['metadata']))
                numbers.append(meta['frame']);timestamps.append(meta['timestamp'])
        selected_actions=actions if actions is not None else list(range(len(action_labels)))
        if args.mode=='state-time':
            if not 0<=args.state_index<len(state_labels):ap.error('state-index out of range')
            matrix=np.abs(np.asarray(values)[:,args.state_index,selected_actions])
            row_labels=[f'{frame} / {tm:.2f}s' for frame,tm in zip(numbers,timestamps)]
            ylabel='Episode frame / timestamp'
            title=f'Input {state_labels[args.state_index]} sensitivity over time'
        else:
            if args.mode=='state-frame' and len(paths)!=1:ap.error('state-frame requires exactly one frame')
            matrix=state_matrix(values,'frame' if args.mode=='state-frame' else 'rms')
            selected_states=states if states is not None else list(range(len(state_labels)))
            matrix=matrix[np.ix_(selected_states,selected_actions)]
            row_labels=[state_labels[i] for i in selected_states]
            ylabel='Physical input state'
            title=('Absolute derivative' if args.mode=='state-frame' else 'Time RMS of derivatives')+f' | action step {first["action_step"]}'
        upper=args.vmax or max(float(np.percentile(matrix,args.percentile)),1e-12)
        fig,ax=plt.subplots(figsize=(14,8),layout='constrained')
        im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=upper,cmap='magma',interpolation='nearest')
        ax.set_xticks(range(len(selected_actions)),[action_labels[i] for i in selected_actions],rotation=75,ha='right',fontsize=8)
        ticks=np.unique(np.linspace(0,len(row_labels)-1,min(len(row_labels),25),dtype=int))
        ax.set_yticks(ticks,[row_labels[i] for i in ticks],fontsize=8)
        ax.set(xlabel='Output action component',ylabel=ylabel,title=title)
        fig.colorbar(im,ax=ax,label='Derivative magnitude (mixed physical units)')
        fig.savefig(args.output,dpi=150);plt.close(fig)
        np.save(args.output.with_suffix('.npy'),matrix)
    record={'mode':args.mode,'run_id':manifest['run_id'],'frames':[load_meta(p)['frame'] for p in paths],
        'action_step':first['action_step'],'actions':actions,'states':states,'state_index':args.state_index,
        'color_upper':upper,'percentile':args.percentile,'global_scale':True,
        'episode_complete':len(paths)==manifest['episode_frames'],'fps':manifest['fps'],
        'scale_method':'fixed vmax' if args.vmax else 'global sampled pixel percentile (camera) / matrix percentile (state)'}
    if args.mode=='camera':
        record.update(gamma=1.0,opacity=0.58,opacity_mode='magnitude',rows=2,cameras=labels,
            normalization=args.normalization,global_scale=args.normalization=='global',
            scale_method='per-frame pixel percentile shared across selected cameras' if args.normalization=='relative' else record['scale_method'])
        record['frame_color_upper']={str(row['frame']):row['color_upper'] for row in diagnostics}
    args.output.with_suffix('.json').write_text(json.dumps(record,indent=2))
    print('Saved',args.output,flush=True)


if __name__=='__main__':main()
