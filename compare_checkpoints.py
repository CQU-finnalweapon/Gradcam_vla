"""Render aligned positive Grad-CAMs from two runs on the same observation."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--labels', nargs=2, default=['5000 steps', '10000 steps'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    metadata = [json.loads((p / 'metadata.json').read_text()) for p in (args.before, args.after)]
    for key in ['dataset', 'episode', 'frame', 'task', 'target', 'seed', 'layer',
                'num_denoising_steps', 'camera_order', 'normalized_state']:
        if metadata[0][key] != metadata[1][key]:
            raise ValueError(f'Comparison settings differ: {key}')
    cameras = metadata[0]['camera_order']
    for camera in cameras:
        for kind in ['original', 'model_input']:
            before = np.asarray(Image.open(args.before / f'{camera}_{kind}.png'))
            after = np.asarray(Image.open(args.after / f'{camera}_{kind}.png'))
            if not np.array_equal(before, after):
                raise ValueError(f'Camera inputs differ: {camera} {kind}')
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 21)
    small = ImageFont.truetype(font_path, 17)
    canvas = Image.new('RGB', (1440, 1255), 'white')
    draw = ImageDraw.Draw(canvas)
    m = metadata[0]
    draw.text((16, 12), f'PI0.5 | {args.labels[0]} vs {args.labels[1]} | positive Grad-CAM', fill='black', font=font)
    target = m['target']
    draw.text((16, 44), f"Episode {m['episode']}, frame {m['frame']} | {target['sign']:+d} * action[{target['action_step']}, {target['action_dim']}] | seed {m['seed']}", fill='black', font=small)
    draw.text((16, 69), 'SigLIP last block / layer_norm1 | Each map scaled independently; color is not comparable magnitude.', fill='black', font=small)
    for row, (directory, kind, label) in enumerate([
        (args.before, 'original', 'Original camera frames'),
        (args.before, 'gradcam', args.labels[0]),
        (args.after, 'gradcam', args.labels[1]),
    ]):
        y = 110 + row * 380
        draw.text((16, y), label, fill='black', font=font)
        for col, camera in enumerate(cameras):
            draw.text((col * 480 + 16, y + 32), camera, fill='black', font=small)
            panel = Image.open(directory / f'{camera}_{kind}.png').convert('RGB')
            panel.thumbnail((460, 310))
            canvas.paste(panel, (col * 480 + (480 - panel.width)//2, y + 60))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(args.out)


if __name__ == '__main__':
    main()
