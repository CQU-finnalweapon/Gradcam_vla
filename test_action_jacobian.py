"""Check spatial packing and full-output sensitivity aggregation independently."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from run_spirit_action_jacobian import jacobian_norm, pack_rgb, unpack_rgb, tile_images, stitch_tile_sums


class ActionJacobianTest(unittest.TestCase):
    def test_merge_squares_and_reject_missing_outputs(self):
        from run_spirit_action_jacobian_multigpu import merge_partials, partition_outputs
        self.assertEqual(partition_outputs(7,3),[(0,2),(2,4),(4,7)])
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            fields=dict(total_action_outputs=7,seed=0,episode=0,frame=150,
                        checkpoint_index_sha256='ckpt',input_sha256='input',task='task',
                        labels=['camera'],image_size=[2,2],wrist_image_size=[2,2],image_resize_mode='resize')
            for i,(start,end,value) in enumerate([(0,2,4.),(2,7,9.)]):
                path=Path(directory)/f'{i}.npz'; paths.append(path)
                np.savez(path,camera_squared=np.full((2,2),value),predicted_actions=np.zeros((1,7)))
                path.with_suffix('.json').write_text(json.dumps(dict(fields,action_range=[start,end])))
            maps,_,_=merge_partials(paths,7)
            np.testing.assert_allclose(maps['camera_model_input'],np.full((2,2),13**.5))
            with self.assertRaises(ValueError):
                merge_partials(paths[:1],7)
            with self.assertRaises(ValueError):
                merge_partials(paths[::-1],7)

    def test_output_partition_and_input_tiles_match_full_context(self):
        # Globally coupled nonlinear model: cropped forwards would be wrong.
        torch.manual_seed(7)
        images = [torch.randn(3,5,7), torch.randn(3,4,3)]
        weights = torch.randn(7, sum(x.numel() for x in images))
        def model(xs):
            x = torch.cat([a.flatten() for a in xs])
            return (weights @ x).sin() + x.mean().square()
        leaves = [x.clone().requires_grad_() for x in images]
        baseline = jacobian_norm(model(leaves), leaves)
        # Uneven output partitions, uneven tile edges, and input minibatches.
        for input_chunk in (0, 2):
            combined = [torch.zeros(x.shape[-2:]) for x in images]
            for start,end in ((0,2),(2,7)):
                tiles,loc,full = tile_images(images,(3,4))
                for got,expected in zip(full,images):
                    torch.testing.assert_close(got,expected)
                parts = jacobian_norm(model(full),tiles,chunk=2,output_start=start,output_end=end,
                                      input_chunk=input_chunk,squared=True)
                for accum,part in zip(combined,stitch_tile_sums(parts,loc,images)):
                    accum.add_(part)
            for expected,actual in zip(baseline,combined):
                torch.testing.assert_close(expected,actual.sqrt())

    def test_cpu_saved_tensors_preserve_gradient(self):
        x=torch.randn(3,4,5,requires_grad=True)
        with torch.autograd.graph.save_on_cpu():
            y=torch.stack([x.sin().sum(), x.square().sum()])
        got, = jacobian_norm(y,[x])
        expected=(x.cos().square()+4*x.square()).sum(0).sqrt()
        torch.testing.assert_close(got,expected)

    def test_matches_huggingface_image_processor(self):
        from transformers import Qwen2VLImageProcessorFast
        processor = Qwen2VLImageProcessorFast(
            patch_size=2, temporal_patch_size=2, merge_size=2,
            image_mean=[.5]*3, image_std=[.5]*3,
        )
        # Asymmetric image and individually distinguishable RGB/spatial values
        # detect swapped channels, misplaced patches and merge-order errors.
        rgb = np.arange(8*12*3, dtype=np.uint16).reshape(8,12,3).astype(np.uint8)
        reference = processor.preprocess(rgb, do_resize=False, return_tensors='pt')
        x = torch.tensor(rgb).permute(2,0,1).float()/255
        actual = pack_rgb(x, processor)
        torch.testing.assert_close(actual, reference['pixel_values'])
        restored = unpack_rgb(actual, reference['image_grid_thw'][0], processor)
        torch.testing.assert_close(restored, x)

    def test_temporal_duplicates_are_tied(self):
        processor = SimpleNamespace(patch_size=2, merge_size=2, temporal_patch_size=2,
                                    image_mean=[.5]*3, image_std=[.5]*3)
        x = torch.zeros(3,8,12, requires_grad=True)
        g, = torch.autograd.grad(pack_rgb(x, processor).sum(), x)
        # Two repeated temporal pixels; each carries derivative 1/std = 2.
        torch.testing.assert_close(g, torch.full_like(x,4.))

    def test_full_outputs_do_not_cancel(self):
        for chunk in (1, 2):
            x = torch.ones(3,2,2, requires_grad=True)
            # Gradients of opposite sign must be squared separately.
            outputs = torch.stack([2*x.sum(), -3*x.sum()])
            heat, = jacobian_norm(outputs, [x], chunk)
            torch.testing.assert_close(heat, torch.full((2,2), (3*(4+9))**.5))

    def test_multiple_cameras_against_explicit_jacobian(self):
        torch.manual_seed(3)
        weights = torch.randn(5, 18)
        reference = [(weights[:, :12].reshape(5,3,2,2)**2).sum((0,1)).sqrt(),
                     (weights[:, 12:].reshape(5,3,1,2)**2).sum((0,1)).sqrt()]
        for chunk in (1,2,3):
            images = [torch.randn(3,2,2,requires_grad=True), torch.randn(3,1,2,requires_grad=True)]
            outputs = weights @ torch.cat([im.flatten() for im in images])
            result = jacobian_norm(outputs, images, chunk)
            for got, expected in zip(result,reference):
                torch.testing.assert_close(got, expected)


if __name__ == '__main__':
    unittest.main()
