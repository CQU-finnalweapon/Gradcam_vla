import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import torch
from episode_jacobian import raw_jacobian, atomic_npz, valid_frame, differentiable_normalize, tile_images, EpisodeReader
from visualize_episode_jacobian import image_norm,state_matrix

class EpisodeTests(unittest.TestCase):
    def test_arbitrary_dataset_episode_and_video_alignment(self):
        import cv2
        import pyarrow as pa
        import pyarrow.parquet as pq
        mapping={'observation.images.cam_high':'cam_high',
                 'observation.images.cam_left_wrist':'left_wrist_high',
                 'observation.images.cam_right_wrist':'right_wrist_high'}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'meta').mkdir()
            cameras=['cam_high','cam_left_wrist','cam_right_wrist']
            info=dict(fps=10,chunks_size=2,data_path='data/{episode_chunk}/ep{episode_index}.parquet',
                      video_path='video/{video_key}/ep{episode_index}.avi',
                      features={k:{'dtype':'video'} for k in cameras})
            (root/'meta/info.json').write_text(json.dumps(info))
            (root/'meta/tasks.jsonl').write_text(json.dumps(dict(task_index=3,task='test task'))+'\n\n')
            (root/'data/2').mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([dict(frame_index=i,episode_index=5,task_index=3,
                              timestamp=i/10,**{'observation.state':[1.,2.]}) for i in range(3)]),root/'data/2/ep5.parquet')
            for cam in cameras:
                path=root/f'video/{cam}/ep5.avi';path.parent.mkdir(parents=True)
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,32))
                self.assertTrue(writer.isOpened())
                for i in range(3):writer.write(np.full((32,32,3),30+i*50,np.uint8))
                writer.release()
            reader=EpisodeReader(root,5,mapping)
            try:
                for frame in (1,2,0):
                    raw,images=reader.read(frame)
                    self.assertEqual(raw['task'],'test task')
                    for image in images.values():self.assertLess(abs(float(image.mean())-(30+frame*50)),3)
            finally:reader.close()

    def test_signed_physical_state_and_tiled_pixels(self):
        for chunk in (1,2):
            for input_chunk in (0,2):
                images=[torch.randn(3,3,5),torch.randn(3,2,2)]
                tiles,locations,full=tile_images(images,(2,3))
                state=torch.tensor([2.,3.,4.],requires_grad=True)
                normalized=state/torch.tensor([2.,4.,5.])
                masked=normalized*torch.tensor([1.,0.,1.])
                total=sum(im.sum() for im in full)
                outputs=torch.stack([2*total+masked.sum(),-3*total-4*masked.sum()])
                jacs,sj=raw_jacobian(outputs,tiles,state,locations,images,chunk,input_chunk)
                for jac in jacs:
                    np.testing.assert_allclose(jac[0],2)
                    np.testing.assert_allclose(jac[1],-3)
                    np.testing.assert_allclose(image_norm(jac),np.sqrt(39))
                np.testing.assert_allclose(sj,[[.5,-2],[0,0],[.2,-.8]],rtol=1e-6)
    def test_no_grad_normalizer_unwrapped(self):
        class Norm:
            @torch.no_grad()
            def forward(self,batch):return {'state':batch['state']/2}
        x=torch.ones(3,requires_grad=True)
        y=differentiable_normalize(Norm(),{'state':x})['state']
        torch.testing.assert_close(torch.autograd.grad(y.sum(),x)[0],torch.full_like(x,.5))
    def test_atomic_resume_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'frame_000001.npz'
            meta=dict(run_id='id',frame=1,schema=1,labels=['cam_high'])
            arrays=dict(metadata=np.asarray(json.dumps(meta)),jac_state=np.ones((2,2),np.float32),jac_image_cam_high=np.ones((2,3,2,2),np.float32))
            atomic_npz(p,arrays)
            self.assertTrue(valid_frame(p,'id',1,True))
            self.assertFalse(valid_frame(p,'other',1,True))
            self.assertFalse(valid_frame(p,'id',2,True))
            p.write_bytes(b'bad')
            self.assertFalse(valid_frame(p,'id',1,True))
    def test_state_rms_not_absolute_mean(self):
        a=np.asarray([[[2.,-3.]], [[-2.,3.]]])
        np.testing.assert_allclose(state_matrix(a),[[2.,3.]])
        np.testing.assert_allclose(state_matrix(a,'frame'),[[2.,3.]])
if __name__=='__main__':unittest.main()
