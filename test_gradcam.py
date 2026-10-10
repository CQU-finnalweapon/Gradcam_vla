import numpy as np
import pytest
import torch
from torch import nn

from gradcam import capture_features, token_gradcam
from run import overlay, project_to_original


def test_sharded_checkpoint_load_and_integrity(tmp_path):
    import json
    from safetensors.torch import save_file
    from run import load_sharded_safetensors

    save_file({'vision': torch.arange(6).reshape(2, 3)}, tmp_path / 'a.safetensors')
    save_file({'action': torch.ones(4)}, tmp_path / 'b.safetensors')
    index = tmp_path / 'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map': {'vision': 'a.safetensors', 'action': 'b.safetensors'}}))
    state = load_sharded_safetensors(index)
    torch.testing.assert_close(state['vision'], torch.arange(6).reshape(2, 3))
    torch.testing.assert_close(state['action'], torch.ones(4))
    save_file({'wrong_action': torch.ones(4)}, tmp_path / 'b.safetensors')
    with pytest.raises(RuntimeError, match='Shard/index key mismatch'):
        load_sharded_safetensors(index)
    (tmp_path / 'b.safetensors').unlink()
    with pytest.raises(FileNotFoundError):
        load_sharded_safetensors(index)


def test_cam_matches_linear_gap_classifier():
    # Known CAM equivalence: GAP + linear classifier has gradient w / N.
    activation = torch.tensor([[[1., 2.], [2., 0.], [0., 3.], [4., 1.]]], requires_grad=True)
    weights = torch.tensor([2., -1.])
    target = (activation.mean(dim=1) * weights).sum()
    gradient, = torch.autograd.grad(target, activation)
    signed, positive = token_gradcam(activation, gradient, (2, 2))
    expected = torch.tensor([[[0., 1.], [-0.75, 1.75]]])
    torch.testing.assert_close(signed, expected)
    torch.testing.assert_close(positive, expected.relu())
    with pytest.raises(ValueError):
        token_gradcam(activation, gradient, (3, 3))


def test_shared_camera_layer_keeps_distinct_gradients_and_restores_state():
    policy = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Dropout(0.5))
    policy.train()
    policy[1].eval()
    with torch.no_grad():
        policy[0].weight.copy_(torch.eye(2))
    with capture_features(policy, policy[0]) as features:
        outputs = [policy(torch.ones(1, 4, 2) * value) for value in [1, 2, 3]]
        target = sum(weight * output.sum() for weight, output in zip([1, 2, 3], outputs))
        gradients = torch.autograd.grad(target, features)
        for weight, gradient in zip([1, 2, 3], gradients):
            torch.testing.assert_close(gradient, torch.full_like(gradient, weight))
    assert policy.training and not policy[1].training
    assert policy[0].weight.requires_grad and policy[0].weight.grad is None
    assert not policy[0]._forward_hooks
    with pytest.raises(RuntimeError):
        with capture_features(policy, policy[0]):
            raise RuntimeError('test cleanup')
    assert not policy[0]._forward_hooks and policy[0].weight.requires_grad


def test_original_projection_removes_top_and_left_padding():
    grid = np.zeros((4, 4), dtype=np.float32)
    grid[:2] = 100  # padding: must not leak into the camera image
    grid[2:] = 1
    projected = project_to_original(grid, (2, 4, 3), (4, 4), 'resize_with_pad')
    np.testing.assert_array_equal(projected, np.ones((2, 4)))
    projected = project_to_original(grid.T, (4, 2, 3), (4, 4), 'resize_with_pad')
    np.testing.assert_array_equal(projected, np.ones((4, 2)))


def test_crop_projection_and_blank_overlay():
    result = project_to_original(np.ones((2, 2), np.float32), (2, 4, 3), (2, 2), 'resize_with_crop')
    np.testing.assert_array_equal(result, np.array([[0, 1, 1, 0], [0, 1, 1, 0]]))
    rgb = np.full((3, 5, 3), 129, np.uint8)
    np.testing.assert_array_equal(overlay(rgb, np.zeros((3, 5))), rgb)
