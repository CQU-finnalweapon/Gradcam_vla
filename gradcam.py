"""Spatial Grad-CAM for mozbrain PI0.5, through the full action sampler."""
from contextlib import contextmanager

import torch


def token_gradcam(activation, gradient, grid):
    """[B, patches, channels] -> signed and positive [B, H, W] maps."""
    if activation.shape != gradient.shape or activation.ndim != 3:
        raise ValueError("Expected matching [batch, patches, channels] tensors")
    if activation.shape[1] != grid[0] * grid[1]:
        raise ValueError("Patch count does not match the image grid")
    activation, gradient = activation.float(), gradient.float()
    weights = gradient.mean(dim=1, keepdim=True)
    signed = (activation * weights).sum(dim=-1).reshape(-1, *grid)
    return signed, signed.relu()


@contextmanager
def capture_features(policy, layer):
    """Freeze weights and start autograd at the target layer; restore on exit.

    PI0.5 calls its vision tower separately for each camera, in image_features
    order. Each hook output remains a separate tensor in the autograd graph.
    """
    flags = [(p, p.requires_grad) for p in policy.parameters()]
    modes = [(m, m.training) for m in policy.modules()]
    activations = []

    def hook(module, inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise ValueError("Choose a vision layer returning [B, patches, channels]")
        output = output.detach().requires_grad_(True)
        activations.append(output)
        return output

    handle = None
    try:
        policy.eval()
        policy.requires_grad_(False)
        handle = layer.register_forward_hook(hook)
        with torch.inference_mode(False), torch.enable_grad():
            yield activations
    finally:
        if handle is not None:
            handle.remove()
        for p, flag in flags:
            p.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


def sample_actions_with_grad(model, images, masks, tokens, token_masks, state, noise):
    """Same Euler/KV-cache sampler as PI05FlowMatching, without no_grad.

    No previous chunk / RTC guidance is supplied. Avoid the original in-place
    update of the time tensor, which invalidates saved tensors in autograd.
    Both the prefix cache and all denoising steps stay differentiable.
    """
    from lerobot.common.policies.pi05.modeling_pi05 import make_att_2d_masks

    prefix, pad, att = model.embed_prefix(images, masks, tokens, token_masks)
    attention_mask = model._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=attention_mask,
        position_ids=torch.cumsum(pad, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix, None],
        use_cache=True,
    )
    steps = model.config.num_steps
    if steps <= 0:
        raise ValueError("num_steps must be positive")
    dt = torch.tensor(-1.0 / steps, device=noise.device, dtype=torch.float32)
    time = torch.tensor(1.0, device=noise.device, dtype=torch.float32)
    actions = noise
    for _ in range(steps):
        velocity = model.denoise_step(state, pad, cache, actions, time.expand(state.shape[0]))
        actions = actions + dt * velocity
        time = time + dt
    return actions


def explain(policy, prepared, noise, action_step, action_dim, sign=1, layer_index=-1):
    """Explain a final normalized action component (not a training loss)."""
    images, masks, tokens, token_masks, state = prepared
    if any(img.shape[0] != 1 for img in images):
        raise ValueError("This visualizer expects one observation at a time")
    if not 0 <= action_step < policy.config.n_action_steps:
        raise ValueError("action_step is outside the action chunk")
    if not 0 <= action_dim < policy.config.action_feature.shape[0]:
        raise ValueError("action_dim is outside the real (unpadded) action space")
    if sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    vision = policy.model.paligemma_with_expert.paligemma.model.vision_tower
    layer = vision.vision_model.encoder.layers[layer_index].layer_norm1
    patch_size = vision.config.patch_size
    with capture_features(policy, layer) as activations:
        actions = sample_actions_with_grad(policy.model, *prepared, noise)
        if not torch.isfinite(actions).all():
            raise RuntimeError("The action sampler produced non-finite values")
        if len(activations) != len(images):
            raise RuntimeError("Vision hook calls do not match the camera count")
        target = sign * actions[0, action_step, action_dim]
        gradients = torch.autograd.grad(target, activations)
        maps, diagnostics = [], []
        for img, activation, gradient in zip(images, activations, gradients, strict=True):
            if not torch.isfinite(gradient).all() or gradient.abs().max() == 0:
                raise RuntimeError("Camera has non-finite or zero gradients; check the graph")
            grid = (img.shape[-2] // patch_size, img.shape[-1] // patch_size)
            signed, positive = token_gradcam(activation, gradient, grid)
            if not torch.isfinite(signed).all():
                raise RuntimeError("Non-finite Grad-CAM")
            maps.append((signed[0].detach().cpu().numpy(), positive[0].detach().cpu().numpy()))
            diagnostics.append({
                "gradient_abs_max": gradient.abs().max().item(),
                "gradient_abs_mean": gradient.float().abs().mean().item(),
                "positive_fraction": (positive > 0).float().mean().item(),
                "positive_max": positive.max().item(),
                "signed_min": signed.min().item(),
                "signed_max": signed.max().item(),
            })
        return actions.detach(), maps, diagnostics
