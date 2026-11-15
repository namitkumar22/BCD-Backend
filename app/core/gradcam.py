"""Grad-CAM visualisation for the VGG19 classifier.

Selçuk et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-Based Localization", ICCV 2017.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the final convolutional layer (conv5_4) of VGG19.

    The convolution itself is targeted rather than the following in-place ReLU:
    full backward hooks cannot be attached to the in-place activation without
    triggering autograd's view/in-place restriction.
    """
    features = model.features
    conv_indices = [
        index
        for index, layer in enumerate(features)
        if isinstance(layer, torch.nn.Conv2d)
    ]
    return features[conv_indices[-1]]


class GradCAM:
    """Compute a class activation map for a single image."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        # In-place ReLUs conflict with full backward hooks (the conv output is a
        # view created by the hook wrapper). Disabling in-place activation is
        # functionally identical and only costs a little memory.
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        layer = _last_conv_layer(model)
        layer.register_forward_hook(self._save_activations)
        layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        """Return a normalised heatmap in [0, 1] for ``image_tensor`` (1, 3, H, W)."""
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                logits = self.model(image_tensor)
                if class_index is None:
                    class_index = int(logits.argmax(dim=1).item())
                score = logits[0, class_index]
                score.backward()
        finally:
            self.model.train(was_training)

        if self.activations is None or self.gradients is None:  # pragma: no cover
            raise RuntimeError("Grad-CAM hooks did not capture activations")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        peak = cam.max()
        if peak > 0:
            cam /= peak
        return cam.astype(np.float32)


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet-coloured heatmap over an RGB uint8 image."""
    import matplotlib.cm as cm

    coloured = (cm.jet(heatmap)[..., :3] * 255).astype(np.uint8)
    blended = (1.0 - alpha) * image_rgb.astype(np.float32) + alpha * coloured.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)
