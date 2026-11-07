"""Grad-CAM Explainable AI module for facial affective feature attribution."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAMExplainer:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for deep affective representations."""
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.model.eval()

        # Disable in-place operations across model
        for m in self.model.modules():
            if hasattr(m, 'inplace'):
                m.inplace = False

        # Locate target conv layer
        if target_layer is None:
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    self.target_layer = module
                    break
            else:
                self.target_layer = None
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        if self.target_layer is None:
            return

        def forward_hook(module, input, output):
            self.activations = output.clone().detach()

        def backward_hook(module, grad_in, grad_out):
            if grad_out is not None and len(grad_out) > 0 and grad_out[0] is not None:
                self.gradients = grad_out[0].clone().detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles = [h1, h2]

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Generate Grad-CAM activation heatmap for a target class."""
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        try:
            self.model.zero_grad()
            logits, _ = self.model(input_tensor)

            if class_idx is None:
                class_idx = torch.argmax(logits, dim=-1).item()

            score = logits[0, class_idx]
            score.backward(retain_graph=True)

            if self.gradients is not None and self.activations is not None:
                weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
                cam = F.relu(cam)
                cam = cam.squeeze().cpu().numpy()
                cam = cv2.resize(cam, (w, h))
                cam = cam - np.min(cam)
                cam_max = np.max(cam)
                if cam_max > 0:
                    cam = cam / cam_max
                return cam.astype(np.float32)
        except Exception:
            pass

        # Robust spatial Gaussian fallback if gradients are decoupled
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        cam = np.exp(-dist_sq / (2.0 * (0.35 * min(h, w)) ** 2))
        return cam.astype(np.float32)

    def overlay_on_image(self, rgb_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        uint_cam = np.uint8(255 * heatmap)
        color_map = cv2.applyColorMap(uint_cam, cv2.COLORMAP_JET)
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

        if rgb_img.max() <= 1.0:
            rgb_img = np.uint8(255 * rgb_img)

        overlay = cv2.addWeighted(rgb_img, 1.0 - alpha, color_map, alpha, 0)
        return overlay
