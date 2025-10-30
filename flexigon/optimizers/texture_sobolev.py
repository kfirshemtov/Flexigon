import torch
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import transforms


# --- Sobolev Preconditioner ---
class SobolevPreconditioner:
    def __init__(self, mode='image', lam=0.02, kernel_size=11, sigma=2.0, device='cuda'):
        self.mode = mode
        self.lam = lam
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = device
        self.kernel = self._create_gaussian_kernel().to(device)

    def _create_gaussian_kernel(self):
        # Create 2D Gaussian kernel for convolution
        k = self.kernel_size
        sigma = self.sigma
        ax = torch.arange(-k // 2 + 1., k // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).repeat(3, 1, 1, 1)  # assume 3 channels
        return kernel.float()

    def apply(self, param):
        # Apply Sobolev preconditioning
        if param.grad is None:
            return
        g = param.grad
        if self.mode == 'image':
            if g.ndim == 4 and g.shape[-1] == 3:
                g2 = g.permute(0, 3, 1, 2).contiguous()
            elif g.ndim == 4 and g.shape[1] == 3:
                g2 = g
            else:
                raise RuntimeError(f"Unsupported grad shape for image mode: {g.shape}")
            g_sob = self._apply_image_sobolev(g2)
            param.grad.data = g_sob.permute(0, 2, 3, 1).contiguous()

    def _apply_image_sobolev(self, grad):
        B, C, H, W = grad.shape
        pad = self.kernel_size // 2
        grad_pad = F.pad(grad, (pad, pad, pad, pad), mode='reflect')
        g_smooth = F.conv2d(grad_pad, self.kernel, groups=C)
        return g_smooth


# --- SobolevAdam optimizer wrapper ---
class SobolevAdam(optim.Adam):
    def __init__(self, params, preconditioner=None, **kwargs):
        super().__init__(params, **kwargs)
        self.pre = preconditioner

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        super().step()
        if self.pre is not None:
            for group in self.param_groups:
                for p in group['params']:
                    self.pre.apply(p)
        return loss