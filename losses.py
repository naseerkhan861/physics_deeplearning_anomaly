"""Loss functions for PGDN training.

* ``L_rec``  = L2 + (1 - SSIM) between reconstruction and clean image.
* ``L_seg``  = binary focal loss between segmentation logits and defect mask.
* ``L_phys`` = MSE between physics features (|grad|, |nabla^2|) of the
  reconstruction and of the clean image  +  an explainability-coupling term
  that penalizes predicted anomaly mass lying in regions where the physics
  residual is low (mass should concentrate where physics is violated).

Total: ``L = L_rec + L_seg + lambda_phys * L_phys``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics.operators import physics_features


# ---------------------------------------------------------------------------
# SSIM (single-channel, 11x11 Gaussian window)
# ---------------------------------------------------------------------------

def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).view(1, 1, size, size)


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM between two (B, C, H, W) images in [0, 1] (any C >= 1)."""
    c = x.shape[1]
    w = _gaussian_window(window_size, sigma).to(x).expand(c, 1, window_size, window_size)
    p = window_size // 2
    mu_x = F.conv2d(x, w, padding=p, groups=c)
    mu_y = F.conv2d(y, w, padding=p, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = F.conv2d(x * x, w, padding=p, groups=c) - mu_x2
    sy = F.conv2d(y * y, w, padding=p, groups=c) - mu_y2
    sxy = F.conv2d(x * y, w, padding=p, groups=c) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_xy + c1) * (2 * sxy + c2)) / ((mu_x2 + mu_y2 + c1) * (sx + sy + c2))
    return m.mean()


def _luminance(x: torch.Tensor) -> torch.Tensor:
    """Rec. 601 luma for (B,3,H,W); identity for single-channel input."""
    if x.shape[1] == 3:
        w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x * w).sum(dim=1, keepdim=True)
    return x


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss on logits; ``target`` in {0, 1}, same shape."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    return (w * (1 - pt) ** gamma * bce).mean()


# ---------------------------------------------------------------------------
# Combined PGDN loss
# ---------------------------------------------------------------------------

class PGDNLoss(nn.Module):
    def __init__(
        self,
        lambda_phys: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        ssim_weight: float = 1.0,
        l2_weight: float = 1.0,
        use_rec: bool = True,
        use_couple: bool = True,
    ):
        super().__init__()
        self.lambda_phys = float(lambda_phys)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ssim_weight = ssim_weight
        self.l2_weight = l2_weight
        self.use_rec = bool(use_rec)
        self.use_couple = bool(use_couple)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        clean: torch.Tensor,
        mask: torch.Tensor,
        lambda_phys: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        lam = self.lambda_phys if lambda_phys is None else float(lambda_phys)
        recon = outputs["recon"]
        seg_logits = outputs["seg_logits"]
        phys = outputs.get("phys")

        l_l2 = F.mse_loss(recon, clean)
        l_ssim = 1.0 - ssim(recon.clamp(0, 1), clean)
        l_rec = self.l2_weight * l_l2 + self.ssim_weight * l_ssim

        l_seg = focal_loss(seg_logits, mask, self.focal_alpha, self.focal_gamma)

        if lam > 0 and phys is not None:
            # (a) physics-feature matching: recon should obey the same
            #     diffusion-equilibrium statistics as the clean surface
            #     (physics operators act on luminance for RGB inputs)
            l_feat = F.mse_loss(physics_features(_luminance(recon)),
                                physics_features(_luminance(clean))) if self.use_rec \
                else torch.zeros((), device=clean.device)
            # (b) explainability coupling: fraction of predicted anomaly mass
            #     sitting in low-physics-residual areas (to be minimized)
            if self.use_couple:
                p = torch.sigmoid(seg_logits)
                r = phys.detach()
                l_couple = (p * (1.0 - r)).sum() / (p.sum() + 1e-8)
            else:
                l_couple = torch.zeros((), device=clean.device)
            l_phys = l_feat + l_couple
        else:
            l_feat = torch.zeros((), device=clean.device)
            l_couple = torch.zeros((), device=clean.device)
            l_phys = torch.zeros((), device=clean.device)

        total = l_rec + l_seg + lam * l_phys
        parts = {
            "loss": float(total.detach()),
            "l_rec": float(l_rec.detach()),
            "l_l2": float(l_l2.detach()),
            "l_ssim": float(l_ssim.detach()),
            "l_seg": float(l_seg.detach()),
            "l_phys": float(l_phys.detach()),
            "l_phys_feat": float(l_feat.detach()),
            "l_phys_couple": float(l_couple.detach()),
        }
        return total, parts
