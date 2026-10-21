"""Physics-Guided Defect Network (PGDN).

DRAEM-style two-head architecture:

* **reconstruction head** (UNet): augmented image -> clean reconstruction.
* **segmentation head** (UNet): ``concat[augmented, reconstruction,
  physics_residual_map(augmented)]`` -> pixel-level anomaly logits.
  With ``use_physics_arch=False`` the physics channel is dropped
  (concat[augmented, reconstruction]); the same code then serves as the
  DRAEM-lite baseline.

The physics residual is computed under ``torch.no_grad`` so it stays a fixed
prior (an expectation the network reads), never a learned shortcut.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from physics.operators import (PhysicsPrior, PhysicsPriorColor, physics_residual_map,
                               variant_map)

from .unet import UNet, count_parameters


def _luminance(x: torch.Tensor) -> torch.Tensor:
    """Rec. 601 luma for (B,3,H,W); identity for single-channel input."""
    if x.shape[1] == 3:
        w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x * w).sum(dim=1, keepdim=True)
    return x


class PGDN(nn.Module):
    def __init__(
        self,
        base_channels: int = 16,
        use_physics_arch: bool = True,
        lambda_phys: float = 1.0,
        prior: Optional[PhysicsPrior] = None,
        in_channels: int = 1,
        phys_use_rec: bool = True,
        phys_use_couple: bool = True,
        channel_variant: str = "full",
    ):
        super().__init__()
        self.use_physics_arch = bool(use_physics_arch)
        self.lambda_phys = float(lambda_phys)
        self.in_channels = int(in_channels)
        self.phys_use_rec = bool(phys_use_rec)
        self.phys_use_couple = bool(phys_use_couple)
        self.channel_variant = channel_variant
        self.recon_unet = UNet(in_channels=self.in_channels, out_channels=self.in_channels,
                               base_channels=base_channels)
        seg_in = 2 * self.in_channels + (1 if self.use_physics_arch else 0)
        self.seg_unet = UNet(in_channels=seg_in, out_channels=1, base_channels=base_channels)
        self._prior = prior

    # -- physics prior ------------------------------------------------------

    @property
    def prior(self) -> Optional[PhysicsPrior]:
        return self._prior

    def set_prior(self, prior: PhysicsPrior) -> None:
        self._prior = prior

    # -- forward ------------------------------------------------------------

    def physics_map(self, x: torch.Tensor) -> torch.Tensor:
        """Detached per-pixel physics-expectation deviation map.

        Colour-aware priors consume the RGB input directly; luminance priors
        consume the luma channel. ``channel_variant`` selects the control
        descriptors (grad / lap / var / shuffle) or the full map.
        """
        with torch.no_grad():
            if self.channel_variant != "full":
                return variant_map(_luminance(x) if x.shape[1] == 3 else x,
                                   self._prior, self.channel_variant)
            if getattr(self._prior, "grad_mean_c", None) and x.shape[1] == 3:
                return physics_residual_map(x, self._prior)
            return physics_residual_map(_luminance(x), self._prior)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon = self.recon_unet(x)
        # z is computed whenever a prior exists -- the segmentation head
        # consumes it only when the channel is on, but the loss components
        # (feature matching / coupling) may use it independently.
        phys = self.physics_map(x) if self._prior is not None else None
        if phys is not None and self.use_physics_arch:
            seg_in = torch.cat([x, recon, phys], dim=1)
        else:
            seg_in = torch.cat([x, recon], dim=1)
        seg_logits = self.seg_unet(seg_in)
        return {"recon": recon, "seg_logits": seg_logits, "phys": phys}

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Probability anomaly map for inference, shape (B, 1, H, W)."""
        return torch.sigmoid(self.forward(x)["seg_logits"])

    # -- config / checkpoint helpers ----------------------------------------

    def get_config(self) -> Dict:
        return {
            "base_channels": self.recon_unet.enc1.block[0].out_channels,
            "use_physics_arch": self.use_physics_arch,
            "lambda_phys": self.lambda_phys,
            "in_channels": self.in_channels,
            "phys_use_rec": self.phys_use_rec,
            "phys_use_couple": self.phys_use_couple,
            "channel_variant": self.channel_variant,
            "prior": self._prior.to_dict() if self._prior is not None else None,
        }

    @classmethod
    def from_config(cls, cfg: Dict) -> "PGDN":
        prior = None
        if cfg.get("prior"):
            pdict = cfg["prior"]
            prior = (PhysicsPriorColor.from_dict(pdict) if pdict.get("color")
                     else PhysicsPrior.from_dict(pdict))
        return cls(
            base_channels=cfg.get("base_channels", 16),
            use_physics_arch=cfg.get("use_physics_arch", True),
            lambda_phys=cfg.get("lambda_phys", 1.0),
            prior=prior,
            in_channels=cfg.get("in_channels", 1),
            phys_use_rec=cfg.get("phys_use_rec", True),
            phys_use_couple=cfg.get("phys_use_couple", True),
            channel_variant=cfg.get("channel_variant", "full"),
        )

    def num_parameters(self) -> int:
        return count_parameters(self)
