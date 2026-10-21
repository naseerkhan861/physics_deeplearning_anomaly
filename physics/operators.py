"""Differentiable physics operators for the Physics-Guided Defect Network (PGDN).

Manufactured defect-free surfaces are assumed to be close to a diffusion
equilibrium: their intensity field has a bounded Laplacian and smooth gradient
statistics.  All operators are implemented as fixed-kernel ``conv2d`` calls so
they are differentiable w.r.t. the input image and run on pure CPU.

Conventions
-----------
* Images are float tensors ``(B, 1, H, W)`` with intensities in ``[0, 1]``.
* Kernels match OpenCV ``Sobel(ksize=3)`` / ``Laplacian(ksize=3)`` exactly so
  that statistics estimated with OpenCV (e.g. ``physics_stats.json`` produced
  by ``datasets/make_construct.py`` on uint8 images) are reproduced by these
  ops up to the linear 1/255 intensity scaling.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Fixed convolution kernels (buffers, never learned)
# ---------------------------------------------------------------------------

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
)  # matches cv2.Sobel(dx=1, dy=0, ksize=3)
_SOBEL_Y = _SOBEL_X.t().contiguous()
_LAPLACIAN = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
)  # matches cv2.Laplacian(ksize=3)

_EPS = 1e-8


def _kernel(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=ref.dtype, device=ref.device).view(1, 1, 3, 3)


def _conv(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # BUGFIX: zero-padding created an artificial high-gradient ring at the
    # image border (~3% of pixels with ~50x typical |grad|), which inflated
    # prior statistics by >5x and made them inconsistent with the OpenCV
    # computed physics_stats.json (cv2 uses BORDER_REFLECT_101).  Explicit
    # reflect padding matches OpenCV exactly (verified: identical output to
    # cv2.Sobel/cv2.Laplacian up to the 1/255 intensity scale).
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k)


# ---------------------------------------------------------------------------
# Basic field operators
# ---------------------------------------------------------------------------

def gradient_field(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude |grad x|, shape (B, 1, H, W)."""
    gx = _conv(x, _kernel(_SOBEL_X, x))
    gy = _conv(x, _kernel(_SOBEL_Y, x))
    return torch.sqrt(gx * gx + gy * gy + _EPS)


def gradient_xy(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient components, shape (B, 2, H, W)."""
    gx = _conv(x, _kernel(_SOBEL_X, x))
    gy = _conv(x, _kernel(_SOBEL_Y, x))
    return torch.cat([gx, gy], dim=1)


def laplacian(x: torch.Tensor) -> torch.Tensor:
    """Laplacian (nabla^2 x), shape (B, 1, H, W)."""
    return _conv(x, _kernel(_LAPLACIAN, x))


def diffusion_residual(x: torch.Tensor) -> torch.Tensor:
    """Diffusion-equilibrium residual R(x) = |nabla^2 x|.

    Defect-free manufactured surfaces satisfy near-equilibrium, i.e. R(x) is
    small and its statistics match the ``laplacian_*`` entries of
    ``physics_stats.json``.
    """
    return torch.abs(laplacian(x))


def curvature_energy(x: torch.Tensor, reduction: str = "map"):
    """Curvature energy map (nabla^2 x)^2 (optionally reduced to a scalar)."""
    e = laplacian(x) ** 2
    if reduction == "map":
        return e
    if reduction == "mean":
        return e.mean()
    if reduction == "sum":
        return e.sum()
    raise ValueError(f"unknown reduction {reduction!r}")


def roughness_energy(x: torch.Tensor, reduction: str = "map"):
    """Surface roughness energy map |grad x|^2 (optionally reduced)."""
    e = gradient_field(x) ** 2
    if reduction == "map":
        return e
    if reduction == "mean":
        return e.mean()
    if reduction == "sum":
        return e.sum()
    raise ValueError(f"unknown reduction {reduction!r}")


def physics_features(x: torch.Tensor) -> torch.Tensor:
    """Two-channel physics feature map [|grad x|, |nabla^2 x|].

    Used both for the physics loss (feature matching between reconstruction
    and clean image) and by :class:`PhysicsPrior`.
    """
    return torch.cat([gradient_field(x), diffusion_residual(x)], dim=1)


def _squash01(z: torch.Tensor) -> torch.Tensor:
    """Monotonic squash of a non-negative deviation map to [0, 1)."""
    z = torch.clamp(z, min=0.0)
    return z / (1.0 + z)


def physics_residual_map(
    x: torch.Tensor, prior: Optional["PhysicsPrior"] = None
) -> torch.Tensor:
    """Normalized per-pixel deviation from the expected surface physics.

    With a :class:`PhysicsPrior`, deviation is a z-score against the learned
    defect-free statistics; without one, the map is self-normalized per batch
    (robust fallback, mainly for smoke tests).  Output is in ``[0, 1)`` with
    the same spatial shape as ``x``.
    """
    if prior is not None:
        return prior.deviation_map(x)
    g = gradient_field(x)
    r = diffusion_residual(x)
    zg = (g - g.mean()) / (g.std() + _EPS)
    zr = (r - r.mean()) / (r.std() + _EPS)
    return _squash01(0.5 * (zg + zr))


# ---------------------------------------------------------------------------
# PhysicsPrior: defect-free surface statistics
# ---------------------------------------------------------------------------

class PhysicsPrior:
    """Holds mu/sigma of |grad| and |nabla^2| of defect-free surfaces.

    Sources, in order of preference:
      1. ``from_json``: CONSTRUCT ``physics_stats.json`` (values are on the
         uint8 scale; they are divided by 255 because these ops work on
         ``[0, 1]`` images and all kernels are linear).
      2. ``estimate_from_folder``: computed with the torch operators above
         from any ``train/good`` folder (e.g. MVTec screw, which ships no
         physics statistics).
      3. ``to_dict`` / ``from_dict``: (de)serialization so a fitted prior can
         be stored inside a training checkpoint.
    """

    def __init__(
        self,
        grad_abs_mean: float,
        grad_abs_std: float,
        laplacian_mean: float,
        laplacian_std: float,
        source: str = "manual",
    ):
        self.grad_abs_mean = float(grad_abs_mean)
        self.grad_abs_std = float(grad_abs_std)
        self.laplacian_mean = float(laplacian_mean)
        self.laplacian_std = float(laplacian_std)
        self.source = source

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_json(cls, path: str, class_name: Optional[str] = None) -> "PhysicsPrior":
        with open(path) as f:
            data = json.load(f)
        if "classes" in data:
            if class_name is None:
                class_name = sorted(data["classes"].keys())[0]
            stats = data["classes"][class_name]
        else:
            stats = data
        s = 1.0 / 255.0  # stats were computed on uint8 images
        return cls(
            grad_abs_mean=stats["grad_abs_mean"] * s,
            grad_abs_std=stats["grad_abs_std"] * s,
            laplacian_mean=stats["laplacian_mean"] * s,
            laplacian_std=stats["laplacian_std"] * s,
            source=f"json:{os.path.basename(path)}:{class_name}",
        )

    @classmethod
    def estimate_from_folder(
        cls,
        folder: str,
        image_size: int = 128,
        max_images: int = 64,
        exts=(".png", ".jpg", ".jpeg", ".bmp"),
    ) -> "PhysicsPrior":
        import cv2  # local import: physics module stays cv2-free otherwise

        files = sorted(
            f for f in os.listdir(folder) if f.lower().endswith(exts)
        )[:max_images]
        if not files:
            raise FileNotFoundError(f"no images found in {folder}")
        grads, laps = [], []
        for f in files:
            im = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            im = cv2.resize(im, (image_size, image_size), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(im.astype("float32") / 255.0)[None, None]
            grads.append(gradient_field(x).numpy().ravel())
            laps.append(laplacian(x).numpy().ravel())
        import numpy as np

        g = np.concatenate(grads)
        l = np.concatenate(laps)
        return cls(
            grad_abs_mean=float(g.mean()),
            grad_abs_std=float(g.std() + _EPS),
            laplacian_mean=float(l.mean()),
            laplacian_std=float(l.std() + _EPS),
            source=f"estimated:{folder}",
        )

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, float]:
        return {
            "grad_abs_mean": self.grad_abs_mean,
            "grad_abs_std": self.grad_abs_std,
            "laplacian_mean": self.laplacian_mean,
            "laplacian_std": self.laplacian_std,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "PhysicsPrior":
        return cls(
            d["grad_abs_mean"],
            d["grad_abs_std"],
            d["laplacian_mean"],
            d["laplacian_std"],
            source=d.get("source", "dict"),
        )

    # -- deviation map -----------------------------------------------------

    @torch.no_grad()
    def deviation_map(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel combined z-score deviation, squashed to [0, 1).

        ``no_grad`` keeps the prior a fixed expectation, not a learned
        shortcut; callers may still safely feed the result into a network.
        """
        g = gradient_field(x)
        l = laplacian(x)
        zg = (g - self.grad_abs_mean) / (self.grad_abs_std + _EPS)
        zl = torch.abs((l - self.laplacian_mean) / (self.laplacian_std + _EPS))
        z = 0.5 * (zg + zl)
        return _squash01(z)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PhysicsPrior(grad={self.grad_abs_mean:.5f}+-{self.grad_abs_std:.5f}, "
            f"lap={self.laplacian_mean:.6f}+-{self.laplacian_std:.5f}, "
            f"source={self.source!r})"
        )


# ---------------------------------------------------------------------------
# Derivative-prior channel variants (review controls): the full z-map vs.
# simpler derivative descriptors and distribution-preserving shuffles.
# ---------------------------------------------------------------------------

def variant_map(x: torch.Tensor, prior: Optional[PhysicsPrior],
                variant: str = "full") -> torch.Tensor:
    """Per-pixel prior-channel variants used as controls.

    full    -- the standard combined |grad| + |lap| z-score map (Eq. 1)
    grad    -- gradient-magnitude z-score only
    lap     -- |Laplacian| z-score only
    var     -- local variance (8x8 patch std), self-normalised per image
    shuffle -- the full map with 16x16 blocks spatially permuted
               (distribution preserved, spatial specificity destroyed)
    """
    g = gradient_field(x)
    l = torch.abs(laplacian(x))
    eps = _EPS
    if variant == "full":
        return physics_residual_map(x, prior)
    if variant == "grad":
        if prior is None:
            return _squash01((g - g.mean()) / (g.std() + eps))
        z = (g - prior.grad_abs_mean) / (prior.grad_abs_std + eps)
        return _squash01(z)
    if variant == "lap":
        if prior is None:
            return _squash01((l - l.mean()) / (l.std() + eps))
        z = (l - abs(prior.laplacian_mean)) / (prior.laplacian_std + eps)
        return _squash01(z)
    if variant == "var":
        k = 8
        mean = F.avg_pool2d(x, k, stride=k)
        var = F.avg_pool2d((x - F.interpolate(mean, size=x.shape[-2:], mode="nearest")) ** 2,
                           k, stride=k)
        var = F.interpolate(var, size=x.shape[-2:], mode="nearest")
        return _squash01((var - var.mean()) / (var.std() + eps))
    if variant == "shuffle":
        z = physics_residual_map(x, prior)
        b = 16
        hb, wb = x.shape[-2] // b, x.shape[-1] // b
        blocks = z[:, :, :hb * b, :wb * b].reshape(z.shape[0], z.shape[1], hb, b, wb, b)
        perm = torch.randperm(hb * wb, device=z.device)
        blocks = blocks.reshape(z.shape[0], z.shape[1], hb * wb, b, b)[:, :, perm]
        out = blocks.reshape(z.shape[0], z.shape[1], hb, b, wb, b)
        out = out.reshape(z.shape[0], z.shape[1], hb * b, wb * b)
        # paste back (edges unchanged if not divisible)
        z = z.clone()
        z[:, :, :hb * b, :wb * b] = out
        return z
    raise ValueError(f"unknown channel variant {variant!r}")


class PhysicsPriorColor(PhysicsPrior):
    """Colour-aware equilibrium prior: per-channel |grad|/|lap| statistics.

    Extends the luminance prior to RGB: each colour channel is modelled as
    its own near-equilibrium field, and the deviation map is the channel-
    average z-score (squashed).  The luminance scalars are kept (channel-0
    means) so the base-class interface still works for grayscale inputs.
    """

    def __init__(self, grad_mean_c, grad_std_c, lap_mean_c, lap_std_c,
                 source: str = "manual"):
        grad_mean_c = [float(v) for v in grad_mean_c]
        super().__init__(grad_abs_mean=grad_mean_c[0],
                         grad_abs_std=float(grad_std_c[0]),
                         laplacian_mean=float(lap_mean_c[0]),
                         laplacian_std=float(lap_std_c[0]), source=source)
        self.grad_mean_c = grad_mean_c
        self.grad_std_c = [float(v) for v in grad_std_c]
        self.lap_mean_c = [float(v) for v in lap_mean_c]
        self.lap_std_c = [float(v) for v in lap_std_c]

    @classmethod
    def estimate_from_folder_rgb(cls, folder, image_size=128, max_images=64,
                                 exts=(".png", ".jpg", ".jpeg", ".bmp")):
        import cv2  # local import keeps the physics module cv2-free otherwise

        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))[:max_images]
        if not files:
            raise FileNotFoundError(f"no images found in {folder}")
        grads = [[], [], []]
        laps = [[], [], []]
        for f in files:
            im = cv2.imread(os.path.join(folder, f), cv2.IMREAD_COLOR)
            if im is None:
                continue
            im = cv2.resize(im, (image_size, image_size), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(im.astype("float32") / 255.0).permute(2, 0, 1)[None]
            for c in range(3):
                grads[c].append(gradient_field(x[:, c : c + 1]).numpy().ravel())
                laps[c].append(laplacian(x[:, c : c + 1]).numpy().ravel())
        import numpy as np

        gm, gs, lm, ls = [], [], [], []
        for c in range(3):
            g = np.concatenate(grads[c])
            l = np.concatenate(laps[c])
            gm.append(float(g.mean()))
            gs.append(float(g.std() + _EPS))
            lm.append(float(l.mean()))
            ls.append(float(l.std() + _EPS))
        return cls(gm, gs, lm, ls, source=f"estimated_rgb:{folder}")

    @torch.no_grad()
    def deviation_map(self, x: torch.Tensor) -> torch.Tensor:
        """Channel-averaged z-score deviation, squashed to [0, 1).

        Accepts (B, 3, H, W); grayscale input falls back to the base class.
        """
        if x.shape[1] != 3:
            return super().deviation_map(x)
        z = 0.0
        for c in range(3):
            g = gradient_field(x[:, c : c + 1])
            l = laplacian(x[:, c : c + 1])
            zg = (g - self.grad_mean_c[c]) / (self.grad_std_c[c] + _EPS)
            zl = torch.abs((l - self.lap_mean_c[c]) / (self.lap_std_c[c] + _EPS))
            z = z + (0.5 * (zg + zl))
        return _squash01(z / 3.0)

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({
            "color": True,
            "grad_mean_c": self.grad_mean_c, "grad_std_c": self.grad_std_c,
            "lap_mean_c": self.lap_mean_c, "lap_std_c": self.lap_std_c,
        })
        return d

    @classmethod
    def from_dict(cls, d: Dict):
        if not d.get("color"):
            return PhysicsPrior.from_dict(d)
        return cls(d["grad_mean_c"], d["grad_std_c"],
                   d["lap_mean_c"], d["lap_std_c"],
                   source=d.get("source", "dict"))

    def __repr__(self) -> str:  # pragma: no cover
        return (f"PhysicsPriorColor(grad_c={self.grad_mean_c}, "
                f"lap_c={self.lap_mean_c}, source={self.source!r})")
