"""PatchCore-lite baseline (no training).

Feature extractors, in order of preference:

1. ``resnet18`` - torchvision ResNet-18 with ImageNet weights; patch features
   are layer2 (stride 8) ++ upsampled layer3 (stride 16) concatenated.
2. ``handcrafted`` - fallback when torchvision/weights are unavailable:
   per-8x8-patch statistics (mean/std of intensity, |grad|, |Laplacian|).

The memory bank is greedily coreset-subsampled (farthest-point) to at most
``coreset_size`` vectors.  Scoring is kNN distance: per-patch distance to its
nearest bank neighbours; the pixel map is the upsampled patch-score map
(Gaussian-smoothed) and the image score is its maximum.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from physics.operators import diffusion_residual, gradient_field

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

class _ResNetPatchFeatures(torch.nn.Module):
    """layer2 ++ upsampled layer3 features -> (B, 384, 16, 16) at 128px."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x3: torch.Tensor) -> torch.Tensor:
        x = self.stem(x3)
        x = self.layer1(x)
        f2 = self.layer2(x)                     # (B, 128, 16, 16)
        f3 = self.layer3(f2)                    # (B, 256, 8, 8)
        f3 = torch.nn.functional.interpolate(
            f3, size=f2.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat([f2, f3], dim=1)       # (B, 384, 16, 16)


class _HandcraftedPatchFeatures:
    """Per-8x8-patch [mean, std, |grad| mean, |grad| std, |lap| mean, |lap| std]."""

    def __init__(self, patch: int = 8):
        self.patch = patch

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        p = self.patch
        gh, gw = h // p, w // p
        feats = []
        for field in (x, gradient_field(x), diffusion_residual(x)):
            u = field.unfold(2, p, p).unfold(3, p, p)   # (B, 1, gh, gw, p, p)
            u = u.reshape(b, gh, gw, p * p)
            feats += [u.mean(-1), u.std(-1)]
        f = torch.stack(feats, dim=1)                   # (B, 6, gh, gw)
        return f


def _build_extractor(require_pretrained=None):
    """Returns (extractor, name). With require_pretrained=True (or the
    PGDN_REQUIRE_PRETRAINED env var set), a torchvision/weights failure
    raises instead of falling back -- required for reproducing
    pretrained-baseline claims."""
    import os as _os
    require = (require_pretrained if require_pretrained is not None
               else _os.environ.get("PGDN_REQUIRE_PRETRAINED", "") not in ("", "0"))
    try:
        return _ResNetPatchFeatures(), "resnet18_layer2+3"
    except Exception as e:  # torchvision missing / no weights / offline
        if require:
            raise RuntimeError(
                f"pretrained extractor required (PGDN_REQUIRE_PRETRAINED) "
                f"but unavailable: {type(e).__name__}: {e}") from e
        print(f"[patchcore-lite] resnet18 unavailable ({type(e).__name__}: {e}); "
              f"falling back to handcrafted patch features")
        return _HandcraftedPatchFeatures(patch=8), "handcrafted_8x8"


# ---------------------------------------------------------------------------
# PatchCore-lite
# ---------------------------------------------------------------------------

class PatchCoreLite:
    def __init__(
        self,
        coreset_size: int = 2000,
        k: int = 1,
        image_size: int = 128,
        seed: int = 0,
        max_bank_before_coreset: int = 40000,
    ):
        self.coreset_size = coreset_size
        self.k = k
        self.image_size = image_size
        self.seed = seed
        self.max_bank_before_coreset = max_bank_before_coreset
        self.extractor, self.extractor_name = _build_extractor()
        self.bank: Optional[torch.Tensor] = None
        self._feat_mean: Optional[torch.Tensor] = None
        self._feat_std: Optional[torch.Tensor] = None

    # -- helpers ------------------------------------------------------------

    def _prep(self, img: np.ndarray) -> torch.Tensor:
        """HxW, (H,W,C) or (C,H,W) float/uint8 -> (1, C', S, S) in [0,1].

        C'=3 keeps true colour for the ResNet path; the handcrafted path is
        reduced to luminance inside _embed.
        """
        if img.ndim == 3:
            if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):  # CHW
                img = img.transpose(1, 2, 0)
            if img.shape[-1] == 1:
                img = img[..., 0]
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = img[..., None]
        return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None]

    @torch.no_grad()
    def _embed(self, img: np.ndarray) -> torch.Tensor:
        """One image -> (n_patches, D) feature matrix."""
        x = self._prep(img)
        if isinstance(self.extractor, torch.nn.Module):
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            x3 = (x - _IMAGENET_MEAN.to(x.dtype)) / _IMAGENET_STD.to(x.dtype)
            f = self.extractor(x3)
        else:
            if x.shape[1] == 3:  # handcrafted features act on luminance
                w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
                x = (x * w).sum(dim=1, keepdim=True)
            f = self.extractor(x)
        b, d, gh, gw = f.shape
        self._grid = (gh, gw)
        return f.permute(0, 2, 3, 1).reshape(-1, d)

    # -- fit ----------------------------------------------------------------

    @torch.no_grad()
    def fit(self, images: List[np.ndarray]) -> "PatchCoreLite":
        feats = [self._embed(im) for im in images]
        bank = torch.cat(feats, dim=0)
        # standardize (important for handcrafted; harmless for resnet feats)
        self._feat_mean = bank.mean(0, keepdim=True)
        self._feat_std = bank.std(0, keepdim=True) + 1e-8
        bank = (bank - self._feat_mean) / self._feat_std
        if bank.shape[0] > self.max_bank_before_coreset:
            rng = np.random.RandomState(self.seed)
            keep = rng.choice(bank.shape[0], self.max_bank_before_coreset, replace=False)
            bank = bank[torch.from_numpy(np.sort(keep))]
        self.bank = self._greedy_coreset(bank, self.coreset_size, self.seed)
        return self

    @staticmethod
    def _greedy_coreset(bank: torch.Tensor, m: int, seed: int) -> torch.Tensor:
        """Greedy farthest-point coreset with incremental min-dist updates."""
        n = bank.shape[0]
        if n <= m:
            return bank
        g = torch.Generator().manual_seed(seed)
        first = int(torch.randint(n, (1,), generator=g))
        sel = [first]
        min_d = torch.cdist(bank, bank[first : first + 1]).squeeze(1)
        min_d[first] = -1.0
        for _ in range(m - 1):
            nxt = int(torch.argmax(min_d))
            sel.append(nxt)
            d = torch.cdist(bank, bank[nxt : nxt + 1]).squeeze(1)
            min_d = torch.minimum(min_d, d)
            min_d[nxt] = -1.0
        return bank[torch.tensor(sel)]

    # -- score --------------------------------------------------------------

    @torch.no_grad()
    def score(self, img: np.ndarray) -> Tuple[float, np.ndarray]:
        """Returns (image_score, pixel_map HxW float32)."""
        assert self.bank is not None, "call fit() first"
        q = self._embed(img)
        q = (q - self._feat_mean) / self._feat_std
        d = torch.cdist(q, self.bank)                      # (n_patches, M)
        k = min(self.k, d.shape[1])
        patch_scores = d.topk(k, dim=1, largest=False).values.mean(1)
        gh, gw = self._grid
        m = patch_scores.reshape(gh, gw).numpy()
        m = cv2.resize(m, (self.image_size, self.image_size),
                       interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=4.0)
        return float(m.max()), m.astype(np.float32)

    def score_batch(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        scores, maps = [], []
        for im in images:
            s, m = self.score(im)
            scores.append(s)
            maps.append(m)
        return np.array(scores, dtype=np.float64), np.stack(maps)
