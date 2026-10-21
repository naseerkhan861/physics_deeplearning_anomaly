"""Dataset loaders for PGDN.

* :class:`MVTecScrewDataset` - MVTec AD ``screw`` category (primary benchmark).
* :class:`ConstructDataset`  - synthetic CONSTRUCT dataset with per-class
  physics statistics and severity metadata (``splits.csv``).
* :class:`NEUDetDataset`     - loader *stub* for NEU-DET (no defect-free
  images, no masks; usable for unlabeled inference/visualisation only).

Train split yields ``{"clean", "augmented", "mask"}`` where ``augmented`` is
produced by DRAEM-style online synthetic-defect augmentation (Perlin/fractal
mask + anomaly source blended into the defect-free image).  Test split yields
``{"image", "mask", "label", "path", "severity"}`` with ground-truth pixel
masks.  All images are grayscale, resized to ``image_size``, float in
``[0, 1]``, shape ``(1, H, W)``.

Deviation from DRAEM: no external anomaly-source dataset (e.g. DTD) is
available in this environment, so the anomaly source is a second fractal
Perlin texture (plus an optional randomly shifted copy of the input image),
which plays the same role.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# ---------------------------------------------------------------------------
# Image IO helpers
# ---------------------------------------------------------------------------

def _list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(_IMG_EXTS)
    )


def _load_gray(path: str, image_size: int, channels: int = 1) -> np.ndarray:
    flag = cv2.IMREAD_COLOR if channels == 3 else cv2.IMREAD_GRAYSCALE
    im = cv2.imread(path, flag)
    if im is None:
        raise IOError(f"could not read image: {path}")
    if im.shape[:2] != (image_size, image_size):
        im = cv2.resize(im, (image_size, image_size), interpolation=cv2.INTER_AREA)
    im = im.astype(np.float32) / 255.0
    if channels == 3 and im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    return im


def _load_mask(path: Optional[str], image_size: int) -> np.ndarray:
    if path is None or not os.path.isfile(path):
        return np.zeros((image_size, image_size), dtype=np.float32)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((image_size, image_size), dtype=np.float32)
    if m.shape[:2] != (image_size, image_size):
        m = cv2.resize(m, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    # MVTec-style masks are {0,255}; VisA masks are {0,1} — adapt threshold
    thr = 127 if (m.max() > 1) else 0
    return (m > thr).astype(np.float32)


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    if a.ndim == 3:  # (H, W, C) -> (C, H, W)
        return torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1))
    return torch.from_numpy(a[None].astype(np.float32))


def _subsample_indices(n: int, frac: float, seed: int) -> np.ndarray:
    idx = np.arange(n)
    if frac >= 1.0 or n == 0:
        return idx
    k = max(1, int(round(n * frac)))
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(idx, size=k, replace=False))


# ---------------------------------------------------------------------------
# Perlin / fractal noise (numpy + cv2, no external deps)
# ---------------------------------------------------------------------------

def _perlin_2d(rng: np.random.RandomState, shape: Tuple[int, int], scale: int) -> np.ndarray:
    h, w = shape
    gh, gw = max(2, h // scale), max(2, w // scale)
    grid = rng.rand(gh, gw).astype(np.float32)
    return cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)


def fractal_noise(
    rng: np.random.RandomState, shape: Tuple[int, int], octaves: int = 6, base_scale: int = 4
) -> np.ndarray:
    """Fractal (multi-octave Perlin) noise normalized to [0, 1]."""
    out = np.zeros(shape, dtype=np.float32)
    amp_total = 0.0
    for o in range(octaves):
        amp = 0.6 ** o
        out += amp * _perlin_2d(rng, shape, base_scale * (2 ** o))
        amp_total += amp
    out /= max(amp_total, 1e-8)
    out -= out.min()
    out /= max(out.max(), 1e-8)
    return out.astype(np.float32)


class SyntheticDefectAugmenter:
    """DRAEM-style online synthetic-defect augmentation.

    Given a defect-free image, produces ``(augmented, mask)`` where ``mask``
    is a non-empty binary defect region and ``augmented`` blends an anomaly
    source into that region with a Perlin-modulated beta map.

    ``source_dir`` optionally switches the anomaly source from fractal
    Perlin noise to an external natural-image bank (DTD-class source; here,
    cross-category images), as in the DRAEM anomaly-source ablations.
    """

    def __init__(
        self,
        image_size: int = 128,
        min_area_frac: float = 0.005,
        max_area_frac: float = 0.20,
        max_attempts: int = 8,
        source_dir: Optional[str] = None,
        channels: int = 1,
    ):
        self.image_size = image_size
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.max_attempts = max_attempts
        self.channels = channels
        self.source_paths: List[str] = []
        if source_dir and os.path.isdir(source_dir):
            self.source_paths = _list_images(source_dir)

    def _make_mask(self, rng: np.random.RandomState) -> np.ndarray:
        shape = (self.image_size, self.image_size)
        n_px = shape[0] * shape[1]
        for _ in range(self.max_attempts):
            noise = fractal_noise(rng, shape)
            # random area fraction -> percentile threshold
            frac = rng.uniform(self.min_area_frac, self.max_area_frac)
            thr = np.quantile(noise, 1.0 - frac)
            mask = (noise >= thr).astype(np.float32)
            if mask.sum() >= max(8, self.min_area_frac * n_px * 0.5):
                return mask
        # fallback: force the top-k pixels (guarantees a non-empty mask)
        k = max(8, int(self.min_area_frac * n_px))
        flat = np.argsort(noise.ravel())[-k:]
        mask = np.zeros(n_px, dtype=np.float32)
        mask[flat] = 1.0
        return mask.reshape(shape)

    def _external_source(self, rng: np.random.RandomState) -> Optional[np.ndarray]:
        """Random external-source image, resized, matching channel count."""
        if not self.source_paths:
            return None
        path = self.source_paths[rng.randint(len(self.source_paths))]
        flag = cv2.IMREAD_COLOR if self.channels == 3 else cv2.IMREAD_GRAYSCALE
        im = cv2.imread(path, flag)
        if im is None:
            return None
        im = cv2.resize(im, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        im = im.astype(np.float32) / 255.0
        if rng.rand() < 0.5:  # cheap variety: flips
            im = cv2.flip(im, rng.randint(3) - 1)
        if self.channels == 3 and im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        return im

    def __call__(
        self, clean: np.ndarray, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray]:
        shape2d = clean.shape[:2]
        mask = self._make_mask(rng)
        # anomaly source: external natural image when a source bank is
        # configured, else fractal texture, occasionally mixed with a shifted
        # copy of the input (keeps anomalies in-distribution in intensity)
        ext = self._external_source(rng)
        source = fractal_noise(rng, shape2d) if ext is None else ext
        if ext is None and rng.rand() < 0.5:
            dx, dy = rng.randint(-16, 17, size=2)
            shifted = np.roll(np.roll(clean, dx, axis=0), dy, axis=1)
            alpha = rng.uniform(0.3, 0.7)
            source = alpha * source[..., None] + (1.0 - alpha) * shifted \
                if clean.ndim == 3 else alpha * source + (1.0 - alpha) * shifted
        if source.ndim != clean.ndim:  # e.g. 2-D fractal source, RGB input
            source = np.broadcast_to(source[..., None], clean.shape).copy()
        # Perlin-modulated blending coefficient (DRAEM beta map)
        beta = fractal_noise(rng, shape2d, octaves=4) * 0.8 + 0.2
        beta = beta[..., None] if clean.ndim == 3 else beta
        blended = beta * source + (1.0 - beta) * clean
        augmented = clean * (1.0 - mask[..., None] if clean.ndim == 3 else 1.0 - mask) \
            + blended * (mask[..., None] if clean.ndim == 3 else mask)
        return augmented.astype(np.float32), mask


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _AnomalyDataset(Dataset):
    """Shared logic: frac subsampling, train-time augmentation, tensors."""

    def __init__(
        self,
        split: str,
        image_size: int = 128,
        frac: float = 1.0,
        seed: int = 0,
        augment: bool = True,
        channels: int = 1,
        source_dir: Optional[str] = None,
    ):
        assert split in ("train", "test")
        self.split = split
        self.image_size = image_size
        self.channels = channels
        self.seed = seed
        self.augmenter = (
            SyntheticDefectAugmenter(image_size, source_dir=source_dir,
                                     channels=channels)
            if (split == "train" and augment)
            else None
        )
        # filled by subclasses: list of dicts with at least "img"; test items
        # additionally have "mask" (or None), "label", "severity", "defect_type"
        self.items: List[Dict] = []
        self._build()
        keep = _subsample_indices(len(self.items), frac, seed)
        self.items = [self.items[i] for i in keep]

    def _build(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.items)

    def _rng(self, idx: int) -> np.random.RandomState:
        # deterministic per (epoch-agnostic) index; workers get different
        # streams through torch's base_seed via get_worker_info if desired
        return np.random.RandomState((self.seed * 100003 + idx * 7919) % (2 ** 31))

    def __getitem__(self, idx: int) -> Dict:
        it = self.items[idx]
        clean = _load_gray(it["img"], self.image_size, self.channels)
        if self.split == "train":
            rng = self._rng(idx)
            if self.augmenter is not None:
                aug, mask = self.augmenter(clean, rng)
            else:
                aug, mask = clean.copy(), np.zeros(clean.shape[:2], dtype=np.float32)
            return {
                "clean": _to_tensor(clean),
                "augmented": _to_tensor(aug),
                "mask": _to_tensor(mask),
                "path": it["img"],
            }
        mask = _load_mask(it.get("mask"), self.image_size)
        return {
            "image": _to_tensor(clean),
            "mask": _to_tensor(mask),
            "label": int(it.get("label", 0)),
            "path": it["img"],
            "severity": it.get("severity") or "",
            "defect_type": it.get("defect_type") or "",
        }


# ---------------------------------------------------------------------------
# MVTec AD screw
# ---------------------------------------------------------------------------

class MVTecScrewDataset(_AnomalyDataset):
    """MVTec AD ``screw`` category.

    Train: ``<root>/train/good``.  Test: ``<root>/test/{good,<defects>}`` with
    pixel masks from ``<root>/ground_truth/<defect>/<name>_mask.png``.
    """

    def __init__(self, root: str, split: str, **kw):
        self.root = root
        super().__init__(split, **kw)

    def _build(self) -> None:
        if self.split == "train":
            self.items = [
                {"img": p, "label": 0} for p in _list_images(os.path.join(self.root, "train", "good"))
            ]
            return
        test_dir = os.path.join(self.root, "test")
        gt_dir = os.path.join(self.root, "ground_truth")
        for defect in sorted(os.listdir(test_dir)):
            dpath = os.path.join(test_dir, defect)
            if not os.path.isdir(dpath):
                continue
            for img in _list_images(dpath):
                if defect == "good":
                    mask_path, label = None, 0
                else:
                    stem = os.path.splitext(os.path.basename(img))[0]
                    mask_path = os.path.join(gt_dir, defect, stem + "_mask.png")
                    label = 1
                self.items.append(
                    {
                        "img": img,
                        "mask": mask_path,
                        "label": label,
                        "defect_type": "" if defect == "good" else defect,
                        "severity": "",
                    }
                )


# ---------------------------------------------------------------------------
# CONSTRUCT synthetic dataset
# ---------------------------------------------------------------------------

class ConstructDataset(_AnomalyDataset):
    """CONSTRUCT controlled-physics dataset.

    ``class_name=None`` uses all classes.  Train: ``<class>/train/good``.
    Test: ``<class>/test/{good,defective}`` with masks in ``<class>/test/masks``
    (same filename as the defective image) and severity metadata from
    ``splits.csv``.
    """

    CLASSES = ("brushed_steel", "cast_plate", "rolled_sheet")

    def __init__(self, root: str, split: str, class_name: Optional[str] = None, **kw):
        self.root = root
        self.class_name = class_name
        super().__init__(split, **kw)

    def _classes(self) -> Sequence[str]:
        if self.class_name:
            return [self.class_name]
        return [c for c in self.CLASSES if os.path.isdir(os.path.join(self.root, c))]

    def _splits_meta(self) -> Dict[str, Dict]:
        path = os.path.join(self.root, "splits.csv")
        meta: Dict[str, Dict] = {}
        if os.path.isfile(path):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    meta[os.path.basename(row["filename"])] = row
        return meta

    def _build(self) -> None:
        if self.split == "train":
            for c in self._classes():
                for p in _list_images(os.path.join(self.root, c, "train", "good")):
                    self.items.append({"img": p, "label": 0, "class": c})
            return
        meta = self._splits_meta()
        for c in self._classes():
            for p in _list_images(os.path.join(self.root, c, "test", "good")):
                self.items.append(
                    {"img": p, "mask": None, "label": 0, "class": c,
                     "defect_type": "", "severity": ""}
                )
            mask_dir = os.path.join(self.root, c, "test", "masks")
            for p in _list_images(os.path.join(self.root, c, "test", "defective")):
                name = os.path.basename(p)
                m = meta.get(name, {})
                self.items.append(
                    {
                        "img": p,
                        "mask": os.path.join(mask_dir, name),
                        "label": 1,
                        "class": c,
                        "defect_type": m.get("defect_type", ""),
                        "severity": m.get("severity", ""),
                    }
                )


# ---------------------------------------------------------------------------
# NEU-DET stub (no defect-free images, no masks)
# ---------------------------------------------------------------------------

class NEUDetDataset(Dataset):
    """Loader stub for NEU-DET.

    NEU-DET has only defective images of 6 classes and no ground-truth masks,
    so it cannot be used for the reconstruction/segmentation training or the
    standard AUROC evaluation.  This stub yields unlabeled images
    (``label=-1``, empty mask) for optional qualitative inference.
    """

    def __init__(self, root: str, class_name: Optional[str] = None, image_size: int = 128):
        self.root = root
        self.image_size = image_size
        classes = [class_name] if class_name else sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        self.items: List[Tuple[str, str]] = []
        for c in classes:
            for p in _list_images(os.path.join(root, c)):
                self.items.append((p, c))
        if not self.items:
            raise FileNotFoundError(f"no NEU-DET images under {root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        p, c = self.items[idx]
        return {
            "image": _to_tensor(_load_gray(p, self.image_size)),
            "mask": torch.zeros(1, self.image_size, self.image_size),
            "label": -1,  # unknown / unlabeled
            "path": p,
            "defect_type": c,
            "severity": "",
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    if shuffle is None:
        shuffle = getattr(dataset, "split", "test") == "train"
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=g,
        drop_last=(shuffle and len(dataset) >= batch_size),
    )
