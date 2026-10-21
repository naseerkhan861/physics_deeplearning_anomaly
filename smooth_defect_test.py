"""Weak-derivative defect test (review, loss-analysis section).

Does the physics channel / coupling loss suppress defects whose derivative
response is weak?  For each category, existing gray-128 twin checkpoints
are scored on two matched synthetic-defect families built over the same
defect-free test images:

  perlin -- the standard training-style augmentation (fractal texture)
  smooth -- large-scale low-frequency masks, soft low-frequency source,
            weak gradients inside the defect (stains / missing texture)

Reports image AUROC per family per arm.  A large perlin-vs-smooth gap for
PGDN but not for the twin is the predicted failure signature.
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from datasets.loaders import MVTecScrewDataset, _perlin_2d
from models.pgdn import PGDN

MVTEC = "/home/edison/VS_Projects/defectsGenerations/mvtec_dataset"


def smooth_defect(clean, rng, size):
    """Low-frequency mask + soft low-contrast source -> weak derivatives."""
    big = _perlin_2d(rng, (size, size), scale=max(4, size // 8))     # few blobs
    thr = np.quantile(big, 0.85)
    mask = (big >= thr).astype(np.float32)
    # smooth the mask edges (soft transition, weak gradient at boundary)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=size / 64.0)
    mask = (mask > 0.3).astype(np.float32)
    src = cv2.GaussianBlur(_perlin_2d(rng, (size, size), scale=max(2, size // 16)),
                           (0, 0), sigmaX=size / 32.0)               # smooth stain
    alpha = 0.25                                                     # low contrast
    out = clean * (1 - mask) + (alpha * src + (1 - alpha) * clean) * mask
    return out.astype(np.float32), mask


def perlin_defect(clean, rng, size):
    from datasets.loaders import SyntheticDefectAugmenter
    return SyntheticDefectAugmenter(size)(clean, rng)


@torch.no_grad()
def score(model, img):
    x = torch.from_numpy(img[None, None].astype(np.float32))
    m = model.anomaly_map(x)[0, 0].numpy()
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=4.0)
    return float(m.max())


def load(run):
    ckpt = torch.load(f"runs/{run}/last.pt", map_location="cpu", weights_only=False)
    m = PGDN.from_config(ckpt["model_config"])
    m.load_state_dict(ckpt["model_state"])
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="screw bottle capsule grid toothbrush")
    ap.add_argument("--n-good", type=int, default=40)
    ap.add_argument("--out", default="results/raw/smooth_defect_test.json")
    args = ap.parse_args()

    res = {}
    for cat in args.cats.split():
        ds = MVTecScrewDataset(os.path.join(MVTEC, cat), "test", image_size=128, seed=42)
        good = [it for it in ds.items if it.get("label", 0) == 0][: args.n_good]
        imgs = []
        for it in good:
            im = cv2.imread(it["img"], cv2.IMREAD_GRAYSCALE)
            imgs.append(cv2.resize(im, (128, 128)).astype(np.float32) / 255.0)
        entry = {}
        for meth, run in (("pgdn", f"mvtec_{cat}_pgdn"), ("draem", f"mvtec_{cat}_draem")):
            if not os.path.isfile(f"runs/{run}/last.pt"):
                continue
            model = load(run)
            aucs = {}
            for family in ("perlin", "smooth"):
                labels, scores = [], []
                for i, im in enumerate(imgs):
                    rng = np.random.RandomState((42 * 10007 + i * 131) % (2 ** 31))
                    if family == "perlin":
                        d, _ = perlin_defect(im, rng, 128)
                    else:
                        d, _ = smooth_defect(im, rng, 128)
                    labels += [0, 1]
                    scores += [score(model, im), score(model, d)]
                aucs[family] = float(roc_auc_score(labels, scores))
            entry[meth] = aucs
            del model
        if entry:
            res[cat] = entry
            print(cat, json.dumps(entry))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
