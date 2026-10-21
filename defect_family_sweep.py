"""Defect-family controlled sweeps (round-3 review Step 9).

The smooth-defect generator changes contrast, defect frequency, and
boundary softness together.  This script varies each factor separately
while holding the others at the smooth-defect defaults, scores the frozen
seed-42 factorial checkpoints on the paired backgrounds, and reports
image AUROC per (category, arm, factor, level) plus a derivative-energy
ratio that measures how weak the defect's derivative signature is.

Factors (defaults from smooth_defect_test.py):
  contrast  alpha  in {0.10, 0.18, 0.25*, 0.40}
  frequency mask blob scale in {size//16 low, size//8*, size//4 high}
  softness  mask blur sigma in {size/128 hard, size/64*, size/32 soft}
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

CONTRASTS = [0.10, 0.18, 0.25, 0.40]
# grid cells per axis at 128px: divisor 16 -> 16 cells (fine, high
# spatial frequency); divisor 4 -> 4 cells (coarse, low frequency)
FREQS = [("high", 16), ("mid", 8), ("low", 4)]     # size // divisor
SOFTS = [("hard", 128), ("mid", 64), ("soft", 32)]  # sigma = size / divisor

ARMS = ("base in rec cpl inrec incpl recpl full").split()


def smooth_defect(clean, rng, size, alpha=0.25, blob_div=8, soft_div=64):
    """Parameterised weak-derivative defect (defaults reproduce the paper's
    smooth family: alpha=0.25, blob_div=8, soft_div=64)."""
    big = _perlin_2d(rng, (size, size), scale=max(2, size // blob_div))
    thr = np.quantile(big, 0.85)
    mask = (big >= thr).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=size / soft_div)
    mask = (mask > 0.3).astype(np.float32)
    src = cv2.GaussianBlur(_perlin_2d(rng, (size, size), scale=max(2, size // 16)),
                           (0, 0), sigmaX=size / 32.0)
    out = clean * (1 - mask) + (alpha * src + (1 - alpha) * clean) * mask
    return out.astype(np.float32), mask


def deriv_ratio(clean, defect, mask):
    """Mean gradient energy of the defect region relative to the clean
    background inside the same mask: low values = weak derivative signal."""
    gx = np.abs(cv2.Sobel(defect, cv2.CV_32F, 1, 0, ksize=3)) \
        + np.abs(cv2.Sobel(defect, cv2.CV_32F, 0, 1, ksize=3))
    g0 = np.abs(cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)) \
        + np.abs(cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3))
    m = mask > 0
    if m.sum() < 4:
        return None
    return float((gx[m].mean() + 1e-6) / (g0[m].mean() + 1e-6))


@torch.no_grad()
def score(model, img):
    x = torch.from_numpy(img[None, None].astype(np.float32))
    m = model.anomaly_map(x)[0, 0].numpy()
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=4.0)
    return float(m.max())


RUNS_ROOT = os.environ.get("PGDN_RUNS_ROOT", "runs")


def load(run):
    ckpt = torch.load(f"{RUNS_ROOT}/{run}/last.pt", map_location="cpu",
                      weights_only=False)
    m = PGDN.from_config(ckpt["model_config"])
    m.load_state_dict(ckpt["model_state"])
    return m.eval()


def variants():
    """(factor, level, kwargs) list; defaults marked level 'default'."""
    out = [("contrast", f"{a:.2f}", dict(alpha=a)) for a in CONTRASTS]
    out += [("frequency", f, dict(blob_div=d)) for f, d in FREQS]
    out += [("softness", s, dict(soft_div=d)) for s, d in SOFTS]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="screw capsule bottle")
    ap.add_argument("--arms", default="")  # subset override, e.g. "base cpl full"
    ap.add_argument("--runs-suffix", default="fact_{cat}_{arm}")
    ap.add_argument("--n-good", type=int, default=30)
    ap.add_argument("--out", default="results/raw/defect_family_sweep.json")
    args = ap.parse_args()
    arms = args.arms.split() if args.arms else ARMS

    res = {"protocol": "paired backgrounds, 30 held-out normals per category, "
                       "fixed per-image evaluation seed; defaults reproduce the "
                       "smooth family (contrast 0.25, blob size//8, sigma size/64)",
           "sweeps": {}}
    for cat in args.cats.split():
        ds = MVTecScrewDataset(os.path.join(MVTEC, cat), "test", image_size=128, seed=42)
        good = [it for it in ds.items if it.get("label", 0) == 0][: args.n_good]
        imgs = []
        for it in good:
            im = cv2.imread(it["img"], cv2.IMREAD_GRAYSCALE)
            imgs.append(cv2.resize(im, (128, 128)).astype(np.float32) / 255.0)
        cat_res = {}
        for arm in arms:
            run = args.runs_suffix.format(cat=cat, arm=arm)
            if not os.path.isfile(f"{RUNS_ROOT}/{run}/last.pt"):
                print("skip (no checkpoint):", run)
                continue
            model = load(run)
            entry = {}
            for factor, level, kw in variants():
                labels, scores, drs = [], [], []
                for i, im in enumerate(imgs):
                    rng = np.random.RandomState((42 * 10007 + i * 131) % (2 ** 31))
                    d, m = smooth_defect(im, rng, 128, **kw)
                    labels += [0, 1]
                    scores += [score(model, im), score(model, d)]
                    dr = deriv_ratio(im, d, m)
                    if dr is not None:
                        drs.append(dr)
                entry[level if factor == "contrast" else f"{factor}:{level}"] = {
                    "image_auroc": float(roc_auc_score(labels, scores)),
                    "deriv_ratio": float(np.mean(drs)) if drs else None,
                }
            cat_res[arm] = entry
            print(cat, arm, json.dumps(entry))
            del model
        res["sweeps"][cat] = cat_res
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
