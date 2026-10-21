"""Single-image inference CLI.

Loads a PGDN checkpoint, scores one image, prints the image-level anomaly
score, and saves heatmap overlays:
  * ``<stem>_anomaly.png``  - predicted anomaly map (seg head) overlaid
  * ``<stem>_physics.png``  - physics residual map overlaid (explainability)

Example:
    python infer.py --checkpoint runs/pgdn_bs/last.pt \
        --image /mnt/agents/work/data/construct/brushed_steel/test/defective/brushed_steel_test_defect_000.png \
        --out-dir runs/pgdn_bs/infer
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from models.pgdn import PGDN


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PGDN single-image inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out-dir", default=None,
                   help="default: <run_dir>/infer")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--alpha", type=float, default=0.6, help="overlay opacity")
    return p.parse_args(argv)


def _normalize01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-8:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _overlay(gray: np.ndarray, heat: np.ndarray, alpha: float) -> np.ndarray:
    base = cv2.cvtColor((gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    hm = cv2.applyColorMap((_normalize01(heat) * 255).astype(np.uint8),
                           cv2.COLORMAP_JET)
    return cv2.addWeighted(base, 1.0 - alpha, hm, alpha, 0)


@torch.no_grad()
def run(args) -> dict:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PGDN.from_config(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"could not read {args.image}")
    img = cv2.resize(img, (args.image_size, args.image_size),
                     interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]

    out = model(x)
    amap = torch.sigmoid(out["seg_logits"])[0, 0].numpy()
    amap_s = cv2.GaussianBlur(amap, (0, 0), sigmaX=4.0)
    score = float(amap_s.max())

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), "infer")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    g = img.astype(np.float32) / 255.0
    anomaly_png = os.path.join(out_dir, f"{stem}_anomaly.png")
    cv2.imwrite(anomaly_png, _overlay(g, amap, args.alpha))

    physics_png = None
    if out["phys"] is not None:
        pmap = out["phys"][0, 0].numpy()
        physics_png = os.path.join(out_dir, f"{stem}_physics.png")
        cv2.imwrite(physics_png, _overlay(g, pmap, args.alpha))

    result = {
        "image": args.image,
        "image_score": score,
        "anomaly_overlay": anomaly_png,
        "physics_overlay": physics_png,
        "use_physics_arch": model.use_physics_arch,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run(parse_args())
