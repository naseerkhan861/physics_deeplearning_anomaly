"""Figure A data: real-defect coverage under matched coupling arms.

For screw/capsule/bottle, seeds 42/0/1, arms base/in/cpl/incpl
(reconstruction-matching penalty OFF; the reconstruction network and
its ordinary loss are retained in every arm):
  - export evaluation maps (anomaly map + the manuscript's Gaussian
    smoothing) for every recorded test image,
  - set a per-(category,arm,seed) threshold at an empirical 1%
    normal-test-PIXEL false-positive budget (evaluation diagnostic,
    not an independently calibrated detector threshold),
  - compute per-image defect coverage (pixel recall inside the GT
    mask) and write image-level records to coverage.csv,
  - export atlas arrays for the pre-declared median-z examples.

Prior calibration is identical across the physics arms (verified);
'base' carries no prior channel by construction.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/home/edison/agents_project/pgdn_project")

import csv
import json
import os

import cv2
import numpy as np
import torch

from datasets.loaders import MVTecScrewDataset
from defect_family_sweep import MVTEC, load

OUT = "results/defect_analysis_v1"
CATS = ("screw", "capsule", "bottle")
ARMS = ("base", "in", "cpl", "incpl")
ALPHA = 0.01
STRATA = {"screw": "manipulated_front", "capsule": "scratch",
          "bottle": "contamination"}


@torch.no_grad()
def eval_map(model, img, dev):
    x = torch.from_numpy(img[None, None].astype(np.float32)).to(dev)
    m = model.anomaly_map(x)[0, 0].cpu().numpy()
    return cv2.GaussianBlur(m, (0, 0), sigmaX=4.0)


def threshold_at_budget(normal_pixels, alpha):
    v = np.asarray(normal_pixels, np.float64).ravel()
    k = min(len(v) - 1, max(0, int(np.ceil((1 - alpha) * len(v))) - 1))
    t = float(np.partition(v, k)[k])
    return t, float(np.mean(v > t))


def main():
    os.makedirs(f"{OUT}/maps", exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # pre-declared example rule: median seed-42 z within stratum
    zrec = json.load(open("results/raw/real_moderation.json"))["categories"]
    selected = {}
    for cat, stratum in STRATA.items():
        cands = sorted((p, z) for p, z in zrec[cat]["z_in"].items()
                       if f"/{stratum}/" in p)
        cands.sort(key=lambda pz: (pz[1], pz[0]))
        selected[cat] = cands[len(cands) // 2][0]
    protocol = {"alpha": ALPHA, "threshold_rule": "strict > empirical "
                "normal-test-pixel budget", "smoothing": "Gaussian sigma 4",
                "arms": ARMS, "note": "evaluation diagnostic; not an "
                "independently calibrated deployment threshold",
                "selected_examples": selected}
    json.dump(protocol, open(f"{OUT}/protocol.json", "w"), indent=2)

    rows = []
    for cat in CATS:
        ds = MVTecScrewDataset(os.path.join(MVTEC, cat), "test",
                               image_size=128, seed=42)
        items = []
        for i, it in enumerate(ds.items):
            item = ds[i]
            img = item["image"].numpy()
            img = img[0] if img.ndim == 3 else img
            m = item["mask"][0].numpy() > 0.5
            defect_type = it["img"].split("/")[-2]
            items.append({"path": it["img"], "label": it.get("label", 0),
                          "dtype": defect_type, "img": img, "mask": m})
        masks_lost = sum(1 for e in items if e["label"] == 1 and not e["mask"].any())
        if masks_lost:
            print(f"WARNING {cat}: {masks_lost} masks empty at 128x128")

        for seed in (42, 0, 1):
            pre = "fact" if seed == 42 else "f2"
            sfx = "" if seed == 42 else f"_s{seed}"
            atlas = {}
            for arm in ARMS:
                model = load(f"{pre}_{cat}_{arm}{sfx}").to(dev)
                peval = {e["path"]: eval_map(model, e["img"], dev) for e in items}
                model = model.cpu()
                normal_px = np.concatenate([peval[e["path"]].ravel()
                                            for e in items if e["label"] == 0])
                t, achieved = threshold_at_budget(normal_px, ALPHA)
                for e in items:
                    if e["label"] != 1:
                        continue
                    cov = (float(np.mean(peval[e["path"]][e["mask"]] > t))
                           if e["mask"].any() else float("nan"))
                    rows.append([cat, e["dtype"], os.path.basename(e["path"]),
                                 seed, arm, e["mask"].sum(), round(cov, 4)])
                if seed == 42 and arm in ("in", "incpl"):
                    for e in items:
                        if e["path"] == selected[cat]:
                            atlas[arm] = peval[e["path"]]
                print(cat, seed, arm, "t=", round(t, 4),
                      "achieved=", round(achieved, 4), flush=True)
            # z map + arrays for the atlas example (seed 42)
            if seed == 42:
                zf = load(f"fact_{cat}_full").to(dev)
                sel = next(e for e in items if e["path"] == selected[cat])
                with torch.no_grad():
                    x = torch.from_numpy(sel["img"][None, None].astype(np.float32)).to(dev)
                    zmap = zf(x)["phys"][0, 0].cpu().numpy()
                zf = zf.cpu()
                np.savez_compressed(
                    f"{OUT}/maps/{cat}_atlas.npz",
                    image=sel["img"], mask=sel["mask"].astype(np.uint8),
                    prior_z=zmap, p_eval_in=atlas.get("in"),
                    p_eval_incpl=atlas.get("incpl"), path=selected[cat])

    with open(f"{OUT}/coverage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "defect_type", "image", "seed", "arm",
                    "n_mask_pixels", "coverage_at_1pct"])
        w.writerows(rows)
    print("wrote", len(rows), "coverage rows")


if __name__ == "__main__":
    main()
