"""Surface-compatibility metrics and held-out validation (review P5).

Measures, per category, properties computable WITHOUT defect labels or
trained models, then tests how well they predict where physics guidance
helps (PGDN - DRAEM delta):

  edge_density   -- fraction of pixels with |grad| above the good-image
                    95th percentile, on defect-free test images (normal
                    contours that may trigger false evidence)
  stationarity   -- coefficient of variation of blockwise (8x8) derivative
                    means on good images (tests the stationarity assumption)
  prior_mask_sep -- label-FREE variant not usable here; instead we report
                    z-response inside vs outside REAL masks (diagnostic,
                    uses defect labels; marked as such)

Rule development/evaluation: fit a sign predictor on half the categories,
evaluate on the held-out half.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch

from datasets.loaders import MVTecScrewDataset
from physics.operators import PhysicsPrior, gradient_field

MVTEC = "/home/edison/VS_Projects/defectsGenerations/mvtec_dataset"
CATS = ("bottle cable capsule carpet grid hazelnut leather metal_nut pill "
        "screw tile toothbrush transistor wood zipper").split()
RF = "results/rgb_full_results.json"


def cat_metrics(root):
    ds_g = MVTecScrewDataset(root, "test", image_size=128, seed=42)
    good = [it for it in ds_g.items if it.get("label", 0) == 0][:40]
    prior = PhysicsPrior.estimate_from_folder(os.path.join(root, "train", "good"),
                                              image_size=128)
    dens, stat_cv, z_in, z_out = [], [], [], []
    for it in good:
        im = cv2.imread(it["img"], cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (128, 128)).astype(np.float32) / 255.0
        x = torch.from_numpy(im)[None, None]
        g = gradient_field(x)[0, 0].numpy()
        dens.append(float((g > np.quantile(g, 0.95)).mean()))
        b = 16
        blocks = g.reshape(8, b, 8, b).mean(axis=(1, 3)).ravel()
        stat_cv.append(float(blocks.std() / (blocks.mean() + 1e-8)))
    # diagnostic (uses labels): z inside vs outside real defect masks
    for it in ds_g.items:
        if it.get("label", 0) != 1:
            continue
        item = ds_g[ds_g.items.index(it)]
        im = item["image"][0].numpy()
        x = torch.from_numpy(im)[None, None]
        z = prior.deviation_map(x)[0, 0].numpy()
        m = item["mask"][0].numpy() > 0.5
        if m.sum() < 8:
            continue
        z_in.append(float(z[m].mean()))
        z_out.append(float(z[~m].mean()))
    return {
        "edge_density": float(np.mean(dens)),
        "stationarity_cv": float(np.mean(stat_cv)),
        "z_in_minus_out": float(np.mean(z_in) - np.mean(z_out)) if z_in else None,
    }


def main():
    rf = json.load(open(RF))
    rows = {}
    for c in CATS:
        m = cat_metrics(os.path.join(MVTEC, c))
        delta = rf["primary"][c]["pgdn"]["image_auroc_mean"] - \
            rf["primary"][c]["draem"]["image_auroc_mean"]
        m["delta"] = delta
        rows[c] = m
        print(f"{c:12s} edge={m['edge_density']:.4f} statcv={m['stationarity_cv']:.3f} "
              f"zin-out={m['z_in_minus_out']} delta={delta:+.3f}")

    # held-out validation: fit sign rule on half, test on the other half
    names = sorted(rows)
    half = len(names) // 2
    dev, test = names[:half], names[half:]

    def evaluate(feature, dev_c, test_c, label):
        d = [(rows[c][feature], rows[c]["delta"]) for c in dev_c]
        # sign rule with threshold chosen on dev: predict help iff feat < thr
        # (for edge density / stationarity: lower = physics helps)
        thr = float(np.median([x for x, _ in d]))
        correct = sum(1 for c in test_c
                      if ((rows[c][feature] < thr) == (rows[c]["delta"] > 0)))
        dd = [(rows[c][feature], rows[c]["delta"]) for c in dev_c + test_c]
        rho = float(np.corrcoef([x for x, _ in dd], [y for _, y in dd])[0, 1])
        print(f"{label}: dev-median thr={thr:.4f}; held-out accuracy "
              f"{correct}/{len(test_c)}; overall rho={rho:+.3f}")
        return correct, len(test_c), rho

    out = {"per_category": rows, "held_out": {}}
    for feat, lab, direction in (("edge_density", "edge density", "less"),
                                 ("stationarity_cv", "stationarity CV", "less"),
                                 ("z_in_minus_out", "z in-out (labelled diag)", "greater")):
        try:
            correct, n, rho = evaluate(feat, dev, test, lab)
            out["held_out"][feat] = {"correct": correct, "n": n, "rho": rho}
        except Exception as e:
            print(lab, "failed:", e)
    # reverse split for stability
    dev2, test2 = names[half:], names[:half]
    for feat, lab in (("edge_density", "edge density"), ("stationarity_cv", "stationarity CV")):
        d = [(rows[c][feat], rows[c]["delta"]) for c in dev2]
        thr = float(np.median([x for x, _ in d]))
        correct = sum(1 for c in test2 if ((rows[c][feat] < thr) == (rows[c]["delta"] > 0)))
        print(f"{lab} (reverse split): held-out accuracy {correct}/{len(test2)}")
        out["held_out"][f"{feat}_rev"] = {"correct": correct, "n": len(test2)}

    with open("results/raw/compat_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/raw/compat_metrics.json")


if __name__ == "__main__":
    main()
