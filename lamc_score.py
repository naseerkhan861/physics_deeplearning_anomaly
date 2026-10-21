"""Score the lambda_C intervention arms (round-3 review Step 10).

Evaluates the coupling-only screw arms trained at several lambda_C values,
plus the C-off baseline (fact_screw_base), on matched Perlin and smooth
(default parameterisation) defect sets over the same 30 backgrounds.
Prediction: increasing lambda_C deepens the smooth-defect failure while
the Perlin scores and the C-off arm stay stable.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from datasets.loaders import MVTecScrewDataset
from defect_family_sweep import MVTEC, load, score, smooth_defect
from smooth_defect_test import perlin_defect

RUNS = [("C-off (base)", "fact_screw_base"),
        ("0.1", "lamc_screw_cpl_l0.1"),
        ("0.25", "lamc_screw_cpl_l0.25"),
        ("0.5", "lamc_screw_cpl_l0.5"),
        ("1.0", "lamc_screw_cpl_l1.0")]

ds = MVTecScrewDataset(MVTEC + "/screw", "test", image_size=128, seed=42)
good = [it for it in ds.items if it.get("label", 0) == 0][:30]
imgs = []
for it in good:
    im = cv2.imread(it["img"], cv2.IMREAD_GRAYSCALE)
    imgs.append(cv2.resize(im, (128, 128)).astype(np.float32) / 255.0)

res = {"protocol": "30 paired backgrounds, fixed per-image evaluation seed; "
                   "coupling-only arm trained at each lambda_C (8-epoch gray "
                   "recipe, seed 42, cuda)"}
for tag, run in RUNS:
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
    res[tag] = aucs
    print(tag, aucs)
    del model

with open("results/raw/lamc_screw_smooth.json", "w") as f:
    json.dump(res, f, indent=2)
print("wrote results/raw/lamc_screw_smooth.json")
