"""Replicate the smooth-factorial panel across training seeds (round-3
review Steps 2/8: 'pending replication').

Rescores the Perlin and smooth (default parameterisation) defect families
on the frozen factorial checkpoints for seeds 42 (fact_*) and 0/1 (f2_*_s*),
using the identical paired-background protocol of smooth_factorial.json,
and writes one record per seed.
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from datasets.loaders import MVTecScrewDataset
from defect_family_sweep import MVTEC, load, score, smooth_defect
from smooth_defect_test import perlin_defect

CATS = ("screw", "capsule", "bottle")
ARMS = ("base in rec cpl inrec incpl recpl full").split()


def run_for_seed(seed, n_good):
    pre = "fact" if seed == 42 else "f2"
    sfx = "" if seed == 42 else f"_s{seed}"
    res = {}
    for cat in CATS:
        ds = MVTecScrewDataset(os.path.join(MVTEC, cat), "test",
                               image_size=128, seed=42)
        good = [it for it in ds.items if it.get("label", 0) == 0][:n_good]
        imgs = []
        for it in good:
            im = cv2.imread(it["img"], cv2.IMREAD_GRAYSCALE)
            imgs.append(cv2.resize(im, (128, 128)).astype(np.float32) / 255.0)
        for arm in ARMS:
            run = f"{pre}_{cat}_{arm}{sfx}"
            if not os.path.isfile(f"runs/{run}/last.pt"):
                print("skip (no checkpoint):", run)
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
            res[f"{cat}/{arm}"] = aucs
            print(seed, f"{cat}/{arm}", aucs, flush=True)
            del model
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0 1")
    ap.add_argument("--n-good", type=int, default=30)
    args = ap.parse_args()
    for seed in [int(s) for s in args.seeds.split()]:
        out = f"results/raw/smooth_factorial_s{seed}.json"
        res = run_for_seed(seed, args.n_good)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print("wrote", out, f"({len(res)} cells)")


if __name__ == "__main__":
    main()
