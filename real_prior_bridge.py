"""Real-defect bridge for the coupling mechanism (round-4 review Step 5).

Groups each category's real MVTec test defects by the prior's own
response inside the ground-truth mask (independent of any detector
outcome): the deviation map z is computed by the PGDN implementation on
the clean defect image, and defects are split at the median into
weak-prior and strong-prior groups.  The factorial arms are then
compared as image-level detectors per group (defect scores versus all
defect-free test scores).  Predictions: enabling C (cpl vs base,
incpl vs in) costs more on the weak-prior group.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from datasets.loaders import MVTecScrewDataset
from defect_family_sweep import MVTEC, load, score

CATS = ("screw", "capsule", "bottle")
ARMS = ("base", "in", "cpl", "incpl")


@torch.no_grad()
def zmap(model, img):
    x = torch.from_numpy(img[None, None].astype(np.float32))
    return model(x)["phys"][0, 0].numpy()


def main():
    out = {"protocol": "defects split at the median of mean prior response z "
                       "inside the GT mask (z from the full-PGDN checkpoint's "
                       "own prior computation); image AUROC per group uses all "
                       "defect-free test images as negatives; seed-42 arms",
           "categories": {}}
    for cat in CATS:
        ds = MVTecScrewDataset(os.path.join(MVTEC, cat), "test",
                               image_size=128, seed=42)
        good = [it for it in ds.items if it.get("label", 0) == 0]
        bad = [it for it in ds.items if it.get("label", 0) == 1]
        zf = load(f"fact_{cat}_full")
        entries = []
        idx_of = {it["img"]: i for i, it in enumerate(ds.items)}
        for it in bad:
            item = ds[idx_of[it["img"]]]
            img = item["image"].numpy()
            if img.ndim == 3:
                img = img[0]
            m = item["mask"][0].numpy() > 0.5
            if m.sum() < 4:
                continue
            entries.append({"img_path": it["img"], "z_in": float(zmap(zf, img)[m].mean())})
        del zf
        zs = sorted(e["z_in"] for e in entries)
        thr = zs[len(zs) // 2]
        for e in entries:
            e["group"] = "weak" if e["z_in"] < thr else "strong"
        cat_res = {"n_defects": len(entries), "z_median": thr,
                   "n_weak": sum(e["group"] == "weak" for e in entries)}
        good_scores = {}
        imgs_good = []
        for it in good:
            item = ds[idx_of[it["img"]]]
            img = item["image"].numpy()
            imgs_good.append(img[0] if img.ndim == 3 else img)
        for arm in ARMS:
            model = load(f"fact_{cat}_{arm}")
            gs = [score(model, im) for im in imgs_good]
            good_scores[arm] = gs
            per_group = {}
            for grp in ("weak", "strong"):
                yy, ss = [], []
                for e in entries:
                    if e["group"] != grp:
                        continue
                    item = ds[idx_of[e["img_path"]]]
                    img = item["image"].numpy()
                    if img.ndim == 3:
                        img = img[0]
                    yy.append(1)
                    ss.append(score(model, img))
                y = [0] * len(gs) + yy
                s = gs + ss
                per_group[grp] = round(float(roc_auc_score(y, s)), 3)
            cat_res[arm] = per_group
            del model
        out["categories"][cat] = cat_res
        print(cat, json.dumps(cat_res), flush=True)
    with open("results/raw/real_prior_bridge.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/raw/real_prior_bridge.json")


if __name__ == "__main__":
    main()
