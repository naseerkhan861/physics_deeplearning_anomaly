"""The physics router: convert the regime boundary into a method.

For each MVTec category and seed, both trained RGB/256 models (PGDN,
DRAEM-lite) are scored on a LABEL-FREE validation task: detecting
DRAEM-style synthetic defects generated over defect-free test/good images
(never seen in training; no real defect information used).  The per-category
winner is routed; the routed ensemble is evaluated on the real test sets.

Routing variants:
  argmax     - pick the higher synthetic image AUROC
  margin>=t  - route to physics only if its synthetic advantage >= t
               (default t = 0.02; otherwise the macro-favoured DRAEM)

Outputs results/raw/router_results.json.
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from datasets.loaders import MVTecScrewDataset, SyntheticDefectAugmenter
from models.pgdn import PGDN

CATS = ("bottle cable capsule carpet grid hazelnut leather metal_nut pill "
        "screw tile toothbrush transistor wood zipper").split()
SEEDS = (42, 0, 1)
RAW = "results/raw"


def load_model(run, device):
    ckpt = torch.load(f"runs/{run}/last.pt", map_location="cpu", weights_only=False)
    m = PGDN.from_config(ckpt["model_config"])
    m.load_state_dict(ckpt["model_state"])
    return m.eval().to(device)


@torch.no_grad()
def image_score(model, x):
    m = model.anomaly_map(x)[0, 0].cpu().numpy()
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=4.0)
    return float(m.max())


def synthetic_val(root, model, seed, device, n_good=60, image_size=256):
    """Image AUROC of `model` on good vs synthetic-defect val images."""
    ds = MVTecScrewDataset(root, "test", image_size=image_size, seed=seed,
                           channels=3)
    good = [it for it in ds.items if it.get("label", 0) == 0][:n_good]
    aug = SyntheticDefectAugmenter(image_size, channels=3)
    labels, scores = [], []
    for i, it in enumerate(good):
        img = cv2.imread(it["img"], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (image_size, image_size),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        clean = torch.from_numpy(img).permute(2, 0, 1)[None].to(device)
        rng = np.random.RandomState((seed * 10007 + i * 131) % (2 ** 31))
        defect, _ = aug(img, rng)
        defect = torch.from_numpy(defect).permute(2, 0, 1)[None].to(device)
        labels += [0, 1]
        scores += [image_score(model, clean), image_score(model, defect)]
    return float(roc_auc_score(labels, scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec-root", default="/home/edison/VS_Projects/defectsGenerations/mvtec_dataset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--out", default=f"{RAW}/router_results.json")
    args = ap.parse_args()
    dev = torch.device(args.device)

    out = {"margin": args.margin, "per_category": {}, "macros": {}}
    acc = {"argmax": [0, 0], "margin": [0, 0]}
    routed = {"argmax": [], "margin": [], "oracle": [],
              "pgdn": [], "draem": []}

    for cat in CATS:
        root = os.path.join(args.mvtec_root, cat)
        row = {"seeds": {}}
        for sd in SEEDS:
            real = {}
            for meth in ("pgdn", "draem"):
                with open(f"{RAW}/r24_{cat}_s{sd}_{meth}_eval.json") as f:
                    real[meth] = json.load(f)["image_auroc"]
            syn = {}
            for meth in ("pgdn", "draem"):
                model = load_model(f"r24_{cat}_s{sd}_{meth}", dev)
                syn[meth] = synthetic_val(root, model, sd, dev)
                del model
            pick_argmax = max(syn, key=syn.get)
            pick_margin = "pgdn" if syn["pgdn"] - syn["draem"] >= args.margin else "draem"
            true_best = max(real, key=real.get)
            for variant, pick in (("argmax", pick_argmax), ("margin", pick_margin)):
                acc[variant][1] += 1
                acc[variant][0] += int(pick == true_best)
                routed[variant].append(real[pick])
            routed["oracle"].append(max(real.values()))
            routed["pgdn"].append(real["pgdn"])
            routed["draem"].append(real["draem"])
            row["seeds"][sd] = {"syn": syn, "real": real, "pick_argmax": pick_argmax,
                                "pick_margin": pick_margin, "true_best": true_best}
        out["per_category"][cat] = row
        e = row["seeds"][42]
        print(f"{cat:12s} syn pgdn={e['syn']['pgdn']:.3f} draem={e['syn']['draem']:.3f} "
              f"-> pick {e['pick_margin']} (true {e['true_best']})")

    def ms(v):
        return {"mean": float(np.mean(v)), "std": float(np.std(v))}

    out["macros"] = {
        "routed_argmax": ms(routed["argmax"]),
        "routed_margin": ms(routed["margin"]),
        "oracle": ms(routed["oracle"]),
        "pgdn": ms(routed["pgdn"]),
        "draem": ms(routed["draem"]),
        "router_accuracy_argmax": f"{acc['argmax'][0]}/{acc['argmax'][1]}",
        "router_accuracy_margin": f"{acc['margin'][0]}/{acc['margin'][1]}",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["macros"], indent=2))


if __name__ == "__main__":
    main()
