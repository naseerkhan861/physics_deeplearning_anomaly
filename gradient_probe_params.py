"""Parameter-level gradient-conflict probe (round-3 review Step 10).

The logit-level probe shows sign opposition; this probe measures the
actual influence of the coupling term on the shared parameters:

  g_seg = grad_theta L_seg          (segmentation objective)
  g_C   = grad_theta (lambda_C * C) (loss-weighted coupling term)

per top-level parameter block, on matched Perlin and smooth defect
batches: the norm ratio ||lambda_C g_C|| / ||g_seg|| and cosine
alignment cos(g_seg, g_C) < 0 = conflict.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch

from datasets.loaders import MVTecScrewDataset
from defect_family_sweep import MVTEC, load
from losses import focal_loss
from smooth_defect_test import perlin_defect, smooth_defect

RUNS = [("screw", "cpl"), ("screw", "incpl"), ("screw", "full"),
        ("capsule", "cpl"), ("capsule", "incpl"),
        ("bottle", "cpl"), ("bottle", "incpl")]
LAM = 0.5
N_BATCH = 2
BATCH = 8


def batches(cat):
    ds = MVTecScrewDataset(os.path.join(MVTEC, cat), "test",
                           image_size=128, seed=42)
    good = [it for it in ds.items if it.get("label", 0) == 0][:N_BATCH * BATCH]
    imgs = []
    for it in good:
        im = cv2.imread(it["img"], cv2.IMREAD_GRAYSCALE)
        imgs.append(cv2.resize(im, (128, 128)).astype(np.float32) / 255.0)
    out = {"perlin": [], "smooth": []}
    for b in range(N_BATCH):
        for family in ("perlin", "smooth"):
            xs, ms = [], []
            for i in range(BATCH):
                idx = b * BATCH + i
                rng = np.random.RandomState((42 * 10007 + idx * 131) % (2 ** 31))
                d, m = (perlin_defect(imgs[idx], rng, 128) if family == "perlin"
                        else smooth_defect(imgs[idx], rng, 128))
                xs.append(d)
                ms.append(m)
            out[family].append((np.stack(xs), np.stack(ms)))
    return out


def probe(model, data):
    res = {}
    blocks = sorted({n.split(".")[0] for n, _ in model.named_parameters()})
    for family in ("perlin", "smooth"):
        fam = {"norm_ratio": [], "cosine": []}
        fam_blocks = {}
        for xs, ms in data[family]:
            x = torch.from_numpy(xs[:, None])
            m = torch.from_numpy(ms[:, None].astype(np.float32))
            model.zero_grad(set_to_none=True)
            out = model(x)
            l_seg = focal_loss(out["seg_logits"], m, 0.25, 2.0)
            p = torch.sigmoid(out["seg_logits"])
            l_c = LAM * ((p * (1.0 - out["phys"].detach())).sum()
                         / (p.sum() + 1e-8))
            l_seg.backward(retain_graph=True)
            g_seg = {n: p_.grad.clone() for n, p_ in model.named_parameters()
                     if p_.grad is not None}
            model.zero_grad(set_to_none=True)
            l_c.backward()
            # global flat vectors
            v_seg = torch.cat([v.flatten() for v in g_seg.values()])
            v_c = torch.cat([(p_.grad.flatten() if p_.grad is not None
                              else torch.zeros_like(p_).flatten())
                             for _, p_ in model.named_parameters()])
            fam["norm_ratio"].append(float(v_c.norm() / (v_seg.norm() + 1e-12)))
            fam["cosine"].append(float(torch.nn.functional.cosine_similarity(
                v_c[None], v_seg[None])))
            for b in blocks:
                fs = torch.cat([v.flatten() for n, v in g_seg.items()
                                if n.startswith(b)]) if any(
                    n.startswith(b) for n in g_seg) else None
                fc = torch.cat([p_.grad.flatten() for n, p_ in
                                model.named_parameters()
                                if n.startswith(b) and p_.grad is not None]) \
                    if any(n.startswith(b) for n, _ in model.named_parameters()) else None
                if fs is not None and fc is not None and fs.numel():
                    d = fam_blocks.setdefault(b, {"norm_ratio": [], "cosine": []})
                    d["norm_ratio"].append(float(fc.norm() / (fs.norm() + 1e-12)))
                    d["cosine"].append(float(torch.nn.functional.cosine_similarity(
                        fc[None], fs[None])))
        res[family] = {
            "norm_ratio": float(np.mean(fam["norm_ratio"])),
            "cosine": float(np.mean(fam["cosine"])),
            "blocks": {b: {k: float(np.mean(v[k])) for k in ("norm_ratio", "cosine")}
                       for b, v in fam_blocks.items()},
        }
    return res


def main():
    out = {"definition": "g_seg=grad L_seg, g_C=grad(lambda_C*C) on shared "
                         "parameters; norm ratio and cosine per family, mean "
                         f"over {N_BATCH} batches x {BATCH} images; lambda_C={LAM}",
           "runs": {}}
    for cat, arm in RUNS:
        run = f"fact_{cat}_{arm}"
        if not os.path.isfile(f"runs/{run}/last.pt"):
            print("skip:", run)
            continue
        model = load(run)
        out["runs"][f"{cat}/{arm}"] = probe(model, batches(cat))
        print(run, json.dumps({f: {k: round(v[k], 3) for k in ("norm_ratio", "cosine")}
                               for f, v in out["runs"][f"{cat}/{arm}"].items()}))
        del model
    with open("results/raw/grad_params.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/raw/grad_params.json")


if __name__ == "__main__":
    main()
