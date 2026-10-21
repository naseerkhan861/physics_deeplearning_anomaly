"""Coupling-gradient conflict probe (round-2 review, mechanism section).

The reviewer's derivation: for C(p,z)=sum_i p_i(1-z_i)/sum_i p_i,
dC/dp_j = (zbar_p - z_j)/P  -- the coupling term pushes scores UP where z
exceeds the prediction-weighted mean prior response and DOWN where it
falls below it, regardless of defectiveness.  On a smooth defect (low z)
inside its mask the coupling gradient therefore OPPOSES the segmentation
gradient.

This probe measures, on trained checkpoints, the fraction of defect-mask
pixels where sign(dC/dlogit) opposes sign(dL_seg/dlogit), for standard
Perlin defects versus smooth low-derivative defects, and the mean coupling
gradient on mask pixels (negative = suppression).
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets.loaders import MVTecScrewDataset, SyntheticDefectAugmenter
from models.pgdn import PGDN
from smooth_defect_test import perlin_defect, smooth_defect

MVTEC = "/home/edison/VS_Projects/defectsGenerations/mvtec_dataset"
CATS = ("screw", "capsule", "bottle")


from defect_family_sweep import RUNS_ROOT


def load(run):
    ckpt = torch.load(f"{RUNS_ROOT}/{run}/last.pt", map_location="cpu",
                      weights_only=False)
    m = PGDN.from_config(ckpt["model_config"])
    m.load_state_dict(ckpt["model_state"])
    return m.eval()


def probe(model, imgs, family, lam=0.5):
    oppose_w, coup_sign, coup_mag, n_pix = [], [], [], 0
    for i, im in enumerate(imgs):
        rng = np.random.RandomState((42 * 10007 + i * 131) % (2 ** 31))
        d, mask = (perlin_defect(im, rng, 128) if family == "perlin"
                   else smooth_defect(im, rng, 128))
        x = torch.from_numpy(d)[None, None]
        out = model(x)
        logits = out["seg_logits"].detach().requires_grad_(True)
        if out.get("phys") is None:
            continue  # no prior -> no coupling term (base config)
        z = out["phys"].detach()
        # segmentation gradient (focal w.r.t. logits, target = mask)
        tgt = torch.from_numpy(mask)[None, None]
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * tgt + (1 - p) * (1 - tgt)
        l_seg = (0.25 * tgt + 0.75 * (1 - tgt)) * (1 - pt) ** 2 * bce
        g_seg = torch.autograd.grad(l_seg.mean(), logits, retain_graph=False)[0]
        # coupling gradient: dC/dp_j * dp/dlogit = (zbar - z_j)/P * p_j(1-p_j)
        P = p.sum()
        zbar = (p * z).sum() / (P + 1e-8)
        g_cpl = (zbar - z) * p * (1 - p) / (P + 1e-8)
        m2d = torch.from_numpy(mask).bool()
        assert m2d.shape == logits.shape[-2:], (m2d.shape, logits.shape)
        m = m2d
        if m.sum() < 8:
            continue
        gs, gc = g_seg[0, 0][m], g_cpl[0, 0][m] * lam
        # |g_seg|-weighted opposition fraction: how much of the segmentation
        # gradient mass sits on pixels where the coupling term pushes the
        # other way (scale-free, unlike the raw 1/P-scaled magnitudes).
        w = gs.abs()
        opp = (((gs > 0) != (gc > 0)).float() * w).sum() / (w.sum() + 1e-12)
        oppose_w.append(float(opp))
        coup_sign.append(float(torch.sign(gc).float().mean()))
        coup_mag.append(float(gc.mean()))
        n_pix += int(m.sum())
    if not oppose_w:
        return None
    return {"oppose_w": float(np.mean(oppose_w)),
            "sign_coup_on_mask": float(np.mean(coup_sign)),
            "mean_coup": float(np.mean(coup_mag)), "pixels": n_pix}


def main():
    res = {}
    for cat in CATS:
        ds = MVTecScrewDataset(f"{MVTEC}/{cat}", "test", image_size=128, seed=42)
        imgs = []
        for it in [i for i in ds.items if i.get("label", 0) == 0][:20]:
            im = cv2.imread(it["img"], cv2.IMREAD_GRAYSCALE)
            imgs.append(cv2.resize(im, (128, 128)).astype(np.float32) / 255.0)
        for cfg in ("cpl", "full"):
            run = f"fact_{cat}_{cfg}"
            try:
                model = load(run)
            except FileNotFoundError:
                continue
            entry = {}
            for family in ("perlin", "smooth"):
                entry[family] = probe(model, imgs, family)
            res[f"{cat}/{cfg}"] = entry
            print(f"{cat:8s} {cfg:5s} perlin: oppose_w={entry['perlin']['oppose_w']:.2f} "
                  f"sign={entry['perlin']['sign_coup_on_mask']:+.2f} | "
                  f"smooth: oppose_w={entry['smooth']['oppose_w']:.2f} "
                  f"sign={entry['smooth']['sign_coup_on_mask']:+.2f}")
            del model
    with open("results/raw/gradient_probe.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote results/raw/gradient_probe.json")


if __name__ == "__main__":
    main()
