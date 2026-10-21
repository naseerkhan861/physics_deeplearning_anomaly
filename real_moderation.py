"""Real-defect matched moderation analysis (round-4 review Step 3).

For each category and training seed, computes the matched coupling
effect on real MVTec defects,

  E_g(I) = AUROC(arm with C on, group g) - AUROC(arm with C off, group g)
  M(I)   = E_weak(I) - E_strong(I),

with reconstruction disabled (arms base/in/cpl/incpl).  Groups are the
weak/strong prior-response split of the real defects, fixed at the
seed-42 z median and reused for every arm and seed.  Uncertainty uses a
paired image-cluster bootstrap: defect and defect-free image indices are
resampled jointly and identically across all arms and groups.  Per-image
scores and group assignments are exported.
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
SEEDS = (42, 0, 1)
N_BOOT = 300


@torch.no_grad()
def zmap(model, img):
    x = torch.from_numpy(img[None, None].astype(np.float32))
    return model(x)["phys"][0, 0].numpy()


def img_of(ds, idx_of, path):
    item = ds[idx_of[path]]
    img = item["image"].numpy()
    return img[0] if img.ndim == 3 else img


def run_cat(cat):
    ds = MVTecScrewDataset(os.path.join(MVTEC, cat), "test",
                           image_size=128, seed=42)
    idx_of = {it["img"]: i for i, it in enumerate(ds.items)}
    good = [it["img"] for it in ds.items if it.get("label", 0) == 0]
    bad = [it["img"] for it in ds.items if it.get("label", 0) == 1]
    zf = load(f"fact_{cat}_full")
    z_in = {}
    for p in bad:
        item = ds[idx_of[p]]
        img = item["image"].numpy()
        if img.ndim == 3:
            img = img[0]
        m = item["mask"][0].numpy() > 0.5
        if m.sum() >= 4:
            z_in[p] = float(zmap(zf, img)[m].mean())
    del zf
    thr = float(np.median(list(z_in.values())))
    grp = {p: ("weak" if z < thr else "strong") for p, z in z_in.items()}
    bad = sorted(grp)  # only defects with usable masks

    S = {}  # S[arm][seed] = dict(defects=..., good=...)
    for seed in SEEDS:
        pre = "fact" if seed == 42 else "f2"
        sfx = "" if seed == 42 else f"_s{seed}"
        for arm in ARMS:
            model = load(f"{pre}_{cat}_{arm}{sfx}")
            S.setdefault(arm, {})[seed] = {
                "defects": [score(model, img_of(ds, idx_of, p)) for p in bad],
                "good": [score(model, img_of(ds, idx_of, g)) for g in good],
            }
            del model
    gsel = np.array([grp[p] == "weak" for p in bad])

    def auroc(s, weak_mask):
        y = np.array([0] * len(s["good"]) + [1] * int(weak_mask.sum()))
        sc = np.array(s["good"] + [d for d, w in zip(s["defects"], weak_mask) if w])
        return roc_auc_score(y, sc)

    def effects(seed, weak_mask_sel=gsel):
        e = {}
        for I, (on, off) in {"off": ("cpl", "base"), "on": ("incpl", "in")}.items():
            e[I] = {g: 100 * (auroc(S[on][seed], gsel == (g == "weak"))
                              - auroc(S[off][seed], gsel == (g == "weak")))
                    for g in ("weak", "strong")}
            e[I]["M"] = e[I]["weak"] - e[I]["strong"]
        return e

    out = {"n_good": len(good), "n_defects": len(bad),
           "n_weak": int(gsel.sum()), "z_median": thr,
           "z_in": {p: z_in[p] for p in bad},
           "effects": {str(s): effects(s) for s in SEEDS},
           "per_image": {arm: {str(s): {"defects": S[arm][s]["defects"],
                                        "good": S[arm][s]["good"]}
                               for s in SEEDS} for arm in ARMS},
           "groups": grp}

    # paired bootstrap: same resampled defect/good indices for all arms+groups
    rng = np.random.RandomState(11)
    nd, ng = len(bad), len(good)
    boot = {str(s): {I: [] for I in ("off", "on")} for s in SEEDS}
    weak_idx = np.where(gsel)[0]
    strong_idx = np.where(~gsel)[0]
    for b in range(N_BOOT):
        dw = rng.choice(weak_idx, len(weak_idx), replace=True)
        ds_ = rng.choice(strong_idx, len(strong_idx), replace=True)
        gg = rng.randint(0, ng, ng)
        wm = np.zeros(nd, bool)
        wm[dw] = True
        wm2 = np.zeros(nd, bool)
        wm2[ds_] = True
        for s in SEEDS:
            for I, (on, off) in {"off": ("cpl", "base"), "on": ("incpl", "in")}.items():
                def ab(arm, mask):
                    sc = np.concatenate([np.array(S[arm][s]["good"])[gg],
                                         np.array(S[arm][s]["defects"])[mask]])
                    y = np.concatenate([np.zeros(len(gg)), np.ones(int(mask.sum()))])
                    return roc_auc_score(y, sc)
                ew = ab(on, wm) - ab(off, wm)
                es = ab(on, wm2) - ab(off, wm2)
                boot[str(s)][I].append(100 * (ew - es))
    out["bootstrap_M_ci95"] = {
        str(s): {I: [round(float(np.percentile(boot[str(s)][I], 2.5)), 1),
                     round(float(np.percentile(boot[str(s)][I], 97.5)), 1)]
                  for I in ("off", "on")} for s in SEEDS}
    return out


def main():
    res = {"protocol": "matched coupling effect on real defects (reconstruction "
                       "off): E_g(I)=AUROC(C-on,g)-AUROC(C-off,g); M=E_weak-"
                       "E_strong; groups fixed at the seed-42 z median; paired "
                       f"image-cluster bootstrap, {N_BOOT} replicates; grouping "
                       "held fixed within the bootstrap",
           "categories": {}}
    for cat in CATS:
        res["categories"][cat] = run_cat(cat)
        e = res["categories"][cat]["effects"]
        print(cat, json.dumps({s: e[s] for s in e}, default=str)[:400], flush=True)
    with open("results/raw/real_moderation.json", "w") as f:
        json.dump(res, f)
    print("wrote results/raw/real_moderation.json")


if __name__ == "__main__":
    main()
