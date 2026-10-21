"""Figure B data: spatial coupling pressure and a local parameter probe.

For the two coupled arms (cpl, incpl), seeds 42/0/1, and all recorded
real defects of screw/capsule/bottle, plus paired synthetic Perlin /
smooth defects on fixed screw backgrounds:
  - logit-space gradients of the EXACT implemented weighted coupling
    loss and segmentation loss -> coupling pressure u = -dlambda*C/dl,
    defect-pixel opposition Omega, suppression fraction, prior stats;
  - an isolated coupling-direction parameter step on a temporary
    in-memory displacement (original checkpoint restored after every
    probe; weights verified) with observed vs linear loss checks;
  - controls (zero, opposite, random, r/2) on the declared development
    images, and a radius pilot used to freeze the main radius.

GT masks are used only to evaluate frozen checkpoints diagnostically.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/home/edison/agents_project/pgdn_project")

import csv
import json
import os
from copy import deepcopy

import cv2
import numpy as np
import torch

from datasets.loaders import MVTecScrewDataset
from defect_family_sweep import MVTEC, load, smooth_defect
from losses import focal_loss
from smooth_defect_test import perlin_defect

OUT = "results/defect_analysis_v3"
CATS = ("screw", "capsule", "bottle")
LAM = 0.5


def coupling(p, z):
    return (p * (1.0 - z)).sum() / (p.sum() + 1e-8)


def analysis(model, x, mask, dev):
    """Logit grads + stats for one image; model stays untouched."""
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = model(x)
        logits = out["seg_logits"]
        z = out["phys"].detach()
        p = torch.sigmoid(logits)
        c = LAM * coupling(p, z)
        s = focal_loss(logits, mask, 0.25, 2.0)
        gc_l = torch.autograd.grad(c, logits, retain_graph=True)[0].detach()
        gs_l = torch.autograd.grad(s, logits)[0].detach()
        p0 = p.detach()
    m = mask[0, 0] > 0.5
    gs_m, gc_m = gs_l[0, 0][m], gc_l[0, 0][m]
    den = gs_m.abs().sum()
    # paper definition: fraction of SEGMENTATION-gradient mass on opposing
    # pixels (round-19 C2); unweighted by coupling magnitude, bounded in [0,1]
    opposed = (gs_m * gc_m < 0).float()
    omega = float((gs_m.abs() * opposed).sum() / den) if den > 0 else float("nan")
    pflat, zflat = p0[0, 0], z[0, 0]
    P = pflat.sum()
    zbar = float((pflat * zflat).sum() / (P + 1e-8))
    return {"omega": omega, "supp_frac": float((gc_m > 0).float().mean()) if den > 0 else float("nan"),
            "mean_z_mask": float(zflat[m].mean()), "zbar_p": zbar,
            "pred_mass": float(P.mean()),
            "grad_c_logit": gc_l[0, 0].cpu().numpy(),
            "grad_seg_logit": gs_l[0, 0].cpu().numpy()}


def param_probe(model, x, mask, dev, radius, direction="coupling", seed=0):
    """Displace along a unit direction scaled to radius*||theta||; restore."""
    params = [p for p in model.parameters() if p.requires_grad]
    sd_backup = deepcopy(model.state_dict())
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = model(x)
        logits = out["seg_logits"]
        z = out["phys"].detach()
        p = torch.sigmoid(logits)
        if direction == "seg":
            scalar = focal_loss(logits, mask, 0.25, 2.0)
        else:
            scalar = LAM * coupling(p, z)
        g = torch.autograd.grad(scalar, params, allow_unused=True)
    g = [torch.zeros_like(p_) if gi is None else gi.detach()
         for p_, gi in zip(params, g)]
    if direction == "random":
        rng = np.random.RandomState(seed)
        g = [torch.from_numpy(rng.standard_normal(gi.shape).astype(np.float32)).to(gi.device)
             for gi in g]
    nrm = float(sum(gi.double().square().sum() for gi in g).sqrt())
    nrm_t = float(sum(p_.detach().double().square().sum() for p_ in params).sqrt())
    if nrm == 0:
        model.load_state_dict(sd_backup)
        return None
    sign = +1 if direction in ("seg", "random") else -1  # descent for coupling
    eta = sign * radius * nrm_t / nrm
    with torch.no_grad():
        for p_, gi in zip(params, g):
            p_.add_(gi, alpha=eta)
    with torch.enable_grad():
        out1 = model(x)
        logits1 = out1["seg_logits"]
        p1 = torch.sigmoid(logits1).detach()
        s1 = focal_loss(logits1, mask, 0.25, 2.0)
    # linear delta needs seg param grad at theta
    model.zero_grad(set_to_none=True)
    model.load_state_dict(sd_backup)
    with torch.enable_grad():
        out2 = model(x)
        s0 = focal_loss(out2["seg_logits"], mask, 0.25, 2.0)
        gs = torch.autograd.grad(s0, params, allow_unused=True)
    gs = [torch.zeros_like(p_) if gi is None else gi.detach()
          for p_, gi in zip(params, gs)]
    lin = float(sum((a.double() * (eta * gi.double())).sum()
                    for a, gi in zip(gs, g)))
    m = mask[0, 0] > 0.5
    d = (p1 - torch.sigmoid(out2["seg_logits"]).detach())
    model.load_state_dict(sd_backup)  # guarantee restore
    return {"delta_mask_mean_p": float(d[0, 0][m].mean()) if m.any() else float("nan"),
            "delta_bg_mean_p": float(d[0, 0][~m].mean()) if (~m).any() else float("nan"),
            "delta_seg_loss": float(s1.detach() - s0.detach()),
            "linear_delta_seg_loss": lin, "radius": radius, "eta": eta}


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f"{OUT}/maps", exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, synthetic = [], {}
    dev_bank = json.load(open(f"{OUT}/protocol.json"))["selected_examples"]

    # radius pilot on the development bank (screw/cpl/seed42)
    pilot = {}
    ds0 = MVTecScrewDataset(os.path.join(MVTEC, "screw"), "test", image_size=128, seed=42)
    idx0 = {it["img"]: i for i, it in enumerate(ds0.items)}
    item0 = ds0[idx0[dev_bank["screw"]]]
    img0 = item0["image"].numpy()
    img0 = img0[0] if img0.ndim == 3 else img0
    x0 = torch.from_numpy(img0[None, None].astype(np.float32)).to(dev)
    m0 = item0["mask"][None].to(dev)
    mp = load("fact_screw_cpl").to(dev)
    for r in (1e-7, 3e-7, 1e-6):
        res = param_probe(mp, x0, m0, dev, r)
        ratio = res["delta_seg_loss"] / res["linear_delta_seg_loss"] if abs(res["linear_delta_seg_loss"]) > 1e-12 else float("nan")
        half = param_probe(mp, x0, m0, dev, r / 2)
        ratio_half = half["delta_seg_loss"] / half["linear_delta_seg_loss"] if abs(half["linear_delta_seg_loss"]) > 1e-12 else float("nan")
        pilot[r] = {"obs_over_linear": round(ratio, 3),
                    "halved_obs_over_linear": round(ratio_half, 3)}
        print("pilot r", r, pilot[r], flush=True)
    RADIUS = 1e-6
    json.dump({"pilot": pilot, "frozen_radius": RADIUS},
              open(f"{OUT}/probe_protocol.json", "w"), indent=2)
    mp = mp.cpu()

    for cat in CATS:
        ds = MVTecScrewDataset(os.path.join(MVTEC, cat), "test", image_size=128, seed=42)
        idx = {it["img"]: i for i, it in enumerate(ds.items)}
        for seed in (42, 0, 1):
            pre = "fact" if seed == 42 else "f2"
            sfx = "" if seed == 42 else f"_s{seed}"
            for arm in ("cpl", "incpl"):
                model = load(f"{pre}_{cat}_{arm}{sfx}").to(dev)
                for it in ds.items:
                    if it.get("label", 0) != 1:
                        continue
                    item = ds[idx[it["img"]]]
                    img = item["image"].numpy()
                    img = img[0] if img.ndim == 3 else img
                    x = torch.from_numpy(img[None, None].astype(np.float32)).to(dev)
                    mk = item["mask"][None].to(dev)
                    if not (mk[0, 0] > 0.5).any():
                        continue
                    a = analysis(model, x, mk, dev)
                    pr = param_probe(model, x, mk, dev, RADIUS)
                    rows.append([cat, it["img"].split("/")[-2],
                                 os.path.basename(it["img"]), seed, arm,
                                 round(a["omega"], 4), round(a["supp_frac"], 4),
                                 round(a["mean_z_mask"], 4), round(a["zbar_p"], 4),
                                 round(pr["delta_mask_mean_p"], 6),
                                 round(pr["delta_bg_mean_p"], 6),
                                 round(pr["delta_seg_loss"], 8),
                                 round(pr["linear_delta_seg_loss"], 8)])
                model = model.cpu()
                print(cat, seed, arm, "done", flush=True)

    # synthetic pairs on fixed screw backgrounds (seed 42, cpl arm)
    model = load("fact_screw_cpl").to(dev)
    good = [it for it in ds0.items if it.get("label", 0) == 0][:3]
    for i, it in enumerate(good):
        im = cv2.imread(it["img"], cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (128, 128)).astype(np.float32) / 255.0
        for fam in ("perlin", "smooth"):
            rng = np.random.RandomState((42 * 10007 + i * 131) % (2 ** 31))
            d, msk = (perlin_defect(im, rng, 128) if fam == "perlin"
                      else smooth_defect(im, rng, 128))
            x = torch.from_numpy(d[None, None].astype(np.float32)).to(dev)
            mk = torch.from_numpy(msk[None, None].astype(np.float32)).to(dev)
            a = analysis(model, x, mk, dev)
            pr = param_probe(model, x, mk, dev, RADIUS)
            synthetic[f"bg{i}_{fam}"] = {
                "image": d, "clean": im, "mask": msk, "z": None,
                "pressure": -a["grad_c_logit"],
                "p0_minus_p1": None, "omega": a["omega"],
                "delta_mask_mean_p": pr["delta_mask_mean_p"]}
            with torch.no_grad():
                out = model(x)
                synthetic[f"bg{i}_{fam}"]["z"] = out["phys"][0, 0].cpu().numpy()
            # prediction change map for the atlas
            params = [p for p in model.parameters() if p.requires_grad]
            backup = deepcopy(model.state_dict())
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                outg = model(x)
                c = LAM * coupling(torch.sigmoid(outg["seg_logits"]),
                                   outg["phys"].detach())
                g = torch.autograd.grad(c, params, allow_unused=True)
            g = [torch.zeros_like(p_) if gi is None else gi.detach()
                 for p_, gi in zip(params, g)]
            nrm = float(sum(gi.double().square().sum() for gi in g).sqrt())
            nrm_t = float(sum(p_.detach().double().square().sum() for p_ in params).sqrt())
            with torch.no_grad():
                for p_, gi in zip(params, g):
                    p_.add_(gi, alpha=-(RADIUS * nrm_t / nrm))
                p1 = torch.sigmoid(model(x)["seg_logits"])[0, 0].cpu().numpy()
            model.load_state_dict(backup)
            with torch.no_grad():
                p0 = torch.sigmoid(model(x)["seg_logits"])[0, 0].cpu().numpy()
            synthetic[f"bg{i}_{fam}"]["p0_minus_p1"] = p0 - p1
    model = model.cpu()
    np.savez_compressed(f"{OUT}/maps/synthetic_probes.npz",
                        **{f"{k}__{f}": v for k, d in synthetic.items()
                           for f, v in d.items() if v is not None})

    # controls on the dev image (screw/cpl/seed42)
    model = load("fact_screw_cpl").to(dev)
    ctrl = {"zero": param_probe(model, x0, m0, dev, 0.0),
            "coupling": param_probe(model, x0, m0, dev, RADIUS),
            "opposite": param_probe(model, x0, m0, dev, RADIUS,
                                    direction="coupling_opposite"),
            "random0": param_probe(model, x0, m0, dev, RADIUS,
                                   direction="random", seed=0),
            "half": param_probe(model, x0, m0, dev, RADIUS / 2)}
    # opposite via sign trick: reuse coupling with negative radius
    ctrl["opposite"] = param_probe(model, x0, m0, dev, -RADIUS)
    model = model.cpu()
    json.dump({k: {kk: round(vv, 8) if isinstance(vv, float) else vv
                   for kk, vv in v.items()} if v else None
               for k, v in ctrl.items()},
              open(f"{OUT}/probe_controls.json", "w"), indent=2)

    with open(f"{OUT}/spatial_probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "defect_type", "image", "seed", "arm",
                    "omega", "supp_frac", "mean_z_mask", "zbar_p",
                    "delta_mask_mean_p", "delta_bg_mean_p",
                    "delta_seg_loss", "linear_delta_seg_loss"])
        w.writerows(rows)
    json.dump({k: {"omega": v["omega"], "delta_mask_mean_p": v["delta_mask_mean_p"]}
               for k, v in synthetic.items()},
              open(f"{OUT}/synthetic_summary.json", "w"), indent=2)
    print("wrote", len(rows), "probe rows; controls:",
          json.dumps({k: v and round(v["delta_mask_mean_p"], 6) for k, v in ctrl.items()}))


if __name__ == "__main__":
    main()
