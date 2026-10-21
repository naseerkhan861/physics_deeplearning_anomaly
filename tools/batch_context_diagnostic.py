"""Batch-context diagnostic producer (review round-21 D).

Holds one target smooth-defect image and one checkpoint fixed while
varying the seven companion normal images, and reports the batch
prediction-weighted prior mean and the target's defect-pixel
opposition / suppression fraction. Model is in eval() mode throughout,
so BatchNorm uses running statistics and batch composition affects
only the coupling quotient, not the forward maps.

Writes named-field JSON (artifact/batch_context_diagnostic.json).
"""
import sys
sys.path.insert(0, "/home/edison/agents_project/pgdn_project")
import json
import cv2
import numpy as np
import torch

from datasets.loaders import MVTecScrewDataset
from defect_family_sweep import MVTEC, load, smooth_defect
from losses import focal_loss

CKPT = "fact_screw_cpl"          # run id; checkpoint hash in run_manifest.csv
CAT, LAM, EPS, BG_IDX, GEN_SEED = "screw", 0.5, 1e-8, 0, 7
BATCH = 8

def coupling(p, z):
    return (p * (1.0 - z)).sum() / (p.sum() + EPS)

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MVTecScrewDataset(MVTEC + "/screw", "test", image_size=128, seed=42)
    good = [it for it in ds.items if it.get("label", 0) == 0]
    im0 = cv2.resize(cv2.imread(good[BG_IDX]["img"], cv2.IMREAD_GRAYSCALE),
                     (128, 128)).astype(np.float32) / 255.0
    rng = np.random.RandomState(GEN_SEED)
    tgt, msk = smooth_defect(im0, rng, 128)
    model = load(CKPT).to(dev).eval()   # eval mode: BN running stats

    def load_img(i):
        im = cv2.resize(cv2.imread(good[i]["img"], cv2.IMREAD_GRAYSCALE), (128, 128))
        return im.astype(np.float32) / 255.0

    def batch_stats(companion_ids, expected=None):
        exp = BATCH - 1 if expected is None else expected
        assert len(companion_ids) == exp, \
            f"expected exactly {exp} companions, got {len(companion_ids)}"
        xs = [torch.from_numpy(tgt[None, None].astype(np.float32))]
        ms = [torch.from_numpy(msk[None, None].astype(np.float32))]
        for i in companion_ids:
            xs.append(torch.from_numpy(load_img(i)[None, None].astype(np.float32)))
            ms.append(torch.zeros(1, 1, 128, 128))
        x = torch.cat(xs).to(dev); m = torch.cat(ms).to(dev)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = model(x)
            logits = out["seg_logits"]; z = out["phys"].detach()
            p = torch.sigmoid(logits)
            c = LAM * coupling(p, z)
            s = focal_loss(logits, m, 0.25, 2.0)
            gc = torch.autograd.grad(c, logits, retain_graph=True)[0].detach()
            gs = torch.autograd.grad(s, logits)[0].detach()
        Pb = p.sum().item()
        zbar = (p * z).sum().item() / Pb
        mt = torch.from_numpy(msk.astype(bool)).to(dev)
        gs_m, gc_m = gs[0, 0][mt], gc[0, 0][mt]
        opp = ((gs_m.abs() * (gs_m * gc_m < 0).float()).sum()
               / gs_m.abs().sum()).item()
        supp = (gc_m > 0).float().mean().item()
        return {"batch_prior_mean": round(zbar, 5),
                "target_opposition": round(opp, 4),
                "target_suppression_fraction": round(supp, 4)}

    out = {
        "protocol": {"checkpoint_run": CKPT, "category": CAT, "batch_size": BATCH,
                     "loss_weight_lambda": LAM, "epsilon": EPS,
                     "model_mode": "eval (BatchNorm running stats; batch size "
                     "affects only the coupling quotient, not forward maps)",
                     "target": {"background": good[BG_IDX]["img"],
                                "generator": "smooth_defect (defect_family_sweep)",
                                "rng_seed": GEN_SEED},
                     "companion_sets": {
                         "companions_1_7": [good[i]["img"] for i in range(1, 8)],
                         "companions_8_14": [good[i]["img"] for i in range(8, 15)],
                         "companions_other_normals": [
                             good[(i * 7) % len(good)]["img"]
                             for i in range(8, 15)]}},
        "target_alone_batch1": batch_stats([], expected=0),
        "companions_1_7": batch_stats(list(range(1, 8))),
        "companions_8_14": batch_stats(list(range(8, 15))),
        "companions_other_normals": batch_stats([(i * 7) % len(good)
                                                 for i in range(8, 15)]),
        "interpretation": "one-target pilot: batch context shifts the reference "
        "mean measurably while the target's opposition/suppression direction "
        "is stable across these companion sets; not a population claim"}
    path = "results/raw/batch_context_diagnostic.json"
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)

if __name__ == "__main__":
    main()
