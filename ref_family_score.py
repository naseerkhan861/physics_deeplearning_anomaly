"""Score completed stronger-recipe fits on the paired Perlin/smooth assay.

Usage: from the experiment-project root, `python ref_family_score.py`
(after setting PGDN_RUNS_ROOT if checkpoints live elsewhere); run
`ref_protocol_analyze.py` afterwards for the frozen contrasts.

For every runs/ref_<cat>_<arm>_s<seed> checkpoint that has a real-eval
JSON but no family record yet, score the matched 30-background
Perlin/smooth families (identical protocol to the lightweight assay)
and write results/raw/ref_<cat>_<arm>_s<seed>_family.json.  The K
contrasts are then derived by the analysis in
ref_protocol_analyze.py (K(i,s) = 100*[F(i,C=1,s) - F(i,C=0,s)],
F = smooth - Perlin, percentage points).
"""

import sys
sys.path.insert(0, "/home/edison/agents_project/pgdn_project")
import glob
import json
import os

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from datasets.loaders import MVTecScrewDataset
from defect_family_sweep import MVTEC, load, smooth_defect
from smooth_defect_test import perlin_defect

def main():
    done = 0
    from defect_family_sweep import RUNS_ROOT
    for ck in sorted(glob.glob(f"{RUNS_ROOT}/ref_*_s*/last.pt")):
        run = os.path.basename(os.path.dirname(ck))
        fam_json = f"results/raw/{run}_family.json"
        ev_json = f"results/raw/{run}_eval.json"
        if not os.path.isfile(ev_json) or os.path.isfile(fam_json):
            continue
        parts = run.split("_")           # ref <cat> <arm> s<seed>
        cat, seed = parts[1], parts[3]
        rec = score_run(cat, run)
        if rec:
            json.dump(rec, open(fam_json, "w"), indent=2)
            print("scored", run, {k: round(v, 3) for k, v in rec.items()
                                  if isinstance(v, float)})
            done += 1
    print("family records written:", done)


def score_run(cat, run):
    ds = MVTecScrewDataset(os.path.join(MVTEC, cat), "test",
                           image_size=128, seed=42)
    good = [it for it in ds.items if it.get("label", 0) == 0][:30]
    imgs = []
    for it in good:
        im = cv2.imread(it["img"], cv2.IMREAD_GRAYSCALE)
        imgs.append(cv2.resize(im, (128, 128)).astype(np.float32) / 255.0)
    model = load(run)
    res = {"run": run, "n_backgrounds": len(imgs)}
    for fam in ("perlin", "smooth"):
        labels, scores = [], []
        for i, im in enumerate(imgs):
            rng = np.random.RandomState((42 * 10007 + i * 131) % (2 ** 31))
            d, _ = (perlin_defect(im, rng, 128) if fam == "perlin"
                    else smooth_defect(im, rng, 128))
            for image, lab in ((im, 0), (d, 1)):
                x = torch.from_numpy(image[None, None].astype(np.float32))
                with torch.no_grad():
                    m = model.anomaly_map(x)[0, 0].numpy()
                scores.append(float(cv2.GaussianBlur(
                    m, (0, 0), sigmaX=4.0).max()))
                labels.append(lab)
        res[fam] = float(roc_auc_score(labels, scores))
    return res


if __name__ == "__main__":
    main()
