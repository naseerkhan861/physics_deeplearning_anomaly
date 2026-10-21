"""Evaluation for PGDN checkpoints or the PatchCore-lite baseline.

Computes image-level AUROC/AP and pixel-level AUROC (pixels subsampled to at
most ``--max-pixels``, seeded) on a test split, plus a per-severity breakdown
for CONSTRUCT (from ``splits.csv`` metadata carried by the dataset).

Examples:
    python eval.py --dataset construct --data-root /mnt/agents/work/data/construct \
        --class-name brushed_steel --checkpoint runs/pgdn_bs/last.pt
    python eval.py --dataset mvtec --data-root /mnt/agents/work/data/mvtec_ad/screw \
        --method patchcore --max-train-images 60
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from datasets.loaders import ConstructDataset, MVTecScrewDataset
from models.patchcore_lite import PatchCoreLite
from models.pgdn import PGDN


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PGDN or PatchCore-lite")
    p.add_argument("--method", choices=["pgdn", "patchcore"], default=None,
                   help="default: pgdn if --checkpoint is given, else patchcore")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--dataset", choices=["mvtec", "construct"], required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--class-name", default=None)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--channels", type=int, default=1, choices=[1, 3])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu",
                   help="torch device for PGDN scoring (e.g. cpu / cuda)")
    p.add_argument("--max-pixels", type=int, default=1_000_000,
                   help="cap on pixels sampled for pixel-level AUROC")
    p.add_argument("--max-test-images", type=int, default=None)
    p.add_argument("--out", default=None, help="results JSON path "
                   "(default: <run_dir>/results.json or ./results_<method>.json)")
    # patchcore options
    p.add_argument("--coreset-size", type=int, default=2000)
    p.add_argument("--max-train-images", type=int, default=None,
                   help="patchcore: cap defect-free fit images")
    p.add_argument("--frac", type=float, default=1.0,
                   help="patchcore: defect-free data fraction for fitting")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Scoring backends
# ---------------------------------------------------------------------------

class PGDNScorer:
    def __init__(self, checkpoint: str, image_size: int = 128, device: str = "cpu"):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = PGDN.from_config(ckpt["model_config"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.name = f"pgdn:{os.path.basename(os.path.dirname(checkpoint)) or checkpoint}"

    @torch.no_grad()
    def score(self, img: np.ndarray):
        if img.ndim == 2:
            x = torch.from_numpy(img[None, None].astype(np.float32))
        else:  # (C, H, W)
            x = torch.from_numpy(img[None].astype(np.float32))
        x = x.to(self.device)
        m = self.model.anomaly_map(x)[0, 0].cpu().numpy()
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=4.0)
        return float(m.max()), m


class PatchCoreScorer:
    def __init__(self, pc: PatchCoreLite):
        self.pc = pc
        self.name = f"patchcore_lite:{pc.extractor_name}"

    def score(self, img: np.ndarray):
        return self.pc.score(img)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _safe_auroc(y, s) -> Optional[float]:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def _safe_ap(y, s) -> Optional[float]:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, s))


def pixel_auroc(pix_labels: np.ndarray, pix_scores: np.ndarray,
                max_pixels: int, seed: int) -> Optional[float]:
    n = pix_labels.shape[0]
    if n > max_pixels:
        rng = np.random.RandomState(seed)
        keep = rng.choice(n, max_pixels, replace=False)
        pix_labels, pix_scores = pix_labels[keep], pix_scores[keep]
    return _safe_auroc(pix_labels, pix_scores)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args) -> Dict:
    method = args.method or ("pgdn" if args.checkpoint else "patchcore")

    test_kw = dict(image_size=args.image_size, seed=args.seed, channels=args.channels)
    if args.dataset == "mvtec":
        test_ds = MVTecScrewDataset(args.data_root, "test", **test_kw)
        train_good_dir = os.path.join(args.data_root, "train", "good")
    else:
        test_ds = ConstructDataset(args.data_root, "test",
                                   class_name=args.class_name, **test_kw)
        train_good_dir = None  # per-class dirs handled below
    if args.max_test_images:
        # stratified cap so quick evals keep both classes (and severities)
        good = [it for it in test_ds.items if it.get("label", 0) == 0]
        bad = [it for it in test_ds.items if it.get("label", 0) == 1]
        n_good = min(len(good), args.max_test_images // 2)
        n_bad = min(len(bad), args.max_test_images - n_good)
        test_ds.items = good[:n_good] + bad[:n_bad]

    if method == "pgdn":
        assert args.checkpoint, "--checkpoint required for method=pgdn"
        scorer = PGDNScorer(args.checkpoint, args.image_size, device=args.device)
    else:
        pc = PatchCoreLite(coreset_size=args.coreset_size,
                           image_size=args.image_size, seed=args.seed)
        train_kw = dict(image_size=args.image_size, frac=args.frac,
                        seed=args.seed, augment=False, channels=args.channels)
        if args.dataset == "mvtec":
            fit_ds = MVTecScrewDataset(args.data_root, "train", **train_kw)
        else:
            fit_ds = ConstructDataset(args.data_root, "train",
                                      class_name=args.class_name, **train_kw)
        if args.max_train_images:
            fit_ds.items = fit_ds.items[: args.max_train_images]
        print(f"[eval] fitting patchcore on {len(fit_ds)} images "
              f"({pc.extractor_name}) ...")
        flag = cv2.IMREAD_COLOR if args.channels == 3 else cv2.IMREAD_GRAYSCALE
        imgs = [cv2.resize(cv2.imread(it["img"], flag),
                           (args.image_size, args.image_size))
                for it in fit_ds.items]
        pc.fit(imgs)
        scorer = PatchCoreScorer(pc)

    print(f"[eval] scoring {len(test_ds)} test images with {scorer.name}")
    labels, scores, severities = [], [], []
    pix_labels: List[np.ndarray] = []
    pix_scores: List[np.ndarray] = []
    # per-severity pixel storage (construct)
    sev_pix: Dict[str, List] = {}

    for i in range(len(test_ds)):
        item = test_ds[i]
        img = item["image"].numpy()
        s, m = scorer.score(img)
        labels.append(item["label"])
        scores.append(s)
        sev = item.get("severity") or ("good" if item["label"] == 0 else "unknown")
        severities.append(sev)
        pix_labels.append(item["mask"][0].numpy().ravel())
        pix_scores.append(m.ravel())
        sev_pix.setdefault(sev, []).append((pix_labels[-1], pix_scores[-1]))
        if (i + 1) % 50 == 0:
            print(f"[eval]   {i + 1}/{len(test_ds)}")

    pix_labels = np.concatenate(pix_labels)
    pix_scores = np.concatenate(pix_scores)

    results: Dict = {
        "method": scorer.name,
        "dataset": args.dataset,
        "class_name": args.class_name,
        "n_test": len(labels),
        "n_anomalous": int(sum(labels)),
        "image_auroc": _safe_auroc(labels, scores),
        "image_ap": _safe_ap(labels, scores),
        "pixel_auroc": pixel_auroc(pix_labels, pix_scores,
                                   args.max_pixels, args.seed),
    }

    # per-severity breakdown (CONSTRUCT; harmless no-op for MVTec)
    sev_set = sorted({s for s in severities if s not in ("good", "")})
    if sev_set:
        breakdown = {}
        good_idx = [i for i, s in enumerate(severities) if s == "good"]
        for sev in sev_set:
            idx = [i for i, s in enumerate(severities) if s == sev]
            sel = good_idx + idx
            y = [labels[i] for i in sel]
            sc = [scores[i] for i in sel]
            pl = np.concatenate([a for a, _ in sev_pix[sev]])
            ps = np.concatenate([b for _, b in sev_pix[sev]])
            breakdown[sev] = {
                "n_anomalous": len(idx),
                "image_auroc": _safe_auroc(y, sc),
                "image_ap": _safe_ap(y, sc),
                "pixel_auroc": pixel_auroc(pl, ps, args.max_pixels, args.seed),
            }
        results["per_severity"] = breakdown

    out = args.out
    if out is None:
        if method == "pgdn":
            out = os.path.join(os.path.dirname(args.checkpoint), "results.json")
        else:
            out = f"results_patchcore_{args.dataset}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"[eval] wrote {out}")
    return results


if __name__ == "__main__":
    evaluate(parse_args())
