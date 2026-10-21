"""Repeatability study (review Step 4): why do nominally identical runs
(0.092 / 0.374 / 0.456 screw baseline) diverge?

Runs R identical trainings of the screw baseline (same seed, same config),
recording per run: shared-layer init-state hash, per-index augmentation
mask hash, label/score-orientation checks, final train loss, and test
metrics.  Then repeats two runs under torch deterministic mode
(use_deterministic_algorithms + cudnn.deterministic) to test whether
determinism closes the spread.  Writes results/raw/repeatability.json.
"""

import hashlib
import json

import numpy as np
import torch

from datasets.loaders import MVTecScrewDataset
import eval as eval_mod
import train as train_mod
from eval import evaluate
from train import train as train_fn


def h(x):
    return hashlib.sha1(np.ascontiguousarray(x)).hexdigest()[:10]


def run_once(rep, deterministic):
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    argv = ["--dataset", "mvtec",
            "--data-root", "/home/edison/VS_Projects/defectsGenerations/mvtec_dataset/screw",
            "--run-name", f"rep_{rep}_{'det' if deterministic else 'nondet'}",
            "--epochs", "8", "--seed", "42", "--no-physics", "--device", "cuda"]
    args = train_mod.parse_args(argv)

    # fingerprint the augmentation stream (identical across reps if the
    # per-index generator is the only randomness source for data)
    ds = train_mod.build_dataset(args)
    masks = []
    for i in range(0, len(ds), 40):
        m = ds[i]["mask"].numpy()
        masks.append(m)
    aug_hash = h(np.stack(masks))

    train_fn(args)
    ck = torch.load(f"runs/{args.run_name}/last.pt", map_location="cpu",
                    weights_only=False)
    log = ck.get("train_config") and json.load(open(f"runs/{args.run_name}/log.json"))
    ev = evaluate(eval_mod.parse_args(["--dataset", "mvtec",
                                       "--data-root", args.data_root,
                                       "--checkpoint", f"runs/{args.run_name}/last.pt",
                                       "--seed", "42", "--device", "cuda",
                                       "--out", f"results/raw/rep_{rep}_eval.json"]))
    # label/score orientation check: mean score of defective vs good test images
    test = MVTecScrewDataset(args.data_root, "test", image_size=128, seed=42)
    from eval import PGDNScorer
    sc = PGDNScorer(f"runs/{args.run_name}/last.pt", device="cuda")
    good = [sc.score(test[i]["image"][0].numpy())[0] for i in range(len(test))
            if test.items[i]["label"] == 0][:30]
    bad = [sc.score(test[i]["image"][0].numpy())[0] for i in range(len(test))
           if test.items[i]["label"] == 1][:30]
    return {"rep": rep, "deterministic": deterministic,
            "aug_stream_hash": aug_hash,
            "final_loss": log[-1]["loss"] if log else None,
            "image_auroc": ev["image_auroc"], "pixel_auroc": ev["pixel_auroc"],
            "orientation_ok": float(np.mean(bad)) > float(np.mean(good)),
            "mean_score_good": float(np.mean(good)), "mean_score_bad": float(np.mean(bad))}


def main():
    out = {"config": "screw baseline, seed 42, 8 epochs, lr 1e-3, fractal, gray-128",
           "runs": []}
    for rep in range(4):
        out["runs"].append(run_once(rep, deterministic=False))
        print(out["runs"][-1])
    for rep in range(2):
        out["runs"].append(run_once(100 + rep, deterministic=True))
        print(out["runs"][-1])
    nd = [r for r in out["runs"] if not r["deterministic"]]
    dd = [r for r in out["runs"] if r["deterministic"]]
    out["summary"] = {
        "nondeterministic_image_auroc": [round(r["image_auroc"], 4) for r in nd],
        "nondeterministic_spread": float(max(r["image_auroc"] for r in nd)
                                         - min(r["image_auroc"] for r in nd)),
        "deterministic_image_auroc": [round(r["image_auroc"], 4) for r in dd],
        "deterministic_spread": float(max(r["image_auroc"] for r in dd)
                                      - min(r["image_auroc"] for r in dd)) if dd else None,
        "aug_streams_identical": len({r["aug_stream_hash"] for r in out["runs"]}) == 1,
        "orientation_ok_all": all(r["orientation_ok"] for r in out["runs"]),
    }
    print(json.dumps(out["summary"], indent=2))
    with open("results/raw/repeatability.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
