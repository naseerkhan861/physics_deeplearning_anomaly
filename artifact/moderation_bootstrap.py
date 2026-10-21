"""Canonical moderation bootstrap (round-5 review Step 1).

Paired image-cluster percentile bootstrap over the saved per-image
scores in real_moderation.json.  Sampling unit: the image, resampled
jointly and identically across arms, groups, and the shared normal
negatives (pooled multinomial over weak, strong, and normal image
sets).  Group membership is held fixed.  The pairwise matrix gives the
empirical AUROC (1 = positive exceeds normal, 0.5 = tie); multiplicity
weights reproduce image resampling while preserving pairing.  Point
effects are validated against the saved record before intervals are
reported.  Intervals are nominal and exploratory; replicate count is
chosen for tail stability, not a journal rule.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

B = 10_000
SEED_BASE = 20261015


def main(record_path="results/raw/real_moderation.json",
         out_path="results/raw/real_moderation_ci.json"):
    record = json.loads(Path(record_path).read_text())
    out = {"replicates": B, "method": "paired image-cluster percentile "
            "bootstrap; pooled multinomial resampling of weak/strong/normal "
            "image sets; groups fixed; rng seed 20261015 + category index",
           "categories": {}}
    for ci, cat in enumerate(("screw", "capsule", "bottle")):
        e = record["categories"][cat]
        paths = sorted(e["groups"])
        weak = np.array([e["groups"][p] == "weak" for p in paths])
        nw, ns, ng = int(weak.sum()), int((~weak).sum()), e["n_good"]
        n_total = nw + ns + ng
        rng = np.random.default_rng(SEED_BASE + ci)
        counts = rng.multinomial(n_total, np.ones(n_total) / n_total, size=B)
        w = counts[:, :nw]
        s = counts[:, nw:nw + ns]
        n = counts[:, nw + ns:]
        denom_w, denom_s = w.sum(1) * n.sum(1), s.sum(1) * n.sum(1)
        assert np.all(denom_w > 0) and np.all(denom_s > 0)

        res = {}
        for seed in ("42", "0", "1"):
            matrices = {}
            for arm in ("base", "in", "cpl", "incpl"):
                scores = e["per_image"][arm][seed]
                positive = np.asarray(scores["defects"])
                negative = np.asarray(scores["good"])
                matrices[arm] = (
                    (positive[:, None] > negative).astype(float)
                    + 0.5 * (positive[:, None] == negative))
            for context, off, on in (("off", "base", "cpl"),
                                     ("on", "in", "incpl")):
                difference = matrices[on] - matrices[off]
                d_weak, d_strong = difference[weak], difference[~weak]
                point = 100 * (d_weak.mean() - d_strong.mean())
                assert abs(point - e["effects"][seed][context]["M"]) < 1e-10
                weak_effect = np.einsum(
                    "bi,ij,bj->b", w, d_weak, n, optimize=True) / denom_w
                strong_effect = np.einsum(
                    "bi,ij,bj->b", s, d_strong, n, optimize=True) / denom_s
                interval = np.quantile(100 * (weak_effect - strong_effect),
                                       [0.025, 0.975])
                res.setdefault(seed, {})[context] = {
                    "M": round(float(point), 2),
                    "ci95": [round(float(interval[0]), 1),
                             round(float(interval[1]), 1)]}
                print(cat, seed, context, round(point, 2),
                      np.round(interval, 1).tolist(), flush=True)
        out["categories"][cat] = res
    Path(out_path).write_text(json.dumps(out, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:  # explicit record/output paths
        main(sys.argv[1], sys.argv[2])
    elif os.path.isfile("artifact/real_moderation.json"):  # repo checkout
        main("artifact/real_moderation.json", "artifact/real_moderation_ci.json")
    else:  # experiment project layout
        main()
