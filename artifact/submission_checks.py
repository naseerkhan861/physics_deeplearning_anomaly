"""Recompute paired manuscript sensitivity checks from released records.

Run from any directory with Python 3 (standard library only):
    python artifact/submission_checks.py

This reanalyzes aggregate endpoints; it does not rerun training, estimate
image-sampling uncertainty, or treat category/seed cells as independent.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


CATEGORIES = {
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
}
SEEDS = ("42", "0", "1")


def endpoint(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a numeric AUROC")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label}: AUROC must be finite and in [0, 1]")
    return float(value)


def analyze(artifact):
    source_names = ("router_results.json", "repeatability.json")
    raw = {name: (artifact / name).read_bytes() for name in source_names}
    paired = json.loads(raw["router_results.json"])["per_category"]
    if set(paired) != CATEGORIES:
        raise ValueError("RGB records must contain the 15 specified categories")
    effects = {}
    for category, record in sorted(paired.items()):
        if set(record["seeds"]) != set(SEEDS):
            raise ValueError(f"{category}: expected seeds 42/0/1")
        effects[category] = {}
        for seed in SEEDS:
            real = record["seeds"][seed]["real"]
            p = endpoint(real["pgdn"], f"{category}/{seed}/pgdn")
            t = endpoint(real["draem"], f"{category}/{seed}/draem")
            effects[category][seed] = 100 * (p - t)

    category_means = {c: mean(v.values()) for c, v in effects.items()}
    seed_macros = {s: mean(v[s] for v in effects.values()) for s in SEEDS}
    leave_one_out = {
        c: mean(v for other, v in category_means.items() if other != c)
        for c in effects
    }
    repeat = json.loads(raw["repeatability.json"])["runs"]
    nondeterministic = [endpoint(r["image_auroc"], "repeat image AUROC")
                        for r in repeat if not r["deterministic"]]
    deterministic = [endpoint(r["image_auroc"], "repeat image AUROC")
                     for r in repeat if r["deterministic"]]
    if not nondeterministic or not deterministic:
        raise ValueError("Both repeatability settings must be represented")

    return {
        "analysis": "paired endpoint sensitivity; descriptive, not an inferential test",
        "source_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in raw.items()
        },
        "rgb": {
            "categories": len(effects),
            "seeds": list(SEEDS),
            "paired_cells": sum(len(v) for v in effects.values()),
            "effect_units": "image-AUROC percentage points, PGDN minus twin",
            "macro_effect_pp": mean(category_means.values()),
            "seed_macro_effects_pp": seed_macros,
            "category_mean_effects_pp": category_means,
            "median_category_mean_effect_pp": median(category_means.values()),
            "positive_category_means": sum(v > 0 for v in category_means.values()),
            "positive_cells": sum(v > 0 for d in effects.values() for v in d.values()),
            "categories_winning_all_seeds": [
                c for c, d in effects.items() if all(v > 0 for v in d.values())
            ],
            "categories_losing_all_seeds": [
                c for c, d in effects.items() if all(v < 0 for v in d.values())
            ],
            "leave_one_category_out_pp": leave_one_out,
            "leave_one_category_out_range_pp": [
                min(leave_one_out.values()), max(leave_one_out.values())
            ],
            "scope": (
                "Fixed categories, test images and fitted checkpoints; no resampling "
                "or new inference, and no assumption of 45 independent datasets."
            ),
        },
        "repeatability": {
            "nondeterministic_runs": len(nondeterministic),
            "nondeterministic_auroc_range": [min(nondeterministic), max(nondeterministic)],
            "deterministic_runs": len(deterministic),
            "deterministic_aurocs": deterministic,
            "deterministic_spread": max(deterministic) - min(deterministic),
            "distinct_augmentation_stream_hashes": sorted({r["aug_stream_hash"] for r in repeat}),
            "scope": (
                "Repeatability controls for a separate screw baseline configuration; "
                "repeated values do not establish evaluation validity or competitive accuracy."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.artifact)
    output = args.output or args.artifact / "submission_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rgb = report["rgb"]
    print(f"Paired macro: {rgb['macro_effect_pp']:.4f} pp")
    print(f"Leave-one-category-out range: {rgb['leave_one_category_out_range_pp']}")
    print(f"Winners at all seeds: {rgb['categories_winning_all_seeds']}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
