"""Derive the predeclared stronger-recipe contrasts from completed fits.

K(i,s) = 100*[F(i, C=1, s) - F(i, C=0, s)],  F = A_smooth - A_perlin  (pp)
D = AUROC(incpl) - in - cpl + base;  B = incpl - max(in, cpl);
T = incpl - base  (real image AUROC, pp)

Primary contrast (frozen): K on screw, input OFF, mean over the three
seeds; persistence rule = adverse (negative) mean with at least 2/3
seeds negative, evaluated at FULL PRECISION (rounding only for
display).  Completion requires all 36 family AND 36 real records with
the expected fields; an incomplete rerun overwrites the output with an
explicit incomplete status so a stale complete snapshot cannot be
mistaken for a current one.
"""
import glob
import hashlib
import json
import math
import os

ARMS = ("base", "in", "cpl", "incpl")
OUT = "results/raw/ref_analysis.json"


def _load(path):
    if not os.path.isfile(path):
        return None
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None


def _valid_family(d):
    return (isinstance(d, dict) and isinstance(d.get("perlin"), (int, float))
            and isinstance(d.get("smooth"), (int, float))
            and math.isfinite(d["perlin"]) and math.isfinite(d["smooth"]))


def _valid_eval(d):
    return (isinstance(d, dict) and isinstance(d.get("image_auroc"), (int, float))
            and math.isfinite(d["image_auroc"]))


def _inputs_digest(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode())
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def main():
    fam_paths = sorted(glob.glob("results/raw/ref_*_family.json"))
    ev_paths = sorted(glob.glob("results/raw/ref_*_eval.json"))
    missing = []
    K = {}
    real = {}
    for cat in ("screw", "capsule", "bottle"):
        for seed in ("42", "0", "1"):
            fam = {a: _load(f"results/raw/ref_{cat}_{a}_s{seed}_family.json")
                   for a in ARMS}
            ev = {a: _load(f"results/raw/ref_{cat}_{a}_s{seed}_eval.json")
                  for a in ARMS}
            for a in ARMS:
                if not _valid_family(fam[a]):
                    missing.append(f"{cat}/{a}/s{seed} family")
                if not _valid_eval(ev[a]):
                    missing.append(f"{cat}/{a}/s{seed} real")
            if any(not _valid_family(fam[a]) for a in ARMS):
                continue
            F = {a: 100.0 * (fam[a]["smooth"] - fam[a]["perlin"]) for a in ARMS}
            K.setdefault(cat, {})[seed] = {
                "input_off": F["cpl"] - F["base"],
                "input_on": F["incpl"] - F["in"],
                "absolute": {a: {"perlin": fam[a]["perlin"],
                                 "smooth": fam[a]["smooth"]} for a in ARMS}}
            if all(_valid_eval(ev[a]) for a in ARMS):
                A = {a: ev[a]["image_auroc"] for a in ARMS}
                real.setdefault(cat, {})[seed] = {
                    "D": 100.0 * (A["incpl"] - A["in"] - A["cpl"] + A["base"]),
                    "B": 100.0 * (A["incpl"] - max(A["in"], A["cpl"])),
                    "T": 100.0 * (A["incpl"] - A["base"]),
                    "absolute": {a: A[a] for a in ARMS}}

    complete = (not missing and len(K) == 3
                and all(len(v) == 3 for v in K.values())
                and len(real) == 3
                and all(len(v) == 3 for v in real.values()))

    def show(d):
        return {c: {s: {k: round(v, 1) for k, v in cell.items()
                        if isinstance(v, float)}
                    for s, cell in seeds.items()}
                for c, seeds in d.items()} if d else {}

    out = {"K_pp_full_precision": K or {},
           "real_contrasts_full_precision": real or {},
           "K_pp_display": show(K), "real_contrasts_pp_display": show(real)}
    if complete:
        off = [K["screw"][s]["input_off"] for s in ("42", "0", "1")]
        mean = sum(off) / len(off)
        n_neg = sum(x < 0 for x in off)
        verdict = ("persists" if (n_neg >= 2 and mean < 0)
                   else "criterion not met")
        out["primary_contrast"] = {
            "definition": "K on screw, input OFF, mean over seeds 42/0/1",
            "rule": "adverse mean with >=2/3 seeds negative (full precision)",
            "seed_values_full_precision": off,
            "mean_full_precision": mean,
            "verdict": verdict,
            "note": "descriptive replication criterion, not a significance "
                    "test; criterion failure is not proof of absence"}
        out["status"] = "complete"
        out["inputs_digest"] = _inputs_digest(fam_paths + ev_paths)
        out["n_family_records"] = len(fam_paths)
        out["n_real_records"] = len(ev_paths)
    else:
        out["status"] = "incomplete"
        out["missing"] = missing[:20]
        out["note"] = ("any previously completed snapshot is superseded by "
                       "this incomplete status; do not interpret stale "
                       "ref_analysis.json content as current")
    json.dump(out, open(OUT, "w"), indent=2)
    print("status:", out["status"],
          "| missing:", len(missing),
          "| verdict:", out.get("primary_contrast", {}).get("verdict", "-"))


if __name__ == "__main__":
    main()
