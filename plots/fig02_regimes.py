"""F2: where the prior helps and hurts.

Panel A: paired benchmark effects from the rounded gray-128 campaign
table (descriptive preview; no full-precision gray export is released).
Panel B: per-category paired RGB/256 effects computed from the released
full-precision records in artifact/router_results.json -- three seed
markers plus the category mean per category.
"""

import re

from . import style as S
import matplotlib.pyplot as plt
import numpy as np

NUM = r"([0-9.]+)"


def bench_effects():
    txt = (S.REPO / "tables" / "main37.tex").read_text()
    rows = []
    for m in re.finditer(r"^(\S[^&]*)&\s*" + NUM + r"\s*&\s*" + NUM, txt, re.M):
        name = m.group(1).strip().replace("\\%", "")
        name = re.sub(r"\\[a-zA-Z]+\{?\}?", "", name).replace("~", " ").strip()
        pg, tw = float(m.group(2)), float(m.group(3))
        if name.startswith("CONSTRUCT"):
            rows.append((name, 100 * (pg - tw), True))
        elif name not in ("benchmark",):
            rows.append((name, 100 * (pg - tw), False))
    return rows


def rgb_effects():
    """Full-precision paired effects from the released routing records:
    effect = 100 x (PGDN - twin) per category and seed."""
    rec = S.load("router_results.json")["per_category"]
    rows = []
    for cat, v in rec.items():
        seeds = v["seeds"]
        effs = {sd: 100 * (seeds[sd]["real"]["pgdn"] - seeds[sd]["real"]["draem"])
                for sd in ("42", "0", "1")}
        effs = [effs[sd] for sd in ("42", "0", "1")]  # explicit seed order
        rows.append((cat.replace("_", " "), effs))
    return rows


def main():
    S.setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    be = [r for r in bench_effects() if not r[2]]
    con = [r for r in bench_effects() if r[2]]
    names = [r[0].replace("Superf.", "Superficial").replace("MVTec AD 2", "MVTecAD2")
             for r in be]
    vals = [r[1] for r in be]
    order = np.argsort(vals)
    ax = axes[0]
    ax.axvline(0, color="k", lw=0.9)
    ax.scatter([vals[i] for i in order], np.arange(len(vals)),
               color="#0072B2", s=20, zorder=3)
    ax.set_yticks(np.arange(len(vals)))
    ax.set_yticklabels([names[i] for i in order], fontsize=6.5)
    if con:
        ax.scatter([con[0][1]], [len(vals)],
                   marker="D", color="#CC79A7", s=26, zorder=3)
        ax.set_yticks(list(ax.get_yticks()) + [len(vals)])
        ax.set_yticklabels([names[i] for i in order] + ["CONSTRUCT (synthetic)"],
                           fontsize=6.5)
    ax.set_xlabel("PGDN $-$ twin, image AUROC (pp)")
    ax.set_title("(A) gray-128 campaign, 6 real benchmarks")

    ax = axes[1]
    re_ = rgb_effects()
    means = np.array([np.mean(e) for _, e in re_])
    order = np.argsort(means)
    ax.axvline(0, color="k", lw=0.9)
    for yi, i in enumerate(order):
        cat, effs = re_[i]
        ax.plot([min(effs), max(effs)], [yi, yi], color="0.75", lw=0.8, zorder=1)
        for sd, e in zip((42, 0, 1), effs):
            ax.scatter(e, yi, marker=S.SEED_MARK[sd], s=12, color="#E69F00",
                       linewidths=0.3, edgecolors="k", zorder=3)
        ax.scatter(np.mean(effs), yi, marker="D", s=22, color="#0072B2",
                   linewidths=0.4, edgecolors="k", zorder=4)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([re_[i][0] for i in order], fontsize=6.5)
    ax.set_xlabel("PGDN $-$ twin, image AUROC (pp)")
    ax.set_title("(B) RGB/256 MVTec AD: seeds and means")
    handles = [plt.Line2D([], [], ls="", marker=S.SEED_MARK[sd], ms=4,
                          color="#E69F00", label=f"seed {sd}") for sd in (42, 0, 1)]
    handles.append(plt.Line2D([], [], ls="", marker="D", ms=5,
                              color="#0072B2", label="category mean"))
    ax.legend(handles=handles, loc="lower right", fontsize=6.5)

    S.save(fig, "fig02_regimes")


if __name__ == "__main__":
    main()
