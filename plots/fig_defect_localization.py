"""Figure A: real-defect localization under matched coupling arms.

Panels A-C: atlas of the pre-declared median-z examples (screw/
manipulated_front, capsule/scratch, bottle/contamination) -- image
with GT contour, prior z, and input-on C-off (in) versus C-on (incpl)
evaluation maps on a shared [0,1] scale with each arm's threshold
contour at the 1% empirical normal-pixel budget.
Panels D-F: paired defect-coverage change (C-on minus C-off, pp) per
recorded defect type for both input contexts and all three training
seeds; image-equal means within type, seed estimates and their mean.
Evaluation diagnostic: thresholds use known normal test pixels.
"""

from . import style as S
import csv
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt

CATS = ("screw", "capsule", "bottle")


def load_all(data):
    rows = list(csv.DictReader((data / "coverage.csv").open()))
    thr = {}
    for r in csv.DictReader((data / "thresholds.csv").open()):
        thr[(r["category"], r["seed"], r["arm"])] = float(r["threshold"])
    expected = {(cat, seed, arm) for cat in CATS
                for seed in ("42", "0", "1")
                for arm in ("base", "in", "cpl", "incpl")}
    if set(thr) != expected:
        raise ValueError("missing or unexpected category/seed/arm thresholds")
    eff = defaultdict(dict)
    for r in rows:
        ctx = "off" if r["arm"] in ("base", "cpl") else "on"
        on = r["arm"] in ("cpl", "incpl")
        key = (r["category"], r["defect_type"], r["image"], r["seed"], ctx)
        value = float(r["coverage_at_1pct"])
        if on in eff[key] or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"duplicate or invalid coverage record: {key}")
        eff[key][on] = value
    per = defaultdict(lambda: defaultdict(list))
    for (cat, dt, img, seed, ctx), d in eff.items():
        if set(d) != {False, True}:
            raise ValueError(f"unpaired C-off/C-on observation: {cat}/{dt}/{img}")
        per[(cat, ctx)][(dt, seed)].append(100 * (d[1] - d[0]))
    for cat in CATS:
        for ctx in ("off", "on"):
            types = {dt for dt, _ in per[(cat, ctx)]}
            if not types or any((dt, sd) not in per[(cat, ctx)]
                                for dt in types for sd in ("42", "0", "1")):
                raise ValueError(f"incomplete seed/type records: {cat}/{ctx}")
    return per, thr


def atlas_panel(ax_row, cat, thr, data, show_titles):
    d = np.load(data / "maps" / f"{cat}_atlas.npz")
    img, mask, z = d["image"], d["mask"].astype(bool), d["prior_z"]
    p_off, p_on = d["p_eval_in"], d["p_eval_incpl"]
    t_off, t_on = thr[(cat, "42", "in")], thr[(cat, "42", "incpl")]
    vmax = float(max(p_off.max(), p_on.max(), 1.0))  # shared pred scale
    panels = [(img, {"cmap": "gray"}, None, None, "image + GT"),
              (z, {"cmap": "viridis", "vmin": 0.0, "vmax": 1.0}, None, None,
               "prior $z$"),
              (p_off, {"cmap": "magma", "vmin": 0.0, "vmax": vmax},
               t_off, ("w", "--"), "C-off ($in$)"),
              (p_on, {"cmap": "magma", "vmin": 0.0, "vmax": vmax},
               t_on, ("w", "-"), "C-on ($incpl$)")]
    meshes = []
    for ax, (arr, kw, t, cst, title) in zip(ax_row, panels):
        meshes.append(ax.imshow(arr, **kw))
        ax.contour(mask.astype(float), levels=[0.5], colors="lime",
                   linewidths=0.7)
        if t is not None and arr.min() < t < arr.max():
            ax.contour(arr, levels=[t], colors=cst[0], linestyles=cst[1],
                       linewidths=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("0.45")
            sp.set_linewidth(0.4)
        if show_titles:
            ax.set_title(title, fontsize=8, pad=3)
    return meshes


def main():
    S.setup()
    data = S.ARTIFACT / "defect_analysis_v1"
    per, thr = load_all(data)
    fig = plt.figure(figsize=(7.16, 6.5))
    # Separate grids give all three charts the complete figure width.
    fig.set_constrained_layout(False)
    outer = fig.add_gridspec(2, 1, left=0.09, right=0.985, top=0.96,
                            bottom=0.135, hspace=0.44,
                            height_ratios=[3.35, 1.65])
    atlas = outer[0].subgridspec(4, 4, height_ratios=[1, 1, 1, 0.09],
                               hspace=0.16, wspace=0.12)
    charts = outer[1].subgridspec(1, 3, wspace=0.38)
    for ri, cat in enumerate(CATS):
        axes = [fig.add_subplot(atlas[ri, c]) for c in range(4)]
        meshes = atlas_panel(axes, cat, thr, data, show_titles=ri == 0)
        axes[0].text(-0.10, 0.5, f"({'ABC'[ri]}) {cat}", rotation=90,
                     transform=axes[0].transAxes, va="center", ha="right",
                     fontsize=8)
    for col, mesh, label in [(1, meshes[1], "prior $z$"),
                              (3, meshes[3], "anomaly score")]:
        cb = fig.colorbar(mesh, cax=fig.add_subplot(atlas[3, col]),
                          orientation="horizontal", ticks=[0, 0.5, 1])
        cb.ax.tick_params(labelsize=7, pad=1)
        cb.set_label(label, fontsize=7.5, labelpad=1)
    # charts: fourth row, one panel per category
    for ci, cat in enumerate(CATS):
        ax = fig.add_subplot(charts[0, ci])
        dts = sorted({dt for (dt, s_) in per[(cat, "off")]})
        xs = np.arange(len(dts))
        for c_i, ctx in enumerate(("off", "on")):
            col = S.CAT_COLOR[cat] if ctx == "on" else "0.55"
            for di, dt in enumerate(dts):
                vals = [np.mean(per[(cat, ctx)][(dt, s_)])
                        for s_ in ("42", "0", "1")
                        if (dt, s_) in per[(cat, ctx)]]
                x = di + (c_i - 0.5) * 0.34
                ax.scatter([x] * len(vals), vals, s=8,
                           marker="o" if ctx == "on" else "s",
                           color=col, linewidths=0.3, edgecolors="k",
                           zorder=3)
                ax.plot([x - 0.13, x + 0.13], [np.mean(vals)] * 2, lw=1.4,
                        color=col, zorder=4)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([d.replace("_", "\n") for d in dts], fontsize=7,
                           rotation=35, ha="right", rotation_mode="anchor")
        ax.set_title(f"({'DEF'[ci]}) {cat}", fontsize=8.5)
        if ci == 0:
            ax.set_ylabel("$\\Delta$ coverage, C-on $-$ C-off (pp)")
        ax.tick_params(axis="y", labelsize=7)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=4, color="0.3",
                          label="input on ($incpl-in$)"),
               plt.Line2D([], [], ls="", marker="s", ms=4, color="0.6",
                          label="input off ($cpl-base$)"),
               plt.Line2D([], [], color="0.3", lw=1.4, label="seed mean")]
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), fontsize=7.5)
    S.save(fig, "fig_defect_localization")


if __name__ == "__main__":
    main()
