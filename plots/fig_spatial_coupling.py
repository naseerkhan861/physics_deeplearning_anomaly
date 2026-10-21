"""Figure B: spatial coupling pressure and local parameter response.

Panel A: paired synthetic defects on one fixed screw background --
input+GT, prior z, signed coupling pressure u (diverging, global
range), and the actual raw-score change after a small isolated
coupling-direction parameter step.
Panel B: defect-pixel opposition Omega per context and seed for the
synthetic pairs and all real defects.
Panel C: mean mask-score change Delta p_M under the same local step.
The probe diagnoses local direction at frozen checkpoints, not
training trajectories; controls (zero/opposite/random/half-radius)
are reported in the artifact records.
"""

from . import style as S
import csv
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt

def load(data):
    rows = list(csv.DictReader((data / "spatial_probe.csv").open()))
    agg = defaultdict(lambda: defaultdict(list))
    seen = set()
    for r in rows:
        identity = tuple(r[k] for k in ("category", "defect_type", "image", "seed", "arm"))
        if identity in seen:
            raise ValueError(f"duplicate spatial-probe identity: {identity}")
        seen.add(identity)
        for key in ("omega", "delta_mask_mean_p"):
            v = float(r[key])
            if v == v:
                agg[(r["category"], r["arm"], key)][r["seed"]].append(v)
    out = {}
    for k, per_seed in agg.items():
        out[k] = {s: float(np.mean(v)) for s, v in per_seed.items()}
    expected = {(cat, arm, key) for cat in S.CATS for arm in ("cpl", "incpl")
                for key in ("omega", "delta_mask_mean_p")}
    if set(out) != expected or any(set(v) != {"42", "0", "1"}
                                   for v in out.values()):
        raise ValueError("incomplete category/arm/seed spatial-probe summaries")
    syn = np.load(data / "maps" / "synthetic_probes.npz")
    return out, syn


def main():
    S.setup()
    data = S.ARTIFACT / "defect_analysis_v3"
    agg, syn = load(data)
    fig = plt.figure(figsize=(7.16, 6.1))
    fig.set_constrained_layout(False)
    outer = fig.add_gridspec(2, 1, left=0.085, right=0.98, top=0.94,
                            bottom=0.13, hspace=0.40,
                            height_ratios=[2.7, 1.7])
    maps = outer[0].subgridspec(3, 4, height_ratios=[1, 1, 0.10],
                              hspace=0.18, wspace=0.15)
    charts = outer[1].subgridspec(1, 2, wspace=0.36)

    # Panel A: two paired synthetic rows (background 0)
    keys = ["bg0_perlin", "bg0_smooth"]
    titles = ["Perlin defect", "smooth defect"]
    pmax = float(max(np.abs(syn[f"{k}__pressure"]).max() for k in keys))
    dmax = float(max(np.abs(syn[f"{k}__p0_minus_p1"]).max() for k in keys))
    for ri, (k, t) in enumerate(zip(keys, titles)):
        img = syn[f"{k}__image"]
        msk = syn[f"{k}__mask"].astype(bool)
        z = syn[f"{k}__z"]
        u = syn[f"{k}__pressure"]
        # Stored arrays are BEFORE minus AFTER; all displayed/summary changes
        # use AFTER minus BEFORE. Keep source arrays untouched.
        dp = -syn[f"{k}__p0_minus_p1"]
        cells = [(img, {"cmap": "gray"}, None, "input + GT"),
                 (z, {"cmap": "viridis", "vmin": 0, "vmax": 1}, None, "prior $z$"),
                 (u / 1e-5, {"cmap": "RdBu_r", "vmin": -pmax/1e-5, "vmax": pmax/1e-5}, "u",
                  "coupling pressure $u$"),
                 (dp / 1e-3, {"cmap": "RdBu_r", "vmin": -dmax/1e-3, "vmax": dmax/1e-3}, "$\\Delta p$",
                  "local-step score change")]
        for ci, (arr, kw, cb, title) in enumerate(cells):
            ax = fig.add_subplot(maps[ri, ci])
            im = ax.imshow(arr, **kw)
            ax.contour(msk.astype(float), levels=[0.5], colors="lime",
                       linewidths=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("0.45")
                sp.set_linewidth(0.4)
            if ri == 0:
                ax.set_title(title, fontsize=8, pad=3)
            if ci == 0:
                ax.text(-0.12, 0.5, t, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=8)
            if ri == 1 and ci in (1, 2, 3):
                bar_ax = fig.add_subplot(maps[2, ci])
                box = bar_ax.get_position()
                bar_ax.set_position([box.x0 + 0.075*box.width, box.y0,
                                     0.85*box.width, box.height])
                bar = fig.colorbar(im, cax=bar_ax, orientation="horizontal")
                bar.ax.tick_params(labelsize=6.5, pad=1)
                bar.set_label(("prior $z$", "$u$ ($\\times10^{-5}$)", "$\\Delta p$ ($\\times10^{-3}$)")[ci-1],
                              fontsize=7, labelpad=1)
                if ci == 1:
                    bar.set_ticks([0, 0.5, 1])
                else:
                    limit = pmax/1e-5 if ci == 2 else dmax/1e-3
                    bar.set_ticks([-limit, 0, limit])
                    bar.formatter.set_scientific(False)
                    bar.formatter.set_useOffset(False)
                    bar.formatter.set_powerlimits((-100, 100))
                    bar.update_ticks()
                    bar.set_ticklabels([f"{-limit:.1f}", "0", f"{limit:.1f}"])
    fig.text(0.5, 0.965, "(A) paired synthetic screw defects (seed 42, $cpl$ arm)",
             ha="center", fontsize=8)

    # Panels B and C: synthetic diamonds + real per category/arm seed dots
    import json
    syns = json.load((data / "synthetic_summary.json").open())
    groups = ([("synth.\nPerlin", "syn", "perlin"),
               ("synth.\nsmooth", "syn", "smooth")] +
              [(f"{cat}\n{'C' if arm == 'cpl' else 'I+C'}", cat, arm) for cat in S.CATS
               for arm in ("cpl", "incpl")])
    for pi, key in enumerate(("omega", "delta_mask_mean_p")):
        ax = fig.add_subplot(charts[0, pi])
        for x, (label, cat, arm) in enumerate(groups):
            if cat == "syn":
                vals = [v[key] for k, v in syns.items() if arm in k]
                col, mk = "0.25", "D"
            else:
                vals = list(agg.get((cat, arm, key), {}).values())
                col = S.CAT_COLOR[cat] if arm == "incpl" else "0.55"
                mk = "o" if arm == "incpl" else "s"
            if not vals:
                raise ValueError(f"missing plotted spatial-probe group: {label}")
            if key == "delta_mask_mean_p":
                vals = [v * 1e4 for v in vals]  # display units only
            ax.scatter([x] * len(vals), vals, s=14, marker=mk, color=col,
                       linewidths=0.3, edgecolors="k", zorder=3)
            ax.plot([x - 0.18, x + 0.18], [np.mean(vals)] * 2, color=col,
                    lw=1.4, zorder=4)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[0] for g in groups], fontsize=7,
                           rotation=35, ha="right", rotation_mode="anchor")
        if key == "omega":
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color="0.7", lw=0.6, ls=":")
            ax.set_ylabel(r"opposition $\Omega$")
            ax.set_title("(B) opposition on defect pixels", fontsize=8)
        else:
            ax.axhline(0, color="k", lw=0.8)
            ax.set_ylabel(r"$\Delta p_M$ ($\times10^{-4}$)")
            ax.set_title("(C) local coupling-direction step", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    fig.text(0.53, 0.02, "C: coupling only    I+C: input + coupling    bars: means (not confidence intervals)",
             ha="center", fontsize=7.5)
    S.save(fig, "fig_spatial_coupling")


if __name__ == "__main__":
    main()
