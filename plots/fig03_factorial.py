"""F3: pathway separation -- matched coupling contrasts (round-9 plan).

Panel A: synthetic family-response contrast K_{i,r} = 100*(F_{i,r,1} -
F_{i,r,0}) for three categories, four input/reconstruction contexts,
three training seeds (36 points), computed from full-precision records.
Panel B: real-defect contrasts with derivative matching disabled -- additive
interaction D versus combined-minus-best B; T supplied as annotations.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from . import style as S

import matplotlib.pyplot as plt

CTX = [("i=0,r=0", "input off / deriv. off"),
       ("i=1,r=0", "input on / deriv. off"),
       ("i=0,r=1", "input off / deriv. on"),
       ("i=1,r=1", "input on / deriv. on")]
PAIR = {"i=0,r=0": ("cpl", "base"), "i=1,r=0": ("incpl", "in"),
        "i=0,r=1": ("recpl", "rec"), "i=1,r=1": ("full", "inrec")}


def panel_a(ax):
    recs = {42: S.load("smooth_factorial.json"),
            0: S.load("smooth_factorial_s0.json"),
            1: S.load("smooth_factorial_s1.json")}
    xs = np.arange(len(CTX))
    for ci, cat in enumerate(S.CATS):
        for xi, (key, _label) in enumerate(CTX):
            on, off = PAIR[key]
            for seed in (42, 0, 1):
                r = recs[seed]
                F = lambda a: (r[f"{cat}/{a}"]["smooth"]
                               - r[f"{cat}/{a}"]["perlin"])
                K = 100 * (F(on) - F(off))
                ax.scatter(xi + (ci - 1) * 0.22, K,
                           marker=S.SEED_MARK[seed], s=16,
                           color=S.CAT_COLOR[cat], linewidths=0.4, edgecolors="k")
    ax.axhline(0, color="k", lw=0.8, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([c[1].replace(" / ", "\n") for c in CTX])
    ax.set_ylabel("matched coupling contrast $K_{i,r}$ (pp)")
    handles = [plt.Line2D([], [], marker=S.SEED_MARK[s], ls="", ms=4,
                          color="k", label=f"seed {s}") for s in (42, 0, 1)]
    handles += [plt.Line2D([], [], ls="", marker="o", ms=5, color=S.CAT_COLOR[c],
                           label=c) for c in S.CATS]
    handles = [h for pair in zip(handles[:3], handles[3:]) for h in pair]
    ax.legend(handles=handles, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), columnspacing=1.0, handletextpad=0.4)
    ax.set_title("(A) synthetic family response: coupling on minus off")


def panel_b(ax):
    real = S.load("real_moderation.json")
    inter = S.load("real_interaction.json")
    rows = []
    for cat in S.CATS:
        e = real["categories"][cat]
        for seed in ("42", "0", "1"):
            def auc(arm):
                g = e["per_image"][arm][seed]["good"]
                df = e["per_image"][arm][seed]["defects"]
                y = [0] * len(g) + [1] * len(df)
                return roc_auc_score(y, g + df)
            Y = {a: auc(a) for a in ("base", "in", "cpl", "incpl")}
            D = 100 * (Y["incpl"] - Y["in"] - Y["cpl"] + Y["base"])
            B = 100 * (Y["incpl"] - max(Y["in"], Y["cpl"]))
            T = 100 * (Y["incpl"] - Y["base"])
            rows.append((cat, seed, D, B, T))
    ce = inter["cashew_endpoints"]["per_seed"]
    for seed in ("42", "0", "1"):
        y = {a: ce[seed][a]["image_auroc"] for a in ("base", "in", "cpl", "incpl")}
        rows.append(("cashew", seed,
                     100 * (y["incpl"] - y["in"] - y["cpl"] + y["base"]),
                     100 * (y["incpl"] - max(y["in"], y["cpl"])),
                     100 * (y["incpl"] - y["base"])))
    cat_colors = {**{c: S.CAT_COLOR[c] for c in S.CATS}, "cashew": "#56B4E9"}
    for cat, seed, D, B, T in rows:
        ax.scatter(D, B, marker=S.SEED_MARK[int(seed)], s=22,
                   color=cat_colors[cat], linewidths=0.4, edgecolors="k")
        ax.annotate(f"{T:+.0f}", (D, B), textcoords="offset points",
                    xytext=(4, 3), fontsize=6.5, color="0.35")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("additive interaction $D$ (pp)")
    ax.set_ylabel("combined $-$ best single $B$ (pp)")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5,
                          color=cat_colors[c], label=c)
               for c in (*S.CATS, "cashew")]
    ax.legend(handles=handles, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.27))
    ax.set_title("(B) real defects: annotations give $T$ (pp)")


def main():
    S.setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    panel_a(axes[0])
    panel_b(axes[1])
    S.save(fig, "fig03_factorial")


if __name__ == "__main__":
    main()
