"""F6: real-defect transfer with uncertainty (round-9 plan).

Panel A: forest plot of all 18 moderation estimates M with the released
nominal 95% bootstrap intervals (three categories, two input contexts,
three training seeds). Panel B: paired weak/strong subgroup coupling
effects E_g, making the screw counterexample visible (coupling helps
both groups there, just less so on the weak group).
"""

from . import style as S
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def forest(ax):
    real = S.load("real_moderation.json")
    ci = S.load("real_moderation_ci.json")
    labels, ys = [], []
    y = 0
    yticks, yticklab = [], []
    for cat in S.CATS:
        e = real["categories"][cat]
        for I in ("off", "on"):
            for seed in ("42", "0", "1"):
                m = e["effects"][seed][I]["M"]
                lo, hi = ci["categories"][cat][seed][I]["ci95"]
                color = S.CAT_COLOR[cat]
                ax.plot([lo, hi], [y, y], color=color, lw=1.4, solid_capstyle="round")
                ax.plot([lo, lo], [y - 0.18, y + 0.18], color=color, lw=1.0)
                ax.plot([hi, hi], [y - 0.18, y + 0.18], color=color, lw=1.0)
                ax.plot(m, y, S.SEED_MARK[int(seed)], color=color,
                        ms=5, markeredgecolor="k", markeredgewidth=0.4)
                yticks.append(y)
                yticklab.append(f"{cat[:3]}/{I}/{seed}")
                y += 1
            y += 0.6
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklab, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("moderation $M = E_{\\mathrm{weak}} - E_{\\mathrm{strong}}$ (pp)")
    ax.set_title("(A) moderation with nominal 95% intervals")


def effects(ax):
    real = S.load("real_moderation.json")
    for cat in S.CATS:
        e = real["categories"][cat]
        weak = np.array([e["groups"][p] == "weak" for p in sorted(e["groups"])])
        for seed in ("42", "0", "1"):
            for xi, I in enumerate(("off", "on")):
                on, off = ("cpl", "base") if I == "off" else ("incpl", "in")

                def auck(arm, mask):
                    g = e["per_image"][arm][seed]["good"]
                    df = [s for s, m in zip(e["per_image"][arm][seed]["defects"], mask) if m]
                    yv = [0] * len(g) + [1] * len(df)
                    return roc_auc_score(yv, g + df)
                Ew = 100 * (auck(on, weak) - auck(off, weak))
                Es = 100 * (auck(on, ~weak) - auck(off, ~weak))
                dx = 0.13 * (["42", "0", "1"].index(seed) - 1)
                ax.plot([xi + dx - 0.06, xi + dx + 0.06], [Ew, Ew],
                        color=S.CAT_COLOR[cat], lw=1.6)
                ax.plot([xi + dx - 0.06, xi + dx + 0.06], [Es, Es],
                        color=S.CAT_COLOR[cat], lw=1.6, ls=(0, (3, 1.6)))
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["input off", "input on"])
    ax.set_ylabel("subgroup coupling effect $E_g$ (pp)")
    handles = [plt.Line2D([], [], color="0.3", lw=1.6, label="$E_{\\mathrm{weak}}$"),
               plt.Line2D([], [], color="0.3", lw=1.6, ls=(0, (3, 1.6)),
                          label="$E_{\\mathrm{strong}}$")]
    handles += [plt.Line2D([], [], ls="", marker=S.SEED_MARK[sd], ms=5,
                           color="0.3", label=f"seed {sd}")
                for sd in (42, 0, 1)]
    handles += [plt.Line2D([], [], ls="", marker="s", ms=5,
                           color=S.CAT_COLOR[c], label=c) for c in S.CATS]
    ax.legend(handles=handles, ncol=2, fontsize=6.5, loc="lower left")
    ax.set_title("(B) subgroup effects by seed")


def main():
    S.setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    forest(axes[0])
    effects(axes[1])
    S.save(fig, "fig06_real_transfer")


if __name__ == "__main__":
    main()
