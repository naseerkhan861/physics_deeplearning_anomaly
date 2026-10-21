"""F5: gradient mechanism -- analytic sign condition and measured conflict.

Panel A (analytic illustration): sign of dC/dp_j = (zbar_p - z_j)/P over a
(z_j, zbar_p) grid; positive entries push p_j down under gradient descent.
Panel B (measured): parameter-gradient cosine alignment versus loss-weighted
norm ratio (log scale) from grad_params.json, Perlin and smooth families.
"""

from . import style as S
import matplotlib.pyplot as plt
import numpy as np


def panel_a(ax):
    z = np.linspace(0, 1, 201)
    Zj, Zb = np.meshgrid(z, z)
    sign = np.sign(Zb - Zj)
    ax.imshow(sign, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
              vmin=-1.5, vmax=1.5, aspect="auto", alpha=0.85)
    ax.plot([0, 1], [0, 1], color="k", lw=0.8)
    ax.set_xlabel(r"pixel prior response $z_j$")
    ax.set_ylabel(r"prediction-weighted mean $\bar z_p$")
    ax.set_title(r"(A) analytic: sign of $\partial C/\partial p_j$")
    ax.text(0.08, 0.78, "suppress $p_j$", fontsize=7.5)
    ax.text(0.60, 0.18, "encourage $p_j$", fontsize=7.5)
    ax.grid(False)


def panel_b(ax):
    gp = S.load("grad_params.json")["runs"]
    fam_style = {"perlin": "o", "smooth": "^"}
    cfg_style = {"cpl": "o", "incpl": "s", "full": "D"}
    # marker interior = configuration; colour = category; small/large = family
    for key, e in gp.items():
        cat, cfg = key.split("/")
        for fam in ("perlin", "smooth"):
            ax.scatter(e[fam]["cosine"], e[fam]["norm_ratio"],
                       marker=cfg_style.get(cfg, "o"),
                       s=30 if fam == "perlin" else 16,
                       facecolors=("none" if fam == "smooth" else
                                   S.CAT_COLOR.get(cat, "#56B4E9")),
                       color=S.CAT_COLOR.get(cat, "#56B4E9"),
                       linewidths=1.1)
    ax.set_yscale("log")
    ax.axhline(1, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"cosine alignment cos$(g_C^{\lambda}, g_{\mathrm{seg}})$")
    ax.set_ylabel(r"norm ratio $\|g_C^{\lambda}\|/\|g_{\mathrm{seg}}\|$")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=6, color="0.4",
                          label="cpl (large: Perlin)"),
               plt.Line2D([], [], ls="", marker="s", mfc="none", ms=6,
                          color="0.4", label="incpl"),
               plt.Line2D([], [], ls="", marker="D", mfc="none", ms=6,
                          color="0.4", label="full"),
               plt.Line2D([], [], ls="", marker="o", mfc="none", ms=4,
                          color="0.4", label="small hollow: smooth")]
    handles += [plt.Line2D([], [], ls="", marker="|", ms=8, lw=2,
                           color=S.CAT_COLOR[c], label=c) for c in S.CATS]
    ax.legend(handles=handles, ncol=2, loc="upper right", fontsize=6)
    ax.set_title("(B) measured: shared parameters")


def main():
    S.setup()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    panel_a(axes[0])
    panel_b(axes[1])
    S.save(fig, "fig05_gradients")


if __name__ == "__main__":
    main()
