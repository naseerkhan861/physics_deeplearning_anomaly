"""F4: defect-family response and coupling-weight intervention (round-9).

Panels A-C: screw family sweeps (contrast / frequency / binary mask
smoothing) for the
four reconstruction-off configurations. Panel D: absolute Perlin and
smooth AUROC versus coupling weight for both seeds, with the seed-42
CPU-trained C-off reference visually distinguished.
"""

from . import style as S
import matplotlib.pyplot as plt
import numpy as np

FAM_ARMS = ("base", "in", "cpl", "incpl")
CONTRASTS = ("0.10", "0.18", "0.25", "0.40")
FREQ = ("frequency:low", "frequency:mid", "frequency:high")
SOFT = ("softness:hard", "softness:mid", "softness:soft")
LVL = {"contrast": CONTRASTS, "frequency:": FREQ, "softness:": SOFT}


def fam_panel(ax, factor_keys, title):
    sw = S.load("defect_family_sweep.json")["sweeps"]["screw"]
    x = np.arange(len(factor_keys))
    for arm in FAM_ARMS:
        ys = [sw[arm][k]["image_auroc"] for k in factor_keys]
        ax.plot(x, ys, "-o", ms=3, lw=1.0, color=S.ARM[arm], label=arm)
    ax.set_xticks(x)
    ax.set_xticklabels([k.split(":")[-1] if ":" in k else k for k in factor_keys])
    ax.set_ylim(0.3, 1.05)
    ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
    ax.set_title(title)


def weight_panel(ax):
    s42 = S.load("lamc_screw_smooth.json")
    s0 = S.load("lamc_screw_smooth_s0.json")

    def series(rec, coff_key, tag, mark):
        lam, per, smo = [], [], []
        reused = None
        for k, v in rec.items():
            if k == "protocol":
                continue
            if k.startswith("C-off"):
                reused = v
                continue
            lam.append(float(k))
            per.append(v["perlin"])
            smo.append(v["smooth"])
        o = np.argsort(lam)
        lam, per, smo = np.array(lam)[o], np.array(per)[o], np.array(smo)[o]
        ax.plot(lam, per, "-" + mark, ms=4, lw=1.0, color="#0072B2",
                label=f"Perlin (seed {tag})")
        ax.plot(lam, smo, "--" + mark, ms=4, lw=1.0, color="#D55E00",
                label=f"smooth (seed {tag})")
        if reused:  # C-off references at a labelled position left of x=0
            offx = -0.04 if tag == "s0" else 0.04
            hollow = "none" if tag == "42" else None
            ax.plot([offx], [reused["perlin"]], marker=mark, ms=6, ls="",
                    color="#0072B2", mfc=hollow, mec="#0072B2")
            ax.plot([offx], [reused["smooth"]], marker=mark, ms=6, ls="",
                    color="#D55E00", mfc=hollow, mec="#D55E00")
    series(s0, "C-off (base, cuda)", "s0", "s")
    series(s42, "C-off (base)", "42", "o")
    ax.set_xlabel(r"coupling weight $\lambda_C$")
    ax.set_ylabel("image AUROC")
    ax.set_ylim(0.3, 1.05)
    ax.set_xticks([0.1, 0.25, 0.5, 1.0])
    ax.legend(ncol=2, loc="lower left", fontsize=6.5)
    ax.set_title("(D) weight intervention (screw)")


def main():
    S.setup()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    fam_panel(axes[0, 0], CONTRASTS, "(A) contrast")
    fam_panel(axes[0, 1], FREQ, "(B) defect frequency")
    fam_panel(axes[1, 0], SOFT, "(C) binary mask smoothing")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    weight_panel(axes[1, 1])
    axes[1, 0].legend(handles, labels, ncol=2, loc="lower left", fontsize=7)
    S.save(fig, "fig04_interventions")


if __name__ == "__main__":
    main()
