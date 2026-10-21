"""Supplementary figure: capsule and bottle defect-family sweeps.

Companion to main-text F4 (screw): the same four reconstruction-off
configurations across contrast / frequency / softness for the other two
factorial categories, from the same defect_family_sweep.json record.
"""

from . import style as S
import matplotlib.pyplot as plt
import numpy as np

FAM_ARMS = ("base", "in", "cpl", "incpl")
PANELS = [("contrast", ("0.10", "0.18", "0.25", "0.40")),
          ("frequency", ("frequency:low", "frequency:mid", "frequency:high")),
          ("mask smoothing", ("softness:hard", "softness:mid", "softness:soft"))]


def main():
    S.setup()
    sw_all = S.load("defect_family_sweep.json")["sweeps"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6))
    for ri, cat in enumerate(("capsule", "bottle")):
        sw = sw_all[cat]
        for ci, (name, keys) in enumerate(PANELS):
            ax = axes[ri, ci]
            for arm in FAM_ARMS:
                ys = [sw[arm][k]["image_auroc"] for k in keys]
                ax.plot(np.arange(len(keys)), ys, "-o", ms=3, lw=1.0,
                        color=S.ARM[arm], label=arm)
            ax.set_ylim(0.3, 1.05)
            ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
            ax.set_xticks(np.arange(len(keys)))
            ax.set_xticklabels([k.split(":")[-1] for k in keys], fontsize=7)
            if ri == 0:
                ax.set_title(name)
            if ci == 0:
                ax.set_ylabel(f"{cat}\nimage AUROC")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="outside lower center")
    S.save(fig, "figS_famsweep")


if __name__ == "__main__":
    main()
