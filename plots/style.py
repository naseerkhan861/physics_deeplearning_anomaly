"""Shared plotting style for the analysis figures (round-9 plan F2-F7)."""

from pathlib import Path

import matplotlib as mpl

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "artifact"
OUTDIR = REPO / "figs" / "analysis"

# arm colors (colorblind-safe, grayscale-distinguishable by lightness)
ARM = {"pgdn": "#0072B2", "draem": "#E69F00", "pc": "#009E73",
       "base": "#555555", "in": "#0072B2", "cpl": "#D55E00",
       "incpl": "#CC79A7", "full": "#009E73"}
CATS = ("screw", "capsule", "bottle")
CAT_COLOR = {"screw": "#0072B2", "capsule": "#E69F00", "bottle": "#009E73"}
SEED_MARK = {42: "o", 0: "s", 1: "^"}


def setup():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600,
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "legend.frameon": False, "legend.fontsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
    })


def load(name):
    import json
    return json.loads((ARTIFACT / name).read_text())


def save(fig, stem):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / f"{stem}.pdf")
    fig.savefig(OUTDIR / f"{stem}.png")
    print("wrote", OUTDIR / f"{stem}.pdf")
