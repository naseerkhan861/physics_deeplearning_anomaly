"""Reproducible entry point for the analysis figures (round-9 plan).

Usage:
    python -m plots.build_all [--artifact-root artifact] [--output figs/analysis]

Builds every figure whose required artifact-backed inputs exist and
reports any that cannot be built; missing required records fail the
build rather than producing placeholder figures.
"""

import argparse
import sys

from . import fig02_regimes, fig03_factorial, fig04_interventions
from . import fig05_gradients, fig06_real_transfer, fig07_qualitative
from . import figS_famsweep
from . import fig_defect_localization, fig_spatial_coupling
from . import style as S

FIGS = {"fig02_regimes": fig02_regimes.main,
        "fig03_factorial": fig03_factorial.main,
        "fig04_interventions": fig04_interventions.main,
        "fig05_gradients": fig05_gradients.main,
        "fig06_real_transfer": fig06_real_transfer.main,
        "fig07_qualitative": fig07_qualitative.main,
        "figS_famsweep": figS_famsweep.main,
        "fig_defect_localization": fig_defect_localization.main,
        "fig_spatial_coupling": fig_spatial_coupling.main}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", default="artifact")
    ap.add_argument("--output", default="figs/analysis")
    args = ap.parse_args(argv)
    S.ARTIFACT = S.REPO / args.artifact_root
    S.OUTDIR = S.REPO / args.output
    failed = []
    for name, fn in FIGS.items():
        try:
            fn()
        except Exception as e:  # noqa: BLE001 -- report, then fail loudly
            print(f"FAILED {name}: {e}", file=sys.stderr)
            failed.append(name)
    if failed:
        raise SystemExit(f"figure build failed for: {', '.join(failed)}")
    print("all figures built:", ", ".join(FIGS))


if __name__ == "__main__":
    main()
