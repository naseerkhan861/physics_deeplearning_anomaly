"""F7: full-page qualitative grid (all eight categories).

Each source panel is sliced into its four sub-images (input, ground
truth, PGDN anomaly map, prior z).  Tiles are content-trimmed -- white
borders, inherited subplot furniture, and inter-image gap fragments
are removed -- then padded to a common square, so the recomposed 8x4
grid has one uniform labeling system.  Headers sit on the first-row
axes.  Sources are the genuine rendered panels; the KolektorSDD2 row
carries the official annotation (see artifact/qualitative_ledger.md).
"""

from . import style as S
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CASES = [
    ("qual_ksdd2", "(a)", "KolektorSDD2"),
    ("qual_mtd", "(b)", "MTD (blowhole)"),
    ("qual_screw", "(c)", "MVTec AD screw"),
    ("qual_loco_screw_bag", "(d)", "MVTec LOCO screw bag"),
    ("qual_bottle", "(e)", "MVTec AD bottle"),
    ("qual_capsule", "(f)", "MVTec AD capsule"),
    ("qual_visa_cashew", "(g)", "VisA cashew (normal)"),
    ("qual_visa_candle", "(h)", "VisA candle (normal)"),
]
COLS = ("input", "ground truth", "PGDN anomaly map", "prior residual $z$")
WHITE = 242  # threshold: brighter than this counts as background


def _trim(arr, pad=4):
    """Crop to image content: remove white borders and any gap fragments
    attached to the edges; keep everything darker than the threshold."""
    g = arr.mean(axis=2)
    content = g < WHITE
    # drop thin noise: require a run of content per row/col
    rows = np.where(content.sum(axis=1) > 4)[0]
    cols = np.where(content.sum(axis=0) > 4)[0]
    if not len(rows) or not len(cols):
        return arr
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + 1 + pad, arr.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + 1 + pad, arr.shape[1])
    return arr[r0:r1, c0:c1]


def _square(arr, size):
    """Place a tile centrally on a white square canvas of `size`."""
    h, w = arr.shape[:2]
    out = np.full((size, size, 3), 255, np.uint8)
    scale = min(size / h, size / w)
    nh, nw = max(int(h * scale), 1), max(int(w * scale), 1)
    im = Image.fromarray(arr).resize((nw, nh), Image.LANCZOS)
    out[(size - nh) // 2:(size - nh) // 2 + nh,
        (size - nw) // 2:(size - nw) // 2 + nw] = np.asarray(im)
    return out


def slice_panel(img):
    """Split a 4-sub-image strip, trimming inherited display furniture."""
    w = img.width
    quarter = w / 4.0
    inset = 3  # discard the white separators between sub-images
    tiles = []
    for i in range(4):
        raw = np.asarray(img.crop((int(i * quarter) + inset, 3,
                                   int((i + 1) * quarter) - inset,
                                   img.height - 3)).convert("RGB"))
        tiles.append(_trim(raw))
    size = max(max(t.shape[0] for t in tiles),
               max(t.shape[1] for t in tiles))
    return [_square(t, size) for t in tiles]


def main():
    S.setup()
    fig = plt.figure(figsize=(7.0, 9.0))
    gs = fig.add_gridspec(8, 4, left=0.115, right=0.985, top=0.955,
                          bottom=0.010, hspace=0.16, wspace=0.06)
    for r, (stem, letter, label) in enumerate(CASES):
        img = Image.open(S.REPO / "figs" / f"{stem}.png")
        subs = slice_panel(img)
        for c, sub in enumerate(subs):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(sub, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("0.45")
                spine.set_linewidth(0.5)
            if c == 0:
                ax.text(-0.055, 0.5, f"{letter} {label}",
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=8.5)
            if r == 0:  # headers on the first-row axes, above clear space
                ax.set_title(COLS[c], fontsize=8.5, pad=5)
    S.save(fig, "fig07_qualitative")


if __name__ == "__main__":
    main()
