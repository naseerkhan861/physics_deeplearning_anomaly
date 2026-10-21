#!/usr/bin/env python3
"""
make_construct.py — Build the CONSTRUCT synthetic defect dataset.

CONSTRUCT: MVTec-style layout + DRAEM-style Perlin masks with controlled
severity, for controlled testing of a physics-guided defect-detection approach.

Output: /mnt/agents/work/data/construct/
  <class>/{train/good, test/good, test/defective, test/masks}/
  splits.csv
  physics_stats.json  (+ per-class mean images under physics/)

Deterministic: numpy random seed 42. Image size 128x128 grayscale.
"""

import csv
import json
import os

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
SIZE = 128
OUT_ROOT = "/mnt/agents/work/data/construct"
CLASSES = ["brushed_steel", "cast_plate", "rolled_sheet"]
N_TRAIN_GOOD = 150
N_TEST_GOOD = 40
N_TEST_DEFECT = 40
SEVERITIES = ["mild"] * 13 + ["medium"] * 14 + ["severe"] * 13
DEFECT_TYPES = ["scratch", "pit", "patch"]

# severity -> (area_frac_range, contrast_delta_range)
SEVERITY_CFG = {
    "mild":   ((0.005, 0.020), (10.0, 25.0)),
    "medium": ((0.020, 0.050), (25.0, 45.0)),
    "severe": ((0.050, 0.100), (45.0, 70.0)),
}

np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Base texture generators (defect-free, "normal physics" samples:
# smooth gradients, bounded curvature)
# ---------------------------------------------------------------------------
def smooth(img, k=3):
    """Gaussian smooth to bound curvature (physics-consistent)."""
    return cv2.GaussianBlur(img, (k, k), 0)


def gen_brushed_steel():
    """Horizontal brushed texture: banded noise with horizontal streaks."""
    mean = np.random.uniform(120, 160)
    noise = np.random.randn(SIZE, SIZE).astype(np.float32)
    # strong horizontal blur -> streaks, light vertical blur -> banding
    streaks = cv2.GaussianBlur(noise, (31, 3), 0)
    fine = cv2.GaussianBlur(np.random.randn(SIZE, SIZE).astype(np.float32), (7, 1), 0)
    img = streaks * 6.0 + fine * 2.0
    img = smooth(img, 3)  # bound curvature
    img = img - img.mean() + mean
    return img


def gen_cast_plate():
    """Isotropic speckle/grain: low-frequency blobs + fine grain."""
    mean = np.random.uniform(100, 140)
    blobs = cv2.GaussianBlur(np.random.randn(SIZE, SIZE).astype(np.float32), (17, 17), 0)
    grain = cv2.GaussianBlur(np.random.randn(SIZE, SIZE).astype(np.float32), (3, 3), 0)
    img = blobs * 9.0 + grain * 2.5
    img = smooth(img, 3)
    img = img - img.mean() + mean
    return img


def gen_rolled_sheet():
    """Low-frequency gradient shading + subtle vertical sinusoidal ridges."""
    mean = np.random.uniform(110, 150)
    # low-frequency shading from heavily blurred noise + linear gradient
    shade = cv2.GaussianBlur(np.random.randn(SIZE, SIZE).astype(np.float32), (31, 31), 0) * 7.0
    gx = np.linspace(-1, 1, SIZE, dtype=np.float32)[None, :] * np.random.uniform(-8, 8)
    gy = np.linspace(-1, 1, SIZE, dtype=np.float32)[:, None] * np.random.uniform(-8, 8)
    # vertical sinusoidal rolling ridges (vary slowly along x)
    period = np.random.uniform(5.0, 9.0)
    phase = np.random.uniform(0, 2 * np.pi)
    ridges = 2.5 * np.sin(2 * np.pi * np.arange(SIZE, dtype=np.float32)[None, :] / period + phase)
    img = shade + gx + gy + ridges
    img = smooth(img, 3)
    img = img - img.mean() + mean
    return img


GEN = {
    "brushed_steel": gen_brushed_steel,
    "cast_plate": gen_cast_plate,
    "rolled_sheet": gen_rolled_sheet,
}


def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# DRAEM-style Perlin / fractal-noise anomaly mask
# ---------------------------------------------------------------------------
def fractal_noise(size=SIZE, octaves=(2, 4, 8, 16), persistence=0.6):
    """Multi-octave value noise: random coarse grids upsampled and summed."""
    out = np.zeros((size, size), dtype=np.float32)
    amp, total = 1.0, 0.0
    for o in octaves:
        g = np.random.rand(o, o).astype(np.float32)
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_CUBIC)
        out += amp * g
        total += amp
        amp *= persistence
    out /= total
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    return out


def perlin_mask(area_frac):
    """Threshold fractal noise at a quantile so mask area ~= area_frac.
    Edges smoothed by blurring the noise before thresholding."""
    noise = fractal_noise()
    noise = cv2.GaussianBlur(noise, (5, 5), 0)  # smooth edges
    thr = np.quantile(noise, 1.0 - area_frac)
    mask = (noise >= thr).astype(np.uint8) * 255
    # clean tiny specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        keep = np.zeros_like(mask)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        for i in range(1, n):
            if i == largest or stats[i, cv2.CC_STAT_AREA] >= 12:
                keep[lab == i] = 255
        mask = keep
    return mask


# ---------------------------------------------------------------------------
# Defect renderers: (img_float, mask) -> defective img_float
# ---------------------------------------------------------------------------
def apply_scratch(img, mask, delta, cls, severity):
    """Thin bright/dark elongated line violating base texture direction.
    Severity scales scratch length/width (area) and contrast delta."""
    h, w = img.shape
    # draw a thin line; force it inside / through the perlin region when possible
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        cx, cy = int(xs.mean()), int(ys.mean())
    else:
        cx, cy = w // 2, h // 2
    len_rng = {"mild": (0.35, 0.6), "medium": (0.6, 0.9), "severe": (0.9, 1.4)}[severity]
    wid_rng = {"mild": (1, 1), "medium": (1, 2), "severe": (2, 3)}[severity]
    length = int(np.random.uniform(*len_rng) * w)
    width = np.random.randint(wid_rng[0], wid_rng[1] + 1)
    # angle must violate base direction: brushed steel is horizontal -> avoid ~0 deg;
    # rolled sheet has vertical ridges -> avoid ~90 deg; cast plate isotropic -> any
    if cls == "brushed_steel":
        angle = np.deg2rad(np.random.uniform(35, 145))
    elif cls == "rolled_sheet":
        angle = np.deg2rad(np.random.choice([np.random.uniform(-30, 30),
                                             np.random.uniform(150, 210)]))
    else:
        angle = np.deg2rad(np.random.uniform(0, 180))
    dx, dy = np.cos(angle) * length / 2, np.sin(angle) * length / 2
    p1 = (int(np.clip(cx - dx, 0, w - 1)), int(np.clip(cy - dy, 0, h - 1)))
    p2 = (int(np.clip(cx + dx, 0, w - 1)), int(np.clip(cy + dy, 0, h - 1)))
    line = np.zeros((h, w), np.uint8)
    cv2.line(line, p1, p2, 255, width)
    # restrict to perlin region when the overlap is usable, else use the line
    region = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    smask = cv2.bitwise_and(line, region)
    if (smask > 0).sum() < 40:  # tiny/degenerate overlap -> use the full line
        smask = line
    sign = np.random.choice([-1.0, 1.0])  # bright or dark scratch
    soft = cv2.GaussianBlur(smask.astype(np.float32) / 255.0, (3, 3), 0)
    out = img + sign * delta * soft
    return out, (soft > 0.25).astype(np.uint8) * 255


def apply_pit(img, mask, delta):
    """Dark blob(s) with bright rim — local material removal."""
    k = np.ones((3, 3), np.uint8)
    inner = cv2.erode(mask, k, iterations=1)
    rim = cv2.dilate(mask, k, iterations=1) - mask
    inner_f = cv2.GaussianBlur(inner.astype(np.float32) / 255.0, (3, 3), 0)
    rim_f = cv2.GaussianBlur(rim.astype(np.float32) / 255.0, (3, 3), 0)
    out = img - delta * inner_f + 0.5 * delta * rim_f
    return out, mask


def apply_patch(img, mask, delta):
    """Region of shifted mean intensity with softened texture — contamination."""
    a = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (5, 5), 0)[..., None].squeeze()
    blurred = cv2.GaussianBlur(img, (9, 9), 0)          # softened texture
    shifted = blurred + np.random.choice([-1.0, 1.0]) * delta
    out = img * (1 - a) + shifted * a
    return out, (a > 0.25).astype(np.uint8) * 255


def make_defect(img, cls, defect_type, severity):
    (a_lo, a_hi), (d_lo, d_hi) = SEVERITY_CFG[severity]
    area_frac = float(np.random.uniform(a_lo, a_hi))
    delta = float(np.random.uniform(d_lo, d_hi))
    if defect_type == "scratch":
        mask = perlin_mask(min(area_frac * 3.0, 0.25))  # region to route the line through
        out, mask = apply_scratch(img, mask, delta, cls, severity)
    else:
        mask = perlin_mask(area_frac)
        if defect_type == "pit":
            out, mask = apply_pit(img, mask, delta)
        else:
            out, mask = apply_patch(img, mask, delta)
    # ensure non-empty mask
    if mask.sum() == 0:
        yy, xx = SIZE // 2, SIZE // 2
        mask = np.zeros((SIZE, SIZE), np.uint8)
        cv2.circle(mask, (xx, yy), 3, 255, -1)
        out = img.copy()
        out[mask > 0] -= delta
    actual_area_frac = float((mask > 0).mean())
    return out, mask, actual_area_frac, delta


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def save_png(img_u8, path):
    """Save with retry — shared mount can raise transient FileExistsError/IOError."""
    import time
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im = Image.fromarray(img_u8, mode="L")
    for attempt in range(10):
        try:
            im.save(path)
            return
        except (FileExistsError, OSError):
            time.sleep(0.05 * (attempt + 1))
            os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)  # final attempt raises if still failing


def grad_stats(imgs):
    """mean/std of |grad| and Laplacian over a stack of float images."""
    g_mags, laps = [], []
    for im in imgs:
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
        g_mags.append(np.sqrt(gx ** 2 + gy ** 2))
        laps.append(cv2.Laplacian(im, cv2.CV_32F, ksize=3))
    g = np.concatenate([x.ravel() for x in g_mags])
    l = np.concatenate([x.ravel() for x in laps])
    return {
        "grad_abs_mean": float(g.mean()), "grad_abs_std": float(g.std()),
        "laplacian_mean": float(l.mean()), "laplacian_std": float(l.std()),
    }


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def main():
    rows = []  # splits.csv rows
    physics = {}

    for cls in CLASSES:
        train_dir = os.path.join(OUT_ROOT, cls, "train", "good")
        tg_dir = os.path.join(OUT_ROOT, cls, "test", "good")
        td_dir = os.path.join(OUT_ROOT, cls, "test", "defective")
        tm_dir = os.path.join(OUT_ROOT, cls, "test", "masks")
        for d in (train_dir, tg_dir, td_dir, tm_dir):
            os.makedirs(d, exist_ok=True)

        # ---- train/good ----
        train_imgs = []
        for i in range(N_TRAIN_GOOD):
            img = GEN[cls]()
            train_imgs.append(img)
            fn = f"{cls}_train_{i:03d}.png"
            save_png(to_uint8(img), os.path.join(train_dir, fn))
            rows.append([cls, "train", f"{cls}/train/good/{fn}", "good", "", "", 0.0, 0.0])

        # ---- test/good ----
        for i in range(N_TEST_GOOD):
            img = GEN[cls]()
            fn = f"{cls}_test_good_{i:03d}.png"
            save_png(to_uint8(img), os.path.join(tg_dir, fn))
            rows.append([cls, "test", f"{cls}/test/good/{fn}", "good", "", "", 0.0, 0.0])

        # ---- test/defective + masks ----
        sev_cycle = SEVERITIES[:]  # deterministic 13/14/13
        for i in range(N_TEST_DEFECT):
            base = GEN[cls]()
            dtype = DEFECT_TYPES[i % len(DEFECT_TYPES)]
            sev = sev_cycle[i % len(sev_cycle)]
            dimg, mask, area_frac, delta = make_defect(base, cls, dtype, sev)
            fn = f"{cls}_test_defect_{i:03d}.png"
            save_png(to_uint8(dimg), os.path.join(td_dir, fn))
            save_png(mask, os.path.join(tm_dir, fn))
            rows.append([cls, "test", f"{cls}/test/defective/{fn}", "defective",
                         dtype, sev, round(area_frac, 5), round(delta, 2)])

        # ---- physics prior: mean image + gradient statistics ----
        stack = np.stack(train_imgs).astype(np.float32)
        mean_img = stack.mean(axis=0)
        phys_dir = os.path.join(OUT_ROOT, "physics")
        os.makedirs(phys_dir, exist_ok=True)
        mean_path = os.path.join(phys_dir, f"{cls}_mean.png")
        save_png(to_uint8(mean_img), mean_path)
        stats = grad_stats(list(stack))
        stats["mean_image"] = os.path.relpath(mean_path, OUT_ROOT)
        stats["intensity_mean"] = float(stack.mean())
        stats["intensity_std"] = float(stack.std())
        physics[cls] = stats

    # ---- splits.csv ----
    csv_path = os.path.join(OUT_ROOT, "splits.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "split", "filename", "label", "defect_type",
                    "severity", "mask_area_frac", "contrast_delta"])
        w.writerows(rows)

    # ---- physics_stats.json ----
    with open(os.path.join(OUT_ROOT, "physics_stats.json"), "w") as f:
        json.dump({"seed": SEED, "image_size": SIZE, "classes": physics}, f, indent=2)

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    print("=== CONSTRUCT sanity checks ===")
    ok = True
    total = 0
    for cls in CLASSES:
        for sub, expect in (("train/good", N_TRAIN_GOOD), ("test/good", N_TEST_GOOD),
                            ("test/defective", N_TEST_DEFECT), ("test/masks", N_TEST_DEFECT)):
            files = [f for f in os.listdir(os.path.join(OUT_ROOT, cls, sub)) if f.endswith(".png")]
            status = "OK" if len(files) == expect else "FAIL"
            ok &= len(files) == expect
            total += len(files) if "masks" not in sub else 0
            print(f"  {cls}/{sub}: {len(files)}/{expect} [{status}]")
        # masks non-empty + align with images
        mdir = os.path.join(OUT_ROOT, cls, "test", "masks")
        for mf in sorted(os.listdir(mdir)):
            m = np.array(Image.open(os.path.join(mdir, mf)))
            assert m.shape == (SIZE, SIZE), f"mask size mismatch {mf}"
            vals = set(np.unique(m).tolist())
            assert vals <= {0, 255}, f"mask not binary {mf}: {vals}"
            if not (m > 0).any():
                print(f"  EMPTY MASK: {cls}/{mf}"); ok = False
        # PIL can open a sample
        s = Image.open(os.path.join(OUT_ROOT, cls, "test", "defective",
                                    sorted(os.listdir(os.path.join(OUT_ROOT, cls, "test", "defective")))[0]))
        s.load()
        assert s.size == (SIZE, SIZE) and s.mode == "L"
        print(f"  {cls}: PIL sample open OK, masks binary & non-empty OK")
    # csv row count
    with open(csv_path) as f:
        n_rows = sum(1 for _ in f) - 1
    exp_rows = len(CLASSES) * (N_TRAIN_GOOD + N_TEST_GOOD + N_TEST_DEFECT)
    print(f"  splits.csv rows: {n_rows}/{exp_rows} [{'OK' if n_rows == exp_rows else 'FAIL'}]")
    ok &= n_rows == exp_rows
    print(f"  total images (excl. masks/mean): {total}")
    print(f"RESULT: {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
