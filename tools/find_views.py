#!/usr/bin/env python
"""Find Apollo-like viewpoints on a DEM: the camera stands on locally flat ground and looks
toward relief, so the skyline falls in the upper part of the frame (as in most Apollo
surface photos). Prints sites.yaml entries.

  python tools/find_views.py --dem path.tif --upscale 4 --n 3 --device cuda:0
"""
import argparse, sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from despina.dem import load_dem  # noqa: E402
from despina.raycast import Camera, RenderConfig, render_view  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", required=True)
    ap.add_argument("--upscale", type=float, default=4)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=60)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--sky-band", default="0.15,0.40", help="target mean skyline row as a fraction of height")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    lo, hi = map(float, a.sky_band.split(","))
    dem = load_dem(a.dem, upscale=a.upscale)
    h = np.nan_to_num(dem.elev, nan=np.nanmean(dem.elev))
    gy, gx = np.gradient(h, dem.res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope = ndimage.uniform_filter(slope, 9)        # flat over roughly 9 cells
    H, W = h.shape
    m = int(0.15 * min(H, W))
    inner = np.zeros_like(slope, bool); inner[m:H - m, m:W - m] = True
    ok = inner & (slope < np.percentile(slope[inner], 10))
    rng = np.random.default_rng(a.seed)
    cand = np.argwhere(ok)
    cand = cand[rng.choice(len(cand), min(a.candidates, len(cand)), replace=False)]
    cfg = RenderConfig(supersample=1, shadows=False)
    scored = []
    for r, c in cand:
        for hd in range(0, 360, 30):
            cam = Camera(row=float(r), col=float(c), heading_deg=hd, width=a.size, height=a.size)
            try:
                out = render_view(dem, cam, cfg, device=a.device)
            except ValueError:
                continue
            sky = out.skyline / a.size
            if not np.isfinite(sky).all():
                continue
            mean, rough = float(sky.mean()), float(np.std(np.diff(sky)) + sky.std())
            if lo <= mean <= hi:
                scored.append((rough, r, c, hd, mean, float(slope[r, c])))
    # prefer varied skylines (hills), one view per location
    scored.sort(reverse=True)
    picked, used = [], []
    for rough, r, c, hd, mean, sl in scored:
        if all(np.hypot(r - r2, c - c2) > 0.1 * min(H, W) for r2, c2 in used):
            picked.append((r, c, hd, mean, sl, rough)); used.append((r, c))
        if len(picked) == a.n:
            break
    for i, (r, c, hd, mean, sl, rough) in enumerate(picked):
        print(f"  - name: flat_view_{i}   # local slope {sl:.1f} deg, skyline at {100*mean:.0f}% height")
        print(f"    camera: {{row_frac: {r / H:.4f}, col_frac: {c / W:.4f}, heading_deg: {hd}, pitch_deg: 0, hfov_deg: 49, eye_height_m: 1.6}}")


if __name__ == "__main__":
    main()
