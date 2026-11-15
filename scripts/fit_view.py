#!/usr/bin/env python
"""Estimate heading / pitch / hfov for a reference photo by matching its skyline
to DEM skylines (coarse-to-fine grid search). Use it when the exact camera pose of
an Apollo or Chang'e-5 photo is unknown; the position must still be given.

  python scripts/fit_view.py --sites configs/sites.yaml --site falcon_lm --device cuda:0
"""
import argparse, itertools, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from despina.metrics import extract_skyline, horizon_rmse  # noqa: E402
from despina.raycast import render_view  # noqa: E402
from despina.sites import load_sites, site_camera, site_dem, site_render_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--hfovs", default="35,40,45,49,55,60")
    a = ap.parse_args()
    site = next(s for s in load_sites(a.sites) if s["name"] == a.site)
    dem = site_dem(site)
    photo = Image.open(site["reference"]).convert("L").resize((a.size, a.size), Image.BICUBIC)
    target = extract_skyline(np.array(photo))
    cfg = site_render_config(site, supersample=1, shadows=False)
    base = site_camera(site, dem, a.size, a.size)

    def score(h, p, f):
        base.heading_deg, base.pitch_deg, base.hfov_deg = h, p, f
        try:
            sky = render_view(dem, base, cfg, device=a.device).skyline
        except ValueError:
            return np.inf
        return horizon_rmse(np.nan_to_num(sky, nan=0.0), target)

    best = (np.inf, None)
    for h, p, f in itertools.product(range(0, 360, 5), (-6, -3, 0, 3, 6), map(float, a.hfovs.split(","))):
        s = score(h, p, f)
        if s < best[0]:
            best = (s, (h, p, f))
    s0, (h0, p0, f0) = best
    for h, p, f in itertools.product(np.arange(h0 - 5, h0 + 5.1, 1.0), np.arange(p0 - 3, p0 + 3.1, 0.5),
                                     np.arange(f0 - 4, f0 + 4.1, 1.0)):
        s = score(h % 360, p, f)
        if s < best[0]:
            best = (s, (h % 360, p, f))
    s, (h, p, f) = best
    print(f"best heading={h:.1f} pitch={p:.1f} hfov={f:.1f}  skyline RMSE={s:.2f}px @ {a.size}px")


if __name__ == "__main__":
    main()
