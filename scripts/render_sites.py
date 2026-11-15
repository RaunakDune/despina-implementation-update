#!/usr/bin/env python
"""Group II for every evaluation site: DEM -> inverse depth, simulated render, soft edges,
skyline; plus the photo-side conditions (DPT depth, PiDiNet) used by the baseline.

  python scripts/render_sites.py --sites configs/sites.yaml --out runs/sites --device cuda:0
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from despina import constraints as C  # noqa: E402
from despina.raycast import render_view  # noqa: E402
from despina.sites import load_sites, site_camera, site_dem, site_render_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--only", default=None, help="comma-separated site names")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    for site in load_sites(a.sites):
        if only and site["name"] not in only:
            continue
        od = Path(a.out) / site["name"]; od.mkdir(parents=True, exist_ok=True)
        dem = site_dem(site)
        cam = site_camera(site, dem, a.size, a.size)
        res = render_view(dem, cam, site_render_config(site), device=a.device)
        Image.fromarray(res.inv_depth).save(od / "dem_depth.png")
        Image.fromarray(res.sim_image).save(od / "dem_sim.png")
        Image.fromarray(C.soft_edges(res.sim_image, a.device)).save(od / "dem_softedge.png")
        np.save(od / "dem_skyline.npy", res.skyline)
        np.save(od / "dem_range.npy", res.range_m.astype(np.float16))
        ref = site.get("reference")
        if ref:
            photo = Image.open(ref).convert("RGB").resize((a.size, a.size), Image.BICUBIC)
            photo.save(od / "photo.png")
            Image.fromarray(C.dpt_depth(photo, a.device)).save(od / "photo_depth.png")
            Image.fromarray(C.soft_edges(photo, a.device)).save(od / "photo_softedge.png")
        prompt = site.get("prompt") or C.dem_prompt(site.get("category", "flat"))
        meta = {"site": site, "prompt": prompt, "d_near": res.d_near, "d_far": float(res.d_far.max()),
                "camera": res.camera, "terrain_frac": float(res.terrain_mask.mean())}
        json.dump(meta, open(od / "meta.json", "w"), indent=1, default=str)
        print(site["name"], "rendered", json.dumps({k: meta[k] for k in ("d_near", "d_far", "terrain_frac")}))


if __name__ == "__main__":
    main()
