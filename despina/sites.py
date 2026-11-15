"""Site configuration: turn a sites.yaml entry into a DEM + Camera + RenderConfig."""
from __future__ import annotations

from pathlib import Path

import yaml

from .dem import load_dem
from .raycast import Camera, RenderConfig


def load_sites(path):
    path = Path(path)
    cfg = yaml.safe_load(open(path))
    sites = cfg["sites"]
    for s in sites:
        for k in ("dem", "reference", "exclude_mask", "albedo"):
            if s.get(k):
                p = Path(s[k])
                s[k] = str(p if p.is_absolute() else (path.parent / p).resolve())
    return sites


def site_camera(site, dem, width=1024, height=1024) -> Camera:
    c = dict(site["camera"])
    if "lon" in c:
        row, col = dem.lonlat_to_pix(c.pop("lon"), c.pop("lat"))
    elif "x" in c:
        row, col = dem.map_to_pix(c.pop("x"), c.pop("y"))
    elif "row_frac" in c:
        row, col = dem.shape[0] * c.pop("row_frac"), dem.shape[1] * c.pop("col_frac")
    else:
        row, col = c.pop("row"), c.pop("col")
    return Camera(row=row, col=col, width=width, height=height, **c)


def site_render_config(site, **over) -> RenderConfig:
    sun = site.get("sun", {})
    kw = dict(sun_az_deg=sun.get("az", 90.0), sun_el_deg=sun.get("el", 15.0))
    kw.update(site.get("render", {}))
    kw.update(over)
    return RenderConfig(**kw)


def site_dem(site):
    return load_dem(site["dem"], upscale=site.get("upscale", 1.0), body=site.get("body", "moon"),
                    albedo_path=site.get("albedo"))
