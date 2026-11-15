"""DEM loading, upscaling and surface normals (DESPINA Group II, items 10 and 11).

All rendering happens in a local metric frame: DEM pixel (row, col) maps to
x = col * res (east), y = -row * res (north), z = elevation in metres.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

BODY_RADIUS_M = {"moon": 1_737_400.0, "earth": 6_371_000.0, "mars": 3_389_500.0}


@dataclass
class DEM:
    elev: np.ndarray              # (H, W) float32, NaN where nodata
    res: float                    # metres per pixel (square pixels assumed)
    transform: object = None      # rasterio Affine of the (possibly upscaled) grid
    crs: object = None
    body_radius: float = BODY_RADIUS_M["moon"]
    albedo: Optional[np.ndarray] = None   # optional orbital mosaic on the same grid, [0,1]
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.elev.shape

    # ---- coordinate helpers -------------------------------------------------
    def map_to_pix(self, x_map: float, y_map: float):
        """Map (CRS) coordinates to fractional (row, col) on this grid."""
        inv = ~self.transform
        col, row = inv * (x_map, y_map)
        return row - 0.5, col - 0.5   # pixel-centre convention

    def lonlat_to_pix(self, lon: float, lat: float):
        """Planetocentric lon/lat (deg) to fractional (row, col). Needs pyproj."""
        from pyproj import CRS, Transformer
        crs = CRS.from_user_input(self.crs.to_wkt() if hasattr(self.crs, "to_wkt") else self.crs)
        geo = crs.geodetic_crs
        tr = Transformer.from_crs(geo, crs, always_xy=True)
        x, y = tr.transform(lon, lat)
        return self.map_to_pix(x, y)

    def height_at(self, row: float, col: float) -> float:
        t = torch.from_numpy(np.nan_to_num(self.elev, nan=-1e9)).float()
        return float(bilinear(t, torch.tensor([row]), torch.tensor([col]))[0])


def load_dem(path: str, upscale: float = 1.0, body: str = "moon",
             albedo_path: Optional[str] = None, band: int = 1) -> DEM:
    """Load a GeoTIFF DEM with rasterio and optionally upscale it.

    The paper upscales LRO DEMs before ray casting (citing Liu et al. 2020b).
    We use bicubic interpolation of the height field, which removes the
    stair-step artefacts that dominate at coarse resolution; any learned
    super-resolution model can be dropped in by replacing `upscale_elevation`.
    """
    import rasterio
    with rasterio.open(path) as ds:
        a = ds.read(band).astype(np.float32)
        nod = ds.nodata
        if nod is not None:
            a[a == nod] = np.nan
        scale = 1.0
        try:  # some LOLA products store km or need scale factors
            scale = float(ds.scales[band - 1] or 1.0)
            off = float(ds.offsets[band - 1] or 0.0)
            a = a * scale + off
        except Exception:
            pass
        transform, crs = ds.transform, ds.crs
        res = float(abs(transform.a))
        if abs(abs(transform.e) - res) / res > 1e-3:
            raise ValueError("Non-square DEM pixels are not supported; reproject first.")
        radius = BODY_RADIUS_M.get(body, BODY_RADIUS_M["moon"])
        try:
            radius = float(crs.to_dict().get("R", radius)) if crs else radius
        except Exception:
            pass
    albedo = None
    if albedo_path:
        albedo = load_albedo_on_grid(albedo_path, transform, a.shape, crs)
    dem = DEM(elev=a, res=res, transform=transform, crs=crs, body_radius=radius,
              albedo=albedo, meta={"source": path})
    if upscale and upscale != 1.0:
        dem = upscale_dem(dem, upscale)
    return dem


def load_albedo_on_grid(path, transform, shape, crs):
    """Reproject an orbital mosaic onto the DEM grid (for the classical drape baseline)."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    out = np.zeros(shape, np.float32)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.bilinear)
    lo, hi = np.percentile(out[out > 0], [1, 99]) if (out > 0).any() else (0, 1)
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)


def upscale_elevation(a: np.ndarray, factor: float) -> np.ndarray:
    t = torch.from_numpy(a)[None, None]
    nanmask = torch.isnan(t)
    filled = torch.where(nanmask, torch.nanmean(t) if (~nanmask).any() else torch.zeros(()), t)
    up = F.interpolate(filled, scale_factor=factor, mode="bicubic", align_corners=False)
    m = F.interpolate(nanmask.float(), scale_factor=factor, mode="nearest") > 0.5
    up[m] = float("nan")
    return up[0, 0].numpy()


def upscale_dem(dem: DEM, factor: float) -> DEM:
    from rasterio.transform import Affine
    elev = upscale_elevation(dem.elev, factor)
    tr = dem.transform * Affine.scale(1.0 / factor) if dem.transform is not None else None
    alb = None
    if dem.albedo is not None:
        alb = F.interpolate(torch.from_numpy(dem.albedo)[None, None], scale_factor=factor,
                            mode="bilinear", align_corners=False)[0, 0].numpy()
    return DEM(elev=elev, res=dem.res / factor, transform=tr, crs=dem.crs,
               body_radius=dem.body_radius, albedo=alb,
               meta={**dem.meta, "upscale": factor})


def synthetic_dem(size=512, res=5.0, seed=0) -> DEM:
    """Procedural cratered terrain, used by the unit tests and as a demo."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    h = np.zeros((size, size), np.float32)
    # fractal noise
    for octave in range(6):
        f = 2 ** octave
        n = rng.standard_normal((4 * f + 1, 4 * f + 1)).astype(np.float32)
        n = F.interpolate(torch.from_numpy(n)[None, None], size=(size, size), mode="bicubic",
                          align_corners=True)[0, 0].numpy()
        h += n * 60.0 / f ** 1.1
    # craters
    for _ in range(40):
        cx, cy = rng.uniform(0, size, 2)
        r = rng.uniform(5, 60)
        d = np.hypot(xx - cx, yy - cy) / r
        bowl = np.where(d < 1, -0.25 * r * res * 0.1 * (1 - d ** 2), 0)
        rim = 0.06 * r * res * 0.1 * np.exp(-((d - 1) / 0.15) ** 2)
        h += bowl + rim
    from rasterio.transform import Affine
    return DEM(elev=h, res=res, transform=Affine(res, 0, 0, 0, -res, 0), crs=None,
               meta={"source": "synthetic"})


# ---- normals (Eq. 1) ---------------------------------------------------------

def surface_normals(elev: np.ndarray, res: float) -> np.ndarray:
    """Eq. 1: n(x, y) = normalize(dh_x, dh_y, 1) with central differences.

    Written with the sign convention that gives an upward, outward normal in
    the local frame (x east, y north): n = normalize(-dh/dx, -dh/dy, 1).
    Returns (H, W, 3) float32.
    """
    h = np.nan_to_num(elev, nan=np.nanmean(elev))
    dhdx = np.zeros_like(h)
    dhdy = np.zeros_like(h)
    dhdx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2 * res)
    dhdx[:, 0], dhdx[:, -1] = dhdx[:, 1], dhdx[:, -2]
    # rows increase southwards, so north-gradient has the opposite sign
    dhdy[1:-1, :] = -(h[2:, :] - h[:-2, :]) / (2 * res)
    dhdy[0, :], dhdy[-1, :] = dhdy[1, :], dhdy[-2, :]
    n = np.stack([-dhdx, -dhdy, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n.astype(np.float32)


def shadow_map(elev: np.ndarray, res: float, sun_az_deg: float, sun_el_deg: float,
               max_dist_m: Optional[float] = None, device="cpu") -> np.ndarray:
    """Cast shadows by marching from every cell toward the sun. 1 = lit, 0 = shadow."""
    h = torch.from_numpy(np.nan_to_num(elev, nan=-1e9)).float().to(device)
    H, W = h.shape
    az, el = np.deg2rad(sun_az_deg), np.deg2rad(sun_el_deg)
    # direction toward the sun in pixel units: east = +col, north = -row
    dcol, drow = np.sin(az), -np.cos(az)
    tan_el = np.tan(el)
    if max_dist_m is None:
        max_dist_m = res * max(H, W)
    rows = torch.arange(H, device=device).float()[:, None].expand(H, W)
    cols = torch.arange(W, device=device).float()[None, :].expand(H, W)
    lit = torch.ones(H, W, dtype=torch.bool, device=device)
    step = 1.0
    s = step
    while s * res < max_dist_m:
        r = rows + drow * s
        c = cols + dcol * s
        hs = bilinear(h, r.flatten(), c.flatten()).view(H, W)
        lit &= ~(hs > h + s * res * tan_el)
        s += step
        step *= 1.02  # coarser further out
    return lit.float().cpu().numpy()


def bilinear(img: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
             fill: float = float("nan")) -> torch.Tensor:
    """Bilinear sample a 2-D tensor at fractional (row, col); out-of-bounds -> fill."""
    H, W = img.shape
    r0 = torch.floor(rows)
    c0 = torch.floor(cols)
    fr = rows - r0
    fc = cols - c0
    r0 = r0.long()
    c0 = c0.long()
    valid = (r0 >= 0) & (c0 >= 0) & (r0 < H - 1) & (c0 < W - 1)
    r0c = r0.clamp(0, H - 2)
    c0c = c0.clamp(0, W - 2)
    flat = img.reshape(-1)
    i00 = r0c * W + c0c
    v00 = flat[i00]
    v01 = flat[i00 + 1]
    v10 = flat[i00 + W]
    v11 = flat[i00 + W + 1]
    out = (v00 * (1 - fr) * (1 - fc) + v01 * (1 - fr) * fc
           + v10 * fr * (1 - fc) + v11 * fr * fc)
    return torch.where(valid, out, torch.full_like(out, fill))
