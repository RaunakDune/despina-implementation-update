"""Horizon-aware DEM constraint generation (DESPINA Group II, item 11 -> items 12, 13).

Column-wise ray caster: every image column is a vertical plane at a fixed
azimuth, so we sample the terrain profile once per column and resolve all
rows of that column with a running-max (horizon) test. This is exact for a
level camera, cheap on a GPU, and gives sub-pixel skylines.

Implements the paper's components:
  * view-adaptive far plane, Eq. 2:   d_far(x) = (1 + eps) * max_{p in V_x} ||p - c||
  * inverse-depth normalisation, Eq. 3 (near = white, far/sky = black)
  * stratified ray casting in a narrow band around the skyline
  * dynamic normals, Eq. 1, used for the crude simulated terrain image (item 12)
  * planetary curvature drop r^2 / 2R
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch

from .dem import DEM, bilinear, surface_normals, shadow_map


@dataclass
class Camera:
    row: float                 # fractional DEM row of the camera
    col: float                 # fractional DEM column of the camera
    heading_deg: float = 0.0   # view azimuth, 0 = north, clockwise
    pitch_deg: float = 0.0     # + looks up
    hfov_deg: float = 49.0     # Apollo Hasselblad 60 mm on 55 mm film is ~49 deg
    eye_height_m: float = 1.6
    width: int = 1024
    height: int = 1024

    @property
    def focal_px(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)


@dataclass
class RenderConfig:
    supersample: int = 4          # stratified samples per axis inside the skyline band
    band_px: int = 3              # half-width of the skyline band (output pixels)
    eps: float = 0.05             # Eq. 2 margin
    far_mode: str = "global"      # "global": one d_far per view; "column": Eq. 2 per azimuth
    near_m: Optional[float] = None  # None -> 0.5th percentile of visible range
    max_range_m: Optional[float] = None  # None -> DEM diagonal
    min_step_frac: float = 0.5    # min march step as a fraction of the DEM resolution
    curvature: bool = True
    chunk_cols: int = 256
    sun_az_deg: float = 90.0
    sun_el_deg: float = 15.0
    shadows: bool = True
    jitter_seed: Optional[int] = 0  # None -> regular sub-pixel grid
    terrain_floor: float = 0.06   # terrain maps to [floor, 1], sky to 0. Keeps distant ridges
                                  # visible against the sky, matching the affine-shifted
                                  # disparity of the DPT maps the ControlNet is trained on.
                                  # Set 0 for the exact Eq. 3 mapping.


@dataclass
class RenderResult:
    inv_depth: np.ndarray        # (H, W) uint8, Eq. 3, the depth constraint (item 13)
    range_m: np.ndarray          # (H, W) float32, metric range, inf for sky
    terrain_mask: np.ndarray     # (H, W) bool
    skyline: np.ndarray          # (W,) float32, sub-pixel row of the DEM skyline (NaN if none)
    sim_image: np.ndarray        # (H, W) uint8, shaded render (item 12), input to PiDiNet
    d_far: np.ndarray            # (W,) per-column far plane (Eq. 2)
    d_near: float
    camera: dict


def _march_distances(res: float, max_range: float, ang_px: float, min_step_frac: float):
    r = [0.25]
    min_step = res * min_step_frac
    while r[-1] < max_range:
        x = r[-1]
        if x < res:
            step = max(0.05 * x, 0.02)
        else:
            step = max(min_step, 0.5 * x * ang_px)
        r.append(x + step)
    return np.asarray(r, np.float32)


@torch.no_grad()
def render_view(dem: DEM, cam: Camera, cfg: RenderConfig = RenderConfig(),
                device: str | torch.device = "cpu") -> RenderResult:
    dev = torch.device(device)
    elev = torch.from_numpy(np.nan_to_num(dem.elev, nan=-1e9)).float().to(dev)
    H_dem, W_dem = elev.shape
    ground = float(bilinear(elev, torch.tensor([cam.row], device=dev),
                            torch.tensor([cam.col], device=dev))[0])
    if ground < -1e8:
        raise ValueError("Camera is outside the DEM or on nodata.")
    cam_z = ground + cam.eye_height_m

    W, H, S = cam.width, cam.height, max(1, cfg.supersample)
    f = cam.focal_px
    max_range = cfg.max_range_m or dem.res * math.hypot(H_dem, W_dem)
    ang_px = math.radians(cam.hfov_deg) / W
    r = torch.from_numpy(_march_distances(dem.res, max_range, ang_px / S,
                                          cfg.min_step_frac)).to(dev)     # (K,)
    K = r.numel()
    R = dem.body_radius

    # sub-pixel sample positions (stratified)
    g = torch.Generator().manual_seed(cfg.jitter_seed) if cfg.jitter_seed is not None else None

    def offsets(n):
        base = (torch.arange(S).float() + 0.5) / S
        if g is not None and S > 1:
            base = (torch.arange(S).float() + torch.rand(S, generator=g)) / S
        return base

    def column_positions(n_pix, ss):
        if ss == 1:
            return torch.arange(n_pix).float() + 0.5
        return (torch.arange(n_pix).float()[:, None] + offsets(n_pix)[None, :]).reshape(-1)

    heading = math.radians(cam.heading_deg)
    pitch = math.radians(cam.pitch_deg)
    normals = None

    def cast(us: torch.Tensor, vs: torch.Tensor):
        """Cast columns at image x = us (Nc,) and rows at image y = vs (Nr,)."""
        us = us.to(dev)
        vs = vs.to(dev)
        dx = us - W / 2
        phi = heading + torch.atan(dx / f)                                  # (Nc,)
        horiz = torch.sqrt(dx ** 2 + f ** 2)                                # (Nc,)
        theta = torch.atan((H / 2 - vs)[None, :] / horiz[:, None]) + pitch  # (Nc, Nr)
        out = {k: [] for k in ("rng", "rowh", "colh", "dfar")}
        for s0 in range(0, us.numel(), cfg.chunk_cols):
            sl = slice(s0, s0 + cfg.chunk_cols)
            ph = phi[sl]
            rows = cam.row - (r[None, :] * torch.cos(ph)[:, None]) / dem.res   # north = -row
            cols = cam.col + (r[None, :] * torch.sin(ph)[:, None]) / dem.res
            h = bilinear(elev, rows.reshape(-1), cols.reshape(-1), fill=-1e9).view(rows.shape)
            dz = h - cam_z
            if cfg.curvature:
                dz = dz - r[None, :] ** 2 / (2 * R)
            a = torch.atan2(dz, r[None, :].expand_as(dz))
            a = torch.where(h < -1e8, torch.full_like(a, -10.0), a)
            A = torch.cummax(a, dim=1).values
            # pre-pass: furthest visible point along each azimuth (Eq. 2)
            prevA = torch.cat([torch.full_like(A[:, :1], -10.0), A[:, :-1]], 1)
            vis = (a >= prevA) & (a > -9.0)
            slant = torch.sqrt(r[None, :] ** 2 + dz ** 2)
            dfar = (1 + cfg.eps) * torch.where(vis, slant, torch.zeros_like(slant)).amax(1)
            th = theta[sl].contiguous()
            idx = torch.searchsorted(A.contiguous(), th)                       # (c, Nr)
            sky = idx >= K
            i1 = idx.clamp(max=K - 1)
            i0 = (i1 - 1).clamp(min=0)
            r0, r1 = r[i0], r[i1]
            z0 = torch.gather(dz, 1, i0)
            z1 = torch.gather(dz, 1, i1)
            tt = torch.tan(th)
            g0 = z0 - r0 * tt
            g1 = z1 - r1 * tt
            t = torch.where((i1 > 0) & (g1 != g0), (g0 / (g0 - g1)).clamp(0, 1), torch.zeros_like(g0))
            rh = r0 + t * (r1 - r0)
            rng = rh / torch.cos(th)
            rng = torch.where(sky, torch.full_like(rng, float("inf")), rng)
            rowh = cam.row - rh * torch.cos(ph)[:, None] / dem.res
            colh = cam.col + rh * torch.sin(ph)[:, None] / dem.res
            out["rng"].append(rng)
            out["rowh"].append(rowh)
            out["colh"].append(colh)
            out["dfar"].append(dfar)
        return {k: torch.cat(v, 0) for k, v in out.items()}, phi, theta

    # ---- 1) centre samples for the whole image --------------------------------
    uc = torch.arange(W).float() + 0.5
    vc = torch.arange(H).float() + 0.5
    C, phi_c, theta_c = cast(uc, vc)
    rng_c = C["rng"].T.contiguous()                                    # (H, W)
    terrain_c = torch.isfinite(rng_c)

    # ---- 2) stratified samples (supersampled) for the skyline band -------------
    sky_cov = (~terrain_c).float()                                     # sky coverage per pixel
    skyline = torch.full((W,), float("nan"), device=dev)
    if S > 1:
        us = column_positions(W, S)
        vs = column_positions(H, S)
        Ssamp, _, _ = cast(us, vs)
        rs = Ssamp["rng"].view(W, S, H, S).permute(2, 3, 0, 1)        # (H, S, W, S)
        sky_cov = (~torch.isfinite(rs)).float().mean((1, 3))
        # sub-pixel skyline: first terrain sub-row per sub-column, averaged
        rcol = Ssamp["rng"]                                            # (W*S, H*S)
        terr = torch.isfinite(rcol)
        first = torch.where(terr.any(1), terr.float().argmax(1).float(), torch.full((W * S,), float("nan"), device=dev))
        skyline = (first.view(W, S) / S).nanmean(1)
        dfar_cols = Ssamp["dfar"].view(W, S).amax(1)
    else:
        first = torch.where(terrain_c.any(0), terrain_c.float().argmax(0).float(),
                            torch.full((W,), float("nan"), device=dev))
        skyline = first
        dfar_cols = C["dfar"]

    # ---- 3) inverse-depth normalisation (Eq. 3) ---------------------------------
    finite = rng_c[terrain_c]
    if finite.numel() == 0:
        raise ValueError("No terrain visible from this camera.")
    d_near = cfg.near_m or float(torch.quantile(finite[:: max(1, finite.numel() // 200000)], 0.005))
    d_near = max(d_near, 0.1)
    if cfg.far_mode == "column":
        d_far = dfar_cols.clamp(min=d_near * 1.01)[None, :].expand(H, W)
    else:
        d_far = torch.full((H, W), float(dfar_cols.max().clamp(min=d_near * 1.01)), device=dev)
    z = 1.0 / rng_c.clamp(min=1e-3)
    z_near, z_far = 1.0 / d_near, 1.0 / d_far
    dn = ((z - z_far) / (z_near - z_far)).clamp(0, 1)
    dn = cfg.terrain_floor + (1 - cfg.terrain_floor) * dn
    dn = torch.where(terrain_c, dn, torch.zeros_like(dn))
    if S > 1:
        # inside the skyline band, blend by stratified terrain coverage
        rows = torch.arange(H, device=dev).float()[:, None]
        band = (rows - skyline[None, :]).abs() <= cfg.band_px
        band &= torch.isfinite(skyline)[None, :]
        # value of the nearest terrain pixel in the column, used for partially covered pixels
        dn_fill = dn.clone()
        for _ in range(cfg.band_px + 1):
            shifted = torch.cat([dn_fill[1:], dn_fill[-1:]], 0)
            dn_fill = torch.where(terrain_c | (dn_fill > 0), dn_fill, shifted)
        dn = torch.where(band, (1 - sky_cov) * dn_fill, dn)
    inv_depth = (dn * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)

    # ---- 4) crude simulated terrain image (item 12) ----------------------------
    n = surface_normals(dem.elev, dem.res)
    nt = torch.from_numpy(n).to(dev)
    rowh = C["rowh"].T
    colh = C["colh"].T
    nx = bilinear(nt[..., 0], rowh.reshape(-1), colh.reshape(-1), fill=0).view(H, W)
    ny = bilinear(nt[..., 1], rowh.reshape(-1), colh.reshape(-1), fill=0).view(H, W)
    nz = bilinear(nt[..., 2], rowh.reshape(-1), colh.reshape(-1), fill=1).view(H, W)
    saz, sel = math.radians(cfg.sun_az_deg), math.radians(cfg.sun_el_deg)
    sun = torch.tensor([math.sin(saz) * math.cos(sel), math.cos(saz) * math.cos(sel), math.sin(sel)], device=dev)
    mu0 = (nx * sun[0] + ny * sun[1] + nz * sun[2]).clamp(min=0)
    ph = phi_c.to(dev)[None, :].expand(H, W)
    th = theta_c.T.to(dev)
    vx, vy, vz = -torch.sin(ph) * torch.cos(th), -torch.cos(ph) * torch.cos(th), -torch.sin(th)
    mu = (nx * vx + ny * vy + nz * vz).clamp(min=1e-3)
    shade = mu0 / (mu0 + mu + 1e-6)                                    # Lommel-Seeliger
    if cfg.shadows:
        lit = torch.from_numpy(shadow_map(dem.elev, dem.res, cfg.sun_az_deg, cfg.sun_el_deg,
                                          max_dist_m=min(max_range, 5000.0), device=dev)).to(dev)
        shade = shade * bilinear(lit, rowh.reshape(-1), colh.reshape(-1), fill=1).view(H, W)
    if dem.albedo is not None:
        alb = torch.from_numpy(dem.albedo).float().to(dev)
        shade = shade * (0.4 + 0.6 * bilinear(alb, rowh.reshape(-1), colh.reshape(-1), fill=0.5).view(H, W))
    sim = (shade / max(float(shade[terrain_c].quantile(0.99)) if terrain_c.any() else 1.0, 1e-6)).clamp(0, 1) ** (1 / 2.2)
    sim = sim * (1 - sky_cov) if S > 1 else torch.where(terrain_c, sim, torch.zeros_like(sim))
    sim_u8 = (sim * 255 + 0.5).clamp(0, 255).to(torch.uint8)

    return RenderResult(
        inv_depth=inv_depth.cpu().numpy(),
        range_m=rng_c.cpu().numpy().astype(np.float32),
        terrain_mask=terrain_c.cpu().numpy(),
        skyline=skyline.cpu().numpy().astype(np.float32),
        sim_image=sim_u8.cpu().numpy(),
        d_far=dfar_cols.cpu().numpy(),
        d_near=float(d_near),
        camera=asdict(cam),
    )
