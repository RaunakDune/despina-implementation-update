"""Geometry-first evaluation (Section 4): SSIM, Horizon RMSE and SoftEdge F1,
all restricted to terrain-only masks so non-terrain objects never count."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage


def to_gray(img, size: Optional[int] = None) -> np.ndarray:
    pil = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img).astype(np.uint8))
    pil = pil.convert("L")
    if size:
        pil = pil.resize((size, size), Image.BICUBIC)
    return np.asarray(pil).astype(np.float32)


def resample_skyline(skyline: np.ndarray, size: int, src_h: int) -> np.ndarray:
    """Resize a skyline (row per column) from a src_h x len(skyline) image to size x size."""
    x_src = (np.arange(len(skyline)) + 0.5) / len(skyline)
    x_dst = (np.arange(size) + 0.5) / size
    ok = np.isfinite(skyline)
    if ok.sum() < 2:
        return np.full(size, np.nan, np.float32)
    s = np.interp(x_dst, x_src[ok], skyline[ok]) * size / src_h
    return s.astype(np.float32)


def extract_skyline(img, sky: str = "auto", smooth: float = 1.5, max_jump: int = 4,
                    bright_weight: float = 2.0) -> np.ndarray:
    """Skyline (row per column) of a photo or generated image.

    Dynamic programming over columns on a boundary cost: a good skyline row
    has dark sky above and terrain below (or the reverse for bright skies) and
    moves by at most `max_jump` rows between neighbouring columns. Stars, sun
    glare and small artefacts are suppressed by the "sky above must be uniform"
    term. Returns float rows; row 0 means no visible sky in that column.
    """
    g = to_gray(img)
    H, W = g.shape
    g = ndimage.gaussian_filter(g, smooth) / 255.0
    if sky == "auto":
        sky = "dark" if g[: H // 10].mean() < g[-H // 10:].mean() else "bright"
    if sky == "bright":
        g = 1.0 - g
    k = 3
    pad = np.pad(g, ((k, k), (0, 0)), mode="edge")
    below = np.stack([pad[k + 1 + i: k + 1 + i + H] for i in range(k)]).mean(0)
    above = np.stack([pad[k - 1 - i: k - 1 - i + H] for i in range(k)]).mean(0)
    edge = below - above                                      # dark above, bright below
    thr = max(0.04, np.percentile(g, 30) * 0.5)
    bright = (g > thr).astype(np.float32)
    bright_above = np.vstack([np.zeros((1, W), np.float32), np.cumsum(bright, 0)[:-1]]) / H
    cost = -edge + bright_weight * bright_above
    # DP
    acc = cost[:, 0].copy()
    back = np.zeros((H, W), np.int32)
    offs = np.arange(-max_jump, max_jump + 1)
    rows = np.arange(H)
    for j in range(1, W):
        cand = np.full((len(offs), H), np.inf, np.float32)
        for t, o in enumerate(offs):
            src = rows + o
            ok = (src >= 0) & (src < H)
            cand[t, ok] = acc[src[ok]] + 0.002 * abs(o)
        best = cand.argmin(0)
        back[:, j] = rows + offs[best]
        acc = cand[best, rows] + cost[:, j]
    path = np.zeros(W, np.int32)
    path[-1] = int(acc.argmin())
    for j in range(W - 1, 0, -1):
        path[j - 1] = back[path[j], j]
    return path.astype(np.float32)


def terrain_mask(skyline: np.ndarray, size: int, exclude: Optional[np.ndarray] = None,
                 margin: int = 2) -> np.ndarray:
    """Pixels below the DEM skyline (plus a margin), minus excluded non-terrain pixels."""
    rows = np.arange(size)[:, None]
    m = rows > (np.nan_to_num(skyline, nan=size)[None, :] + margin)
    if exclude is not None:
        ex = np.asarray(Image.fromarray((exclude > 0).astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 0
        m &= ~ex
    return m


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    _, smap = structural_similarity(a, b, data_range=255.0, gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False, full=True)
    return float(smap[mask].mean()) if mask.any() else float("nan")


def horizon_rmse(sky_dem: np.ndarray, sky_img: np.ndarray) -> float:
    ok = np.isfinite(sky_dem) & np.isfinite(sky_img)
    return float(np.sqrt(np.mean((sky_dem[ok] - sky_img[ok]) ** 2))) if ok.any() else float("nan")


def edge_f1(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray, thr: float = 0.25,
            tol_px: int = 2) -> dict:
    """Edge F1 with a matching tolerance (BSDS style). Inputs are soft-edge maps in [0,255]."""
    p = (pred.astype(np.float32) / 255.0 > thr) & mask
    r = (ref.astype(np.float32) / 255.0 > thr) & mask
    st = ndimage.generate_binary_structure(2, 1)
    r_d = ndimage.binary_dilation(r, st, iterations=tol_px)
    p_d = ndimage.binary_dilation(p, st, iterations=tol_px)
    prec = (p & r_d).sum() / max(p.sum(), 1)
    rec = (r & p_d).sum() / max(r.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def evaluate_image(gen, ref, dem_skyline: np.ndarray, dem_h: int, size: int = 512,
                   exclude: Optional[np.ndarray] = None, gen_edges=None, ref_edges=None,
                   edge_fn=None) -> dict:
    """All three metrics for one generated image against one reference photo.

    dem_skyline is the ray-cast skyline at the render resolution (dem_h rows).
    gen_edges/ref_edges may be precomputed PiDiNet maps; otherwise edge_fn is called.
    """
    sky = resample_skyline(dem_skyline, size, dem_h)
    m = terrain_mask(sky, size, exclude)
    g = to_gray(gen, size)
    out = {}
    if ref is not None:
        r = to_gray(ref, size)
        out["ssim"] = masked_ssim(g, r, m)
        if gen_edges is None and edge_fn is not None:
            gen_edges = edge_fn(g.astype(np.uint8))
        if ref_edges is None and edge_fn is not None:
            ref_edges = edge_fn(r.astype(np.uint8))
        if gen_edges is not None and ref_edges is not None:
            ge = to_gray(gen_edges, size)
            re_ = to_gray(ref_edges, size)
            ef = edge_f1(ge, re_, m)
            out.update({"softedge_f1": ef["f1"], "softedge_p": ef["precision"], "softedge_r": ef["recall"]})
    out["horizon_rmse"] = horizon_rmse(sky, extract_skyline(g.astype(np.uint8)))
    out["terrain_px"] = int(m.sum())
    return out


def relative_improvement(row: dict, base: dict) -> dict:
    """Two readings of Table 2's 'Relative Improvement'.

    ssim_only reproduces 4 of the 5 published values exactly (0.462/0.378 = +22.2%,
    0.489/0.378 = +29.4%, 0.534/0.378 = +41.3%); composite is the mean of the
    three normalised gains described in the table footnote.
    """
    s = row["ssim"] / base["ssim"] - 1
    f = row["softedge_f1"] / base["softedge_f1"] - 1
    h = (base["horizon_rmse"] - row["horizon_rmse"]) / base["horizon_rmse"]
    return {"rel_ssim_only": s, "rel_composite": (s + f + h) / 3}
