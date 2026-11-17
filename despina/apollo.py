"""Apollo Hasselblad frame extraction and fiducial (reseau) removal (DESPINA Group I, item 1).

The ASU March to the Moon scans show the whole 70 mm film strip: sprocket holes, the dark
film border and parts of the neighbouring frames. `crop_frame` finds the 55 mm square
frame; `remove_reseau` finds the 5 x 5 grid of reseau crosses exposed onto every frame
(10 mm apart on the Hasselblad Data Camera) and inpaints them.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
FRAME_MM = 55.0
RESEAU_MM = 10.0


def _runs(mask):
    """Start/end (inclusive) of True runs in a 1-D boolean array."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def load_scan(path) -> Image.Image:
    """Open a scan as 8-bit RGB. The B&W magazines are 16-bit greyscale PNGs; converting
    them with PIL's .convert() clips everything above 255 to white, so scale explicitly."""
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(im).astype(np.float32)
        a = np.clip(a / (257.0 if a.max() > 255 else 1.0), 0, 255).astype(np.uint8)
        return Image.fromarray(a).convert("RGB")
    return im.convert("RGB")


LAST_FRAME_QC: dict = {}
FRAME_TO_SCAN_WIDTH = 0.758   # 55 mm frame / scanned 70 mm strip width (measured on AS17 and AS12)


def _longest_run(mask):
    runs = _runs(mask)
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def find_frame_terrain(gray: np.ndarray, dark: float = 30.0) -> tuple[int, int, int]:
    """Return (x0, y0, side) of the square image frame inside a film-strip scan.

    Every scan shows a vertical strip (sprockets left and right) spanning the scan width,
    so the frame side is about 0.758 x the scan width. The frame's left, right and bottom
    edges are found from the bright terrain, which in surface photographs reaches all
    three; the top edge usually merges with black sky, so y0 = bottom - side.
    """
    H, W = gray.shape
    side_prior = int(round(FRAME_TO_SCAN_WIDTH * W))
    # ignore the sprocket bands at the outer 8% on each side; white sprocket holes would
    # otherwise set the threshold and hide dark terrain
    band = int(0.08 * W)
    thr = max(30.0, 0.2 * float(np.percentile(gray[:, band:W - band], 99.5)))
    bright = gray > thr
    colfrac = bright[:, band:W - band].mean(0)
    run = _longest_run(colfrac > 0.03)
    if run is not None:
        x0, x1 = band + run[0], band + run[1]
    else:
        x0, x1 = (W - side_prior) // 2, (W + side_prior) // 2
    width = x1 - x0 + 1
    global LAST_FRAME_QC
    LAST_FRAME_QC = {"sides_from_terrain": bool(abs(width - side_prior) <= 0.03 * W), "bottom_from_edge": False}
    if abs(width - side_prior) <= 0.03 * W:
        side = width
    else:  # terrain does not reach both edges: keep the prior size, centred on the terrain
        side = side_prior
        x0 = int(np.clip((x0 + x1) // 2 - side // 2, 0, W - side))
    rowfrac = bright[:, x0:x0 + side].mean(1)
    rrun = _longest_run(rowfrac > 0.05)
    y0 = None
    if rrun is not None:
        y1 = rrun[1]
        below = gray[y1 + 3:min(y1 + 3 + max(10, side // 60), H), x0:x0 + side]
        if below.size and float(np.median(below)) < dark:   # a real frame edge, then border
            y0 = y1 - side + 1
            LAST_FRAME_QC["bottom_from_edge"] = True
    if y0 is None:
        # fall back to the gap before the next frame, then to the centre
        rp99 = np.percentile(gray[:, x0:x0 + side], 99, axis=1)
        gaps = [r for r in _runs(rp99 < dark) if r[0] > H * 0.5 and r[1] - r[0] > side * 0.02]
        y0 = gaps[0][0] - side if gaps else (H - side) // 2
    y0 = int(np.clip(y0, 0, max(H - side, 0)))
    return int(x0), y0, int(min(side, H))


def locate_reseau_grid(gray: np.ndarray, min_score: float = 5.0):
    """Find the reseau lattice in a whole film-strip scan.

    Works at a scale where the frame is about 1024 px. Peaks of a cross matched filter vote
    for the lattice pitch and phase; the frame is the 5 x 5 window of that lattice.
    Returns lattice parameters in full-scan pixels, or None when too few crosses show.
    """
    import cv2
    from scipy import ndimage
    H, W = gray.shape
    f = 1024.0 / (FRAME_TO_SCAN_WIDTH * W)
    small = cv2.resize(gray.astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    bh = cv2.morphologyEx(small, cv2.MORPH_BLACKHAT, np.ones((9, 9), np.uint8))
    p0 = 1024.0 * RESEAU_MM / FRAME_MM
    score = cv2.filter2D(bh, -1, _cross_kernel(int(round(p0 * 0.065))))
    band = int(0.08 * small.shape[1])
    score[:, :band] = 0
    score[:, -band:] = 0
    peaks = (ndimage.maximum_filter(score, 25) == score) & (score > min_score)
    ys, xs = np.nonzero(peaks)
    if len(xs) < 6:
        return None
    w = score[ys, xs]
    order = np.argsort(-w)[:200]
    xs, ys, w = xs[order].astype(float), ys[order].astype(float), w[order]
    best = None
    for pp in p0 * np.linspace(0.94, 1.06, 49):
        zx = (w * np.exp(2j * np.pi * xs / pp)).sum() / w.sum()
        zy = (w * np.exp(2j * np.pi * ys / pp)).sum() / w.sum()
        r = abs(zx) * abs(zy)
        if best is None or r > best[0]:
            best = (r, pp, zx, zy)
    _, pp, zx, zy = best
    ox = (np.angle(zx) / (2 * np.pi) * pp) % pp
    oy = (np.angle(zy) / (2 * np.pi) * pp) % pp
    i = np.round((xs - ox) / pp).astype(int)
    j = np.round((ys - oy) / pp).astype(int)
    res = np.hypot(xs - ox - i * pp, ys - oy - j * pp)
    inl = res < 0.1 * pp
    # a real cross sits on exposed film; frame-number digits (white on black) and sprocket
    # edges also excite the filter but sit on a dark or saturated surround
    r = max(int(0.3 * pp), 6)
    loc = np.array([np.median(small[max(int(y) - r, 0):int(y) + r, max(int(x) - r, 0):int(x) + r])
                    for x, y in zip(xs, ys)])
    if inl.any():
        ref = np.median(loc[inl])
        inl &= (loc > max(20.0, 0.4 * ref)) & (loc < 250)
    if inl.sum() < 6:
        return None
    return {"f": f, "p": pp / f, "ox": ox / f, "oy": oy / f, "i": i[inl], "j": j[inl],
            "x": xs[inl] / f, "y": ys[inl] / f, "w": w[inl]}


def _window(idx, wts, prefer, origin, pitch, profile=None):
    """Pick the 5-wide lattice window with the most crosses.

    When some crosses are hidden (shadow, black sky) several windows contain all the
    visible ones. With a brightness `profile` along this axis, ties go to the window whose
    implied frame edges separate bright inside from dark outside; otherwise to the window
    centred nearest `prefer`.
    """
    cands = []
    for k0 in range(idx.min() - 4, idx.max() + 1):
        sel = (idx >= k0) & (idx <= k0 + 4)
        cands.append((wts[sel].sum(), sel.sum(), k0))
    top = max(c[0] for c in cands)
    near = [c for c in cands if c[0] >= 0.9 * top]
    if profile is not None and len(near) > 1:
        n = len(profile)

        def edge_score(k0):
            c = origin + (k0 + 2) * pitch
            lo, hi = int(round(c - 2.75 * pitch)), int(round(c + 2.75 * pitch))
            m = max(int(0.25 * pitch), 4)
            inside = profile[max(lo, 0):min(hi, n)]
            outside = np.concatenate([profile[max(lo - m, 0):max(lo, 0)], profile[min(hi, n):min(hi + m, n)]])
            if inside.size == 0:
                return -1.0
            return float(inside.mean() - (outside.mean() if outside.size else 0.0))
        return max(near, key=lambda c: edge_score(c[2]))[2]
    return min(near, key=lambda c: abs(origin + (c[2] + 2) * pitch - prefer))[2]


def find_frame(gray: np.ndarray, dark: float = 30.0, min_crosses: int = 8) -> tuple[int, int, int]:
    """Return (x0, y0, side) of the square frame in a film-strip scan.

    The reseau grid fixes the frame's horizontal position and size (the crosses are 10 mm
    apart, centred on the 55 mm frame) and is visible exactly where the terrain method fails:
    bright, edge-to-edge ground close-ups. The vertical position comes from the frame's bottom
    edge when it is visible, otherwise from the grid. With too few crosses the terrain method
    (find_frame_terrain) is used alone.
    """
    H, W = gray.shape
    tx0, ty0, tside = find_frame_terrain(gray, dark)
    tqc = dict(LAST_FRAME_QC)
    g = locate_reseau_grid(gray)
    qc = {**tqc, "method": "terrain", "grid_crosses": 0}
    if g is not None:
        p = g["p"]
        band = int(0.08 * W)
        thr = max(30.0, 0.2 * float(np.percentile(gray[:, band:W - band], 99.5)))
        colprof = (gray > thr).mean(0).astype(np.float32)
        colprof[:band] = 0; colprof[W - band:] = 0          # sprocket holes are not frame
        i0 = _window(g["i"], g["w"], tx0 + tside / 2, g["ox"], p, profile=colprof)
        insx = (g["i"] >= i0) & (g["i"] <= i0 + 4)
        n_cols = len(set(g["i"][insx]))
        bottom_t = ty0 + tside
        if tqc.get("bottom_from_edge"):
            # rows window whose implied bottom edge best matches the terrain bottom edge
            j0 = min(range(g["j"].min() - 4, g["j"].max() + 1),
                     key=lambda k: abs(g["oy"] + (k + 2) * p + 2.75 * p - bottom_t))
        else:
            j0 = _window(g["j"], g["w"], H / 2, g["oy"], p)
        ins = insx & (g["j"] >= j0) & (g["j"] <= j0 + 4)
        rows_in = (g["j"] >= j0) & (g["j"] <= j0 + 4)
        # keep columns with crosses in several rows, then the best contiguous run of <= 5
        counts = {c: int((rows_in & (g["i"] == c)).sum()) for c in set(g["i"][rows_in])}
        if counts:
            cmax = max(counts.values())
            good = sorted(c for c, n in counts.items() if n >= max(2, 0.4 * cmax))
            runs, cur = [], []
            for c in good:
                if cur and c != cur[-1] + 1:
                    runs.append(cur); cur = []
                cur.append(c)
            if cur:
                runs.append(cur)
            best_run = []
            for run in runs:
                for k in range(0, max(len(run) - 4, 1)):
                    sub = run[k:k + 5]
                    if sum(counts[c] for c in sub) > sum(counts[c] for c in best_run):
                        best_run = sub
            rows_in &= np.isin(g["i"], best_run)
        cols = sorted(set(g["i"][rows_in]))
        side_prior = FRAME_TO_SCAN_WIDTH * W
        span = (len(cols) - 1 + 1.5) * p if cols else 0      # implied frame width
        if rows_in.sum() >= min_crosses and len(cols) >= 4 and span >= 0.78 * side_prior:
            # Frame edges sit 7.5 mm (0.75 pitch) outside the outermost cross columns that are
            # actually exposed. Some magazines have a narrower aperture with only four columns.
            colx = {c: float(np.median(g["x"][rows_in & (g["i"] == c)])) for c in cols}
            pitch = (colx[cols[-1]] - colx[cols[0]]) / (cols[-1] - cols[0])
            if not (0.9 * p < pitch < 1.1 * p):
                pitch = p
            x_left = colx[cols[0]] - 0.75 * pitch
            x_right = colx[cols[-1]] + 0.75 * pitch
            side = int(round(min(x_right - x_left, FRAME_MM / RESEAU_MM * pitch)))
            x0 = int(round((x_left + x_right) / 2 - side / 2))
            rows = sorted(set(g["j"][rows_in]))
            rowy = {r: float(np.median(g["y"][rows_in & (g["j"] == r)])) for r in rows}
            grid_bottom = rowy[rows[-1]] + 0.75 * pitch
            if tqc.get("bottom_from_edge") and abs(grid_bottom - bottom_t) < 0.4 * pitch:
                y1 = bottom_t
            elif j0 + 4 in rows:
                y1 = grid_bottom
            else:
                y1 = g["oy"] + (j0 + 4) * pitch + 0.75 * pitch
            y0 = int(round(y1 - side))
            qc = {**tqc, "method": "reseau", "grid_crosses": int(rows_in.sum()), "grid_cols": len(cols),
                  "terrain_offset_px": int(round(abs(x0 - tx0)))}
            x0 = int(np.clip(x0, 0, max(W - side, 0)))
            y0 = int(np.clip(y0, 0, max(H - side, 0)))
            LAST_FRAME_QC.clear(); LAST_FRAME_QC.update(qc)
            return x0, y0, int(min(side, H, W))
    LAST_FRAME_QC.clear(); LAST_FRAME_QC.update(qc)
    return tx0, ty0, tside


def crop_frame(img: Image.Image, size: int = 1024, inset: float = 0.015):
    """Square frame crop resized to `size`. `inset` trims a thin margin at the frame edge."""
    rgb = img.convert("RGB")
    g = np.asarray(rgb.convert("L")).astype(np.float32)
    x0, y0, s = find_frame(g)
    t = int(s * inset)
    box = (x0 + t, y0 + t, x0 + s - t, y0 + s - t)
    return rgb.crop(box).resize((size, size), Image.LANCZOS), {"x0": x0, "y0": y0, "side": s, "inset_px": t}


def reseau_grid(size: int, inset: float = 0.015):
    """Predicted pixel centres of the 5 x 5 reseau crosses in a `crop_frame` output."""
    span = 1 - 2 * inset
    step = size * (RESEAU_MM / FRAME_MM) / span
    c = size / 2
    return [(c + i * step, c + j * step) for j in range(-2, 3) for i in range(-2, 3)], step


def _cross_kernel(arm: int, width: int = 1, ring: int = 4):
    """Zero-mean matched filter: +1 on a thin cross, -1 on the surrounding square."""
    n = 2 * (arm + ring) + 1
    c = n // 2
    k = np.zeros((n, n), np.float32)
    on = np.zeros((n, n), bool)
    on[c - width // 2:c + width // 2 + 1, c - arm:c + arm + 1] = True
    on[c - arm:c + arm + 1, c - width // 2:c + width // 2 + 1] = True
    k[on] = 1.0 / on.sum()
    k[~on] = -1.0 / (~on).sum()
    return k


def remove_reseau(img: Image.Image, inset: float = 0.015, search: int = 40,
                  thr: float = 6.0, thr_fitted: float = 2.5, return_info: bool = False):
    """Find the 25 reseau crosses and inpaint them.

    1. Matched-filter score on the black-hat image (thin dark structures) for a '+' shape.
    2. Best candidate within `search` px of each predicted grid point.
    3. Robust fit of a grid model (offset, x/y pitch, rotation) to the confident candidates,
       because rocks and footprints in textured ground can outscore a faint cross.
    4. Final position = model prediction refined within +-4 px; inpaint where the score at
       that position clears `thr`. Crosses lost in black sky or shadow need no repair.
    """
    import cv2
    rgb = np.asarray(img.convert("RGB")).copy()
    size = rgb.shape[0]
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((9, 9), np.uint8))
    pts, step = reseau_grid(size, inset)
    arm = max(6, int(round(step * 0.065)))
    k_small, k_big = _cross_kernel(arm), _cross_kernel(int(arm * 1.7))
    sc_small, sc_big = cv2.filter2D(bh, -1, k_small), cv2.filter2D(bh, -1, k_big)
    ij = [(i, j) for j in range(-2, 3) for i in range(-2, 3)]

    def best(score, x, y, r):
        x0, x1 = int(max(x - r, 0)), int(min(x + r, size - 1))
        y0, y1 = int(max(y - r, 0)), int(min(y + r, size - 1))
        win = score[y0:y1 + 1, x0:x1 + 1]
        yy, xx = np.unravel_index(np.argmax(win), win.shape)
        return x0 + xx, y0 + yy, float(win[yy, xx])

    cand = []
    for n, (px, py) in enumerate(pts):
        sc = sc_big if n == 12 else sc_small
        cand.append(best(sc, px, py, search))
    # robust grid fit: x = cx + a*i - b*j, y = cy + b*i + a2*j  (pitch a, a2; rotation b)
    use = np.array([c[2] > 2 * thr for c in cand])
    model = np.array([size / 2, step, step, 0.0])      # cx, cy share the predicted centre
    cxy = np.array([size / 2, size / 2])
    for _ in range(4):
        if use.sum() < 4:
            break
        A, bvec = [], []
        for (i, j), c, u in zip(ij, cand, use):
            if u:
                A.append([1, 0, i, 0, -j]); bvec.append(c[0])
                A.append([0, 1, 0, j, i]); bvec.append(c[1])
        sol, *_ = np.linalg.lstsq(np.array(A, float), np.array(bvec, float), rcond=None)
        cxy, model = sol[:2], np.array([0, sol[2], sol[3], sol[4]])
        pred = np.array([[cxy[0] + sol[2] * i - sol[4] * j, cxy[1] + sol[3] * j + sol[4] * i] for i, j in ij])
        res = np.hypot(pred[:, 0] - [c[0] for c in cand], pred[:, 1] - [c[1] for c in cand])
        new_use = use & (res < max(4.0, 2.5 * np.median(res[use])))
        if (new_use == use).all():
            break
        use = new_use
    else:
        pass
    if use.sum() >= 4:
        grid, fitted = pred, True
    else:
        grid, fitted = np.array(pts), False
    mask = np.zeros(g.shape, np.uint8)
    out = rgb.astype(np.float32)
    src = out.copy()
    found = []
    hw, off = 2, 4          # band half-width covering the 1-2 px line; sample just outside it
    for n, (gx, gy) in enumerate(grid):
        sc = sc_big if n == 12 else sc_small
        a = int(arm * (1.7 if n == 12 else 1.0) * 1.3) + 3
        cx, cy, s = best(sc, gx, gy, 4)
        hit = s > (thr_fitted if fitted else thr)
        found.append({"i": n, "x": int(cx), "y": int(cy), "score": round(s, 2), "inpainted": bool(hit)})
        if not hit:
            continue
        xa, xb = max(cx - a, 0), min(cx + a + 1, size)
        ya, yb = max(cy - a, 0), min(cy + a + 1, size)
        # vertical arm first (from the original), then the horizontal arm from the already
        # cleaned image, so the centre of the cross is filled from line-free pixels
        for dx in range(-hw, hw + 1):
            x = cx + dx
            if off <= x < size - off and 0 <= cx - off and cx + off < size:
                t = (dx + hw + 1) / (2 * hw + 2)
                out[ya:yb, x] = (1 - t) * src[ya:yb, cx - off] + t * src[ya:yb, cx + off]
                mask[ya:yb, x] = 255
        clean = out.copy()
        for dy in range(-hw, hw + 1):
            y = cy + dy
            if off <= y < size - off and 0 <= cy - off and cy + off < size:
                t = (dy + hw + 1) / (2 * hw + 2)
                out[y, xa:xb] = (1 - t) * clean[cy - off, xa:xb] + t * clean[cy + off, xa:xb]
                mask[y, xa:xb] = 255
    res_img = Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
    return (res_img, found, mask) if return_info else res_img


def grey_world(img: Image.Image, lo: float = 20, hi: float = 98, strength: float = 0.75) -> Image.Image:
    """Grey-world white balance on mid-tone pixels. The lunar surface is close to neutral,
    and the ASU colour scans carry strong film-dye casts (purple AS12-46, orange AS11-37)."""
    a = np.asarray(img).astype(np.float32)
    lum = a.mean(2)
    if np.ptp(a.reshape(-1, 3).mean(0)) < 1e-3:      # already grey (B&W magazines)
        return img
    lo_v, hi_v = np.percentile(lum, [lo, hi])
    sel = (lum >= max(lo_v, 15)) & (lum <= hi_v)
    if sel.sum() < 1000:
        return img
    means = a[sel].mean(0)
    gain = (means.mean() / np.maximum(means, 1e-3)) ** strength
    return Image.fromarray(np.clip(a * gain, 0, 255).astype(np.uint8))


def border_in_crop(img: Image.Image, strip: float = 0.03) -> bool:
    """True if the crop shows film border at its left or right edge: a strip whose lower 40%
    is black (frame-number band, border) or that contains sprocket holes (bright patches on
    black). Terrain reaches the bottom corners of a correctly cropped surface photo."""
    g = np.asarray(img.convert("L")).astype(np.float32)
    H, W = g.shape
    w = max(int(strip * W), 4)
    for band in (g[:, :w], g[:, W - w:]):
        low = band[int(0.6 * H):]
        if float(np.median(low)) < 20:
            return True
        frac_white = float((band > 235).mean()); frac_black = float((band < 20).mean())
        if frac_white > 0.08 and frac_black > 0.3:
            return True
    return False


def process_scan(path, size: int = 1024, inset: float = 0.015, white_balance: bool = True):
    """Film-strip scan -> square, reseau-free frame (the training image, item 1)."""
    crop, info = crop_frame(load_scan(path), size=size, inset=inset)
    info.update(LAST_FRAME_QC)
    clean, found, _ = remove_reseau(crop, inset=inset, return_info=True)
    if white_balance:
        clean = grey_world(clean)
    info["reseau_inpainted"] = int(sum(f["inpainted"] for f in found))
    info["border_in_crop"] = border_in_crop(clean)
    info["qc_ok"] = bool(not info["border_in_crop"] and (info.get("method") == "reseau" or
                         info["sides_from_terrain"] or info["bottom_from_edge"]))
    return clean, info
