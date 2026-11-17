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


def find_frame(gray: np.ndarray, dark: float = 30.0) -> tuple[int, int, int]:
    """Return (x0, y0, side) of the square image frame inside a film-strip scan.

    Every scan shows a vertical strip (sprockets left and right) spanning the scan width,
    so the frame side is about 0.758 x the scan width. The frame's left, right and bottom
    edges are found from the bright terrain, which in surface photographs reaches all
    three; the top edge usually merges with black sky, so y0 = bottom - side.
    """
    H, W = gray.shape
    side_prior = int(round(FRAME_TO_SCAN_WIDTH * W))
    thr = max(40.0, 0.35 * float(np.percentile(gray, 99.5)))
    bright = gray > thr
    # ignore the sprocket bands at the outer 8% on each side
    band = int(0.08 * W)
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


def grey_world(img: Image.Image, lo: float = 20, hi: float = 98) -> Image.Image:
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
    gain = means.mean() / np.maximum(means, 1e-3)
    return Image.fromarray(np.clip(a * gain, 0, 255).astype(np.uint8))


def process_scan(path, size: int = 1024, inset: float = 0.015, white_balance: bool = True):
    """Film-strip scan -> square, reseau-free frame (the training image, item 1)."""
    crop, info = crop_frame(load_scan(path), size=size, inset=inset)
    info.update(LAST_FRAME_QC)
    clean, found, _ = remove_reseau(crop, inset=inset, return_info=True)
    if white_balance:
        clean = grey_world(clean)
    info["reseau_inpainted"] = int(sum(f["inpainted"] for f in found))
    info["qc_ok"] = bool(info["sides_from_terrain"] and info["bottom_from_edge"])
    return clean, info
