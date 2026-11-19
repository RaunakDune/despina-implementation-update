#!/usr/bin/env python
"""One-shot DESPINA synthesis for a single view: DEM + camera -> constraints -> images.

Renders the inverse-depth map, the shaded simulated view and its PiDiNet soft edges from the
DEM (Group II), then generates images with SDXL + lunar LoRA + ControlNets (Group III).
Use it to produce synthetic surface views for downstream work such as visual place
recognition. Writes into --out:

  dem_depth.png  dem_sim.png  dem_softedge.png  dem_skyline.npy  meta.json  image_seed<k>.png

Example (camera given as fractions of the DEM grid; see --help for lon/lat, x/y, row/col):

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 python scripts/synthesize_view.py \
      --dem DESPINA_technical_details/TIF-to-image/dem_data.tif --upscale 4 \
      --frac 0.2005,0.6348 --heading 270 --sun-az 120 --sun-el 12 \
      --lora runs/main/lora_noff --cn-depth runs/main/cn_depth --cn-softedge runs/main/cn_softedge \
      --seeds 0,1 --out runs/synth/example --devices cuda:0,cuda:1
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from despina import constraints as C  # noqa: E402
from despina.dem import load_dem  # noqa: E402
from despina.raycast import Camera, RenderConfig, render_view  # noqa: E402

PRESETS = {  # the Table 2 configurations; "full" is the DESPINA system
    "full": dict(use_lora=True, controls=["depth", "softedge"], control_scales=[0.8, 0.5]),
    "lora_depth": dict(use_lora=True, controls=["depth"], control_scales=[0.8]),
    "lora_softedge": dict(use_lora=True, controls=["softedge"], control_scales=[0.6]),
    "base_depth": dict(use_lora=False, controls=["depth"], control_scales=[0.8]),
    "base_softedge": dict(use_lora=False, controls=["softedge"], control_scales=[0.6]),
}


def pair(s):
    a, b = s.split(",")
    return float(a), float(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", required=True, help="GeoTIFF DEM")
    ap.add_argument("--upscale", type=float, default=1.0, help="bicubic DEM upscaling factor")
    ap.add_argument("--body", default="moon", choices=["moon", "earth", "mars"])
    pos = ap.add_mutually_exclusive_group(required=True)
    pos.add_argument("--lonlat", type=pair, help="LON,LAT in degrees (planetocentric; needs a CRS)")
    pos.add_argument("--xy", type=pair, help="X,Y in the DEM's map CRS (metres)")
    pos.add_argument("--rowcol", type=pair, help="ROW,COL in DEM pixels (after upscaling)")
    pos.add_argument("--frac", type=pair, help="ROW_FRAC,COL_FRAC of the DEM grid, 0..1")
    ap.add_argument("--heading", type=float, default=0.0, help="view azimuth, 0 = north, clockwise")
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--hfov", type=float, default=49.0, help="Apollo Hasselblad 60 mm is about 49 deg")
    ap.add_argument("--eye-height", type=float, default=1.6)
    ap.add_argument("--sun-az", type=float, default=90.0)
    ap.add_argument("--sun-el", type=float, default=15.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--category", default="hilly", choices=["flat", "hilly", "rocky"],
                    help="selects the prompt sentences when --prompt is not given")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--config", default="full", choices=sorted(PRESETS))
    ap.add_argument("--lora", default=None, help="lunar LoRA folder (needed by lora/full configs)")
    ap.add_argument("--cn-depth", default=None, help="depth ControlNet folder (default: pretrained)")
    ap.add_argument("--cn-softedge", default=None, help="soft-edge ControlNet folder (default: pretrained)")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--devices", default="cuda:0,cuda:1",
                    help="SDXL on the first, ControlNets on the second (two P40s); one device if it fits")
    ap.add_argument("--render-only", action="store_true", help="write the DEM constraints and stop")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    render_dev = a.devices.split(",")[0]
    dem = load_dem(a.dem, upscale=a.upscale, body=a.body)
    if a.lonlat:
        row, col = dem.lonlat_to_pix(*a.lonlat)
    elif a.xy:
        row, col = dem.map_to_pix(*a.xy)
    elif a.rowcol:
        row, col = a.rowcol
    else:
        row, col = dem.shape[0] * a.frac[0], dem.shape[1] * a.frac[1]
    cam = Camera(row=row, col=col, heading_deg=a.heading, pitch_deg=a.pitch, hfov_deg=a.hfov,
                 eye_height_m=a.eye_height, width=a.size, height=a.size)
    res = render_view(dem, cam, RenderConfig(sun_az_deg=a.sun_az, sun_el_deg=a.sun_el), device=render_dev)
    Image.fromarray(res.inv_depth).save(out / "dem_depth.png")
    Image.fromarray(res.sim_image).save(out / "dem_sim.png")
    soft = C.soft_edges(res.sim_image, render_dev)
    Image.fromarray(soft).save(out / "dem_softedge.png")
    np.save(out / "dem_skyline.npy", res.skyline)
    prompt = a.prompt or C.dem_prompt(a.category)
    meta = {"args": vars(a), "camera": res.camera, "d_near": res.d_near, "d_far": float(res.d_far.max()),
            "terrain_frac": float(res.terrain_mask.mean()), "prompt": prompt}
    json.dump(meta, open(out / "meta.json", "w"), indent=1, default=str)
    print(f"rendered: terrain {meta['terrain_frac']:.2f} of frame, far plane {meta['d_far']:.0f} m", flush=True)
    if a.render_only:
        return

    from despina.pipeline import Despina, GenConfig
    preset = PRESETS[a.config]
    if preset["use_lora"] and not a.lora:
        sys.exit(f"--config {a.config} needs --lora")
    cn_paths = {k: v for k, v in {"depth": a.cn_depth, "softedge": a.cn_softedge}.items() if v}
    model = Despina(a.devices, lora_path=a.lora, controlnet_paths=cn_paths)
    cfg = GenConfig(name=a.config, steps=a.steps, guidance=a.guidance, size=a.size, **preset)
    cond = {"depth": Image.fromarray(res.inv_depth), "softedge": Image.fromarray(soft)}
    for seed in [int(s) for s in a.seeds.split(",")]:
        img = model.generate(prompt, cond, cfg, seed=seed)
        img.save(out / f"image_seed{seed}.png")
        print("wrote", out / f"image_seed{seed}.png", flush=True)


if __name__ == "__main__":
    main()
