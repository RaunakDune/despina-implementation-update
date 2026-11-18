#!/usr/bin/env python
"""Diagnose washed-out generations: LoRA alone (ControlNets at zero weight), LoRA checkpoint
comparison, and a LoRA-strength x guidance sweep with the trained ControlNets.

  python scripts/diagnose_lora.py --out runs/diag --part 0 --parts 3 --devices cuda:0,cuda:1
"""
import argparse, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from despina.pipeline import Despina, GenConfig  # noqa: E402

PROMPTS = {
    "flat": "A realistic image of the Moon's surface, from the perspective of a rover. Scattered rocks visible on the lunar surface. Darks shadows cast by the terrain. A straight, flat horizon line on the surface.",
    "hilly": "A realistic image of the Moon's surface, from the perspective of a rover. A lunar landscape with hills in the background. Scattered rocks visible on the lunar surface. Darks shadows cast by the terrain.",
}


def jobs(site_dir):
    J = []
    # 1. LoRA alone: ControlNets at zero weight
    for pk in PROMPTS:
        for lora, scale in (("none", 0.0), ("final", 0.9), ("c2000", 0.9)):
            J.append(dict(name=f"loraonly_{pk}_{lora}", prompt=PROMPTS[pk], lora=lora, lora_scale=scale,
                          guidance=7.0, cscales=[0.0, 0.0]))
    # 2. sweep with trained ControlNets on one flat view
    for scale in (0.5, 0.9, 1.2):
        for g in (4.0, 7.0):
            J.append(dict(name=f"sweep_l{scale}_g{g}", prompt=PROMPTS["hilly"], lora="final", lora_scale=scale,
                          guidance=g, cscales=[0.8, 0.5]))
    J.append(dict(name="sweep_c2000_l0.9_g7.0", prompt=PROMPTS["hilly"], lora="c2000", lora_scale=0.9, guidance=7.0, cscales=[0.8, 0.5]))
    J.append(dict(name="sweep_nolora_g7.0", prompt=PROMPTS["hilly"], lora="none", lora_scale=0.0, guidance=7.0, cscales=[0.8, 0.5]))
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--site", default="runs/check/sites/flat_view_1")
    ap.add_argument("--lora", default="runs/main/lora")
    ap.add_argument("--cn-depth", default="runs/main/cn_depth")
    ap.add_argument("--cn-softedge", default="runs/main/cn_softedge")
    ap.add_argument("--devices", default="cuda:0,cuda:1")
    ap.add_argument("--ckpt", default="checkpoint-2000")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    todo = [j for k, j in enumerate(jobs(a.site)) if k % a.parts == a.part and not (out / f"{j['name']}.png").exists()]
    if not todo:
        return
    m = Despina(a.devices, lora_path=None, controlnet_paths={"depth": a.cn_depth, "softedge": a.cn_softedge})
    m.pipe.load_lora_weights(a.lora, adapter_name="final")
    m.pipe.load_lora_weights(str(Path(a.lora) / a.ckpt), adapter_name="c2000")
    for n in ("unet", "text_encoder", "text_encoder_2"):
        getattr(m.pipe, n).to(m.home)
    cond = {"depth": Image.open(Path(a.site) / "dem_depth.png"), "softedge": Image.open(Path(a.site) / "dem_softedge.png")}
    for j in todo:
        if j["lora"] == "none":
            m.pipe.set_adapters(["final"], adapter_weights=[0.0])
        else:
            m.pipe.set_adapters([j["lora"]], adapter_weights=[j["lora_scale"]])
        cfg = GenConfig(name=j["name"], use_lora=False, controls=["depth", "softedge"], control_scales=j["cscales"],
                        guidance=j["guidance"], steps=30)
        m.lora_loaded = False                     # adapters already set above
        img = m.generate(j["prompt"], cond, cfg, seed=0)
        img.save(out / f"{j['name']}.png")
        a_ = np.asarray(img.convert("L")).astype(float)
        print(j["name"], "mean", round(a_.mean(), 1), "std", round(a_.std(), 1), "top10%", round(a_[:102].mean(), 1), flush=True)


if __name__ == "__main__":
    main()
