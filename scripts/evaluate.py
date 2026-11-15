#!/usr/bin/env python
"""Section 4 evaluation: per-image metrics -> Table 1 (baseline vs DESPINA) and Table 2 (ablation).

  python scripts/evaluate.py --sites-dir runs/sites --gen runs/gen --out runs/eval --device cuda:0
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from despina import constraints as C  # noqa: E402
from despina.metrics import evaluate_image, relative_improvement  # noqa: E402

ABLATION = ["base_softedge", "base_depth", "lora_softedge", "lora_depth", "full"]
LABELS = {"base_softedge": "Base + Soft Edge", "base_depth": "Base + Depth Map",
          "lora_softedge": "LoRA + Soft Edge", "lora_depth": "LoRA + Depth Map",
          "full": "Full DESPINA system", "baseline_imgcond": "Baseline (image-conditioned)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-dir", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--size", type=int, default=512, help="evaluation resolution")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sd in sorted(p for p in Path(a.sites_dir).iterdir() if (p / "meta.json").exists()):
        sky = np.load(sd / "dem_skyline.npy")
        dem_h = Image.open(sd / "dem_depth.png").size[1]
        ref = Image.open(sd / "photo.png") if (sd / "photo.png").exists() else None
        ref_edges = Image.open(sd / "photo_softedge.png") if ref is not None else None
        meta = json.load(open(sd / "meta.json"))
        ex = meta["site"].get("exclude_mask")
        exclude = np.array(Image.open(ex).convert("L").resize((a.size, a.size))) > 127 if ex else None
        for cfg_dir in sorted((Path(a.gen) / sd.name).glob("*")):
            for f in sorted(cfg_dir.glob("seed*.png")):
                gen = Image.open(f).convert("RGB")
                ge = C.soft_edges(gen, a.device) if ref is not None else None
                m = evaluate_image(gen, ref, sky, dem_h, a.size, exclude, gen_edges=ge, ref_edges=ref_edges)
                rows.append({"site": sd.name, "config": cfg_dir.name, "seed": int(f.stem[4:]), **m})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image.csv", index=False)
    if df.empty:
        print("no generated images found"); return
    cols = [c for c in ("ssim", "horizon_rmse", "softedge_f1") if c in df]
    per_site = df.groupby(["site", "config"])[cols].mean().reset_index()
    per_site.to_csv(out / "per_site.csv", index=False)

    lines = ["# Table 1: image-conditioned baseline vs DESPINA (mean over seeds)", ""]
    t1 = per_site[per_site.config.isin(["baseline_imgcond", "full"])]
    if not t1.empty:
        piv = t1.pivot(index="site", columns="config", values=cols)
        lines.append(piv.round(3).to_markdown())
        lines += ["", "Mean ± std across sites:", ""]
        agg = t1.groupby("config")[cols].agg(["mean", "std"]).round(3)
        lines.append(agg.to_markdown())
        if {"baseline_imgcond", "full"} <= set(t1.config) and "ssim" in cols:
            b = t1[t1.config == "baseline_imgcond"][cols].mean(); d = t1[t1.config == "full"][cols].mean()
            lines += ["", f"SSIM change {100*(d.ssim/b.ssim-1):+.1f}%, SoftEdge F1 change "
                      f"{100*(d.softedge_f1/b.softedge_f1-1):+.1f}%, horizon RMSE change "
                      f"{100*(d.horizon_rmse/b.horizon_rmse-1):+.1f}%  (paper: +83%, +25%, more than halved)"]
    lines += ["", "# Table 2: ablation (mean over sites and seeds)", ""]
    t2 = df[df.config.isin(ABLATION)].groupby("config")[cols].agg(["mean", "std"])
    if not t2.empty:
        t2 = t2.reindex([c for c in ABLATION if c in t2.index])
        flat = t2.xs("mean", axis=1, level=1)
        if "base_softedge" in flat.index and set(cols) == {"ssim", "horizon_rmse", "softedge_f1"}:
            base = flat.loc["base_softedge"].to_dict()
            rel = pd.DataFrame({k: relative_improvement(v.to_dict(), base) for k, v in flat.iterrows()}).T
            flat = flat.join((rel * 100).round(1))
        flat.index = [LABELS[i] for i in flat.index]
        lines.append(flat.round(3).to_markdown())
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
