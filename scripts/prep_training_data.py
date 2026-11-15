#!/usr/bin/env python
"""Build the Group I training set: image, DPT depth, PiDiNet soft edge, BLIP prompt.

Reuses whatever already exists (DESPINA_dataset depth_maps/, prompts/, dataset.csv)
and computes only what is missing. Writes parquet shards that both the LoRA and the
ControlNet training scripts read directly, plus a split file (paper: 2,900 / 600).

  python scripts/prep_training_data.py --images DESPINA_dataset/processed_apollo_imgs \
      --dataset-root DESPINA_dataset --out data/train_set --device cuda:0
"""
import argparse, csv, json, os, sys, hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from despina import constraints as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--dataset-root", default=None, help="folder with dataset.csv, depth_maps/, prompts/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--val-frac", type=float, default=600 / 3500)
    ap.add_argument("--shard", type=int, default=0, help="this worker's index")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--recompute-depth", action="store_true")
    ap.add_argument("--recompute-prompts", action="store_true")
    a = ap.parse_args()

    out = Path(a.out); (out / "cache").mkdir(parents=True, exist_ok=True)
    root = Path(a.dataset_root) if a.dataset_root else None
    csv_prompts = {}
    if root and (root / "dataset.csv").exists():
        for r in csv.DictReader(open(root / "dataset.csv")):
            csv_prompts[Path(r["filename"]).stem] = r["prompt"]
    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif"))
    files = files[a.shard::a.num_shards]
    rows = []
    for p in tqdm(files, desc=f"shard {a.shard}"):
        stem = p.stem
        cdir = out / "cache" / stem
        done = cdir / "meta.json"
        if done.exists():
            rows.append(json.load(open(done))); continue
        cdir.mkdir(exist_ok=True)
        img = Image.open(p).convert("RGB").resize((a.size, a.size), Image.BICUBIC)
        img.save(cdir / "image.png")
        dpath = root / "depth_maps" / f"{stem}.png" if root else None
        if dpath and dpath.exists() and not a.recompute_depth:
            depth = Image.open(dpath).convert("L").resize((a.size, a.size), Image.BICUBIC)
        else:
            depth = Image.fromarray(C.dpt_depth(img, a.device))
        depth.save(cdir / "depth.png")
        Image.fromarray(C.soft_edges(img, a.device)).save(cdir / "softedge.png")
        ann_file = root / "prompts" / f"{stem}_annotations.txt" if root else None
        if ann_file and ann_file.exists() and not a.recompute_prompts:
            ann = C.parse_annotation_file(ann_file)
            prompt = C.annotations_to_prompt(ann)
        elif stem in csv_prompts and not a.recompute_prompts:
            prompt = csv_prompts[stem]; ann = None
        else:
            ann = C.blip_annotations(img, a.device); prompt = C.annotations_to_prompt(ann)
        h = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 10_000 / 10_000
        meta = {"stem": stem, "text": prompt, "split": "val" if h < a.val_frac else "train",
                "category": C.terrain_category(ann) if ann else None}
        json.dump(meta, open(done, "w"))
        rows.append(meta)
    json.dump(rows, open(out / f"meta_shard{a.shard}.json", "w"))
    print(f"shard {a.shard}: {len(rows)} images")


if __name__ == "__main__":
    main()
