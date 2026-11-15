#!/usr/bin/env python
"""Group III: generate images for sites x configs x seeds. Resumable (skips existing files).

  python scripts/generate.py --sites-dir runs/sites --experiments configs/experiments.yaml \
      --out runs/gen --devices cuda:0,cuda:1 --lora runs/lora/final --cn-depth runs/cn_depth/final
"""
import argparse, json, sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from despina.pipeline import Despina, GenConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-dir", required=True)
    ap.add_argument("--experiments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--devices", default="cuda:0")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--cn-depth", default=None, help="fine-tuned depth ControlNet (default: pretrained)")
    ap.add_argument("--cn-softedge", default=None)
    ap.add_argument("--configs", default=None, help="comma-separated subset of config names")
    ap.add_argument("--sites", default=None, help="comma-separated subset of site names")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--prompt-suffix", default="", help='appended to every site prompt, e.g. "Pitch black sky."')
    a = ap.parse_args()

    ex = yaml.safe_load(open(a.experiments))
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else ex["seeds"]
    cfgs = []
    for c in ex["configs"]:
        if a.configs and c["name"] not in a.configs.split(","):
            continue
        kw = {**ex.get("defaults", {}), **{k: v for k, v in c.items() if k != "sources"}}
        if a.steps: kw["steps"] = a.steps
        if a.size: kw["size"] = a.size
        cfgs.append((GenConfig(**kw), c["sources"]))
    if not a.lora:
        dropped = [g.name for g, _ in cfgs if g.use_lora]
        cfgs = [(g, s) for g, s in cfgs if not g.use_lora]
        if dropped:
            print("No --lora given; skipping", dropped)
    cn_paths = {k: v for k, v in {"depth": a.cn_depth, "softedge": a.cn_softedge}.items() if v}
    model = Despina(a.devices, lora_path=a.lora, controlnet_paths=cn_paths)
    site_dirs = sorted(p for p in Path(a.sites_dir).iterdir() if (p / "meta.json").exists())
    if a.sites:
        site_dirs = [p for p in site_dirs if p.name in a.sites.split(",")]
    for sd in site_dirs:
        meta = json.load(open(sd / "meta.json"))
        for g, sources in cfgs:
            cond, ok = {}, True
            for ctl, src in sources.items():
                f = sd / f"{src}_{ctl}.png"
                if not f.exists():
                    ok = False; break
                cond[ctl] = Image.open(f)
            if not ok:
                print(f"skip {sd.name}/{g.name}: missing {src} condition"); continue
            for seed in seeds:
                of = Path(a.out) / sd.name / g.name / f"seed{seed}.png"
                if of.exists():
                    continue
                of.parent.mkdir(parents=True, exist_ok=True)
                img = model.generate((meta["prompt"] + " " + a.prompt_suffix).strip(), cond, g, seed=seed)
                img.save(of)
                print("wrote", of, flush=True)


if __name__ == "__main__":
    main()
