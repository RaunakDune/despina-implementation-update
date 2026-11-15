#!/usr/bin/env python
"""Pack the prepared cache into parquet (HF datasets Image features), one folder per split."""
import argparse, json, sys
from pathlib import Path

from datasets import Dataset, Features, Image as HFImage, Value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    a = ap.parse_args()
    prep = Path(a.prep)
    rows = [json.load(open(p)) for p in sorted((prep / "cache").glob("*/meta.json"))]
    feats = Features({"image": HFImage(), "depth": HFImage(), "softedge": HFImage(),
                      "text": Value("string"), "stem": Value("string")})
    for split in ("train", "val"):
        sel = [r for r in rows if r["split"] == split]
        d = {"image": [], "depth": [], "softedge": [], "text": [], "stem": []}
        for r in sel:
            c = prep / "cache" / r["stem"]
            b = lambda f: {"bytes": (c / f).read_bytes(), "path": None}  # embed, not link
            d["image"].append(b("image.png")); d["depth"].append(b("depth.png"))
            d["softedge"].append(b("softedge.png")); d["text"].append(r["text"]); d["stem"].append(r["stem"])
        ds = Dataset.from_dict(d, features=feats)
        od = prep / split; od.mkdir(exist_ok=True)
        ds.to_parquet(str(od / "data.parquet"))
        print(split, len(ds))


if __name__ == "__main__":
    main()
