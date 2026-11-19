#!/usr/bin/env python
"""Group I, item 1: film-strip scans -> square, reseau-free 1024 px training images.

Output names follow DESPINA_dataset/dataset.csv (<frame>HR.png) so its prompts line up.
Resumable and parallel; writes processing.jsonl with the crop box and crosses inpainted.

  python scripts/process_apollo.py --raw data/apollo_raw --out data/apollo_processed --workers 24
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from despina.apollo import process_scan  # noqa: E402

ARGS = None


def work(p):
    dst = Path(ARGS.out) / f"{p.stem}HR.png"
    if dst.exists():
        return None
    try:
        img, info = process_scan(p, size=ARGS.size)
        img.save(dst)
        return {"frame": p.stem, **info}
    except Exception as e:  # noqa: BLE001
        return {"frame": p.stem, "error": repr(e)}


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default=None, help="comma-separated frame names")
    ap.add_argument("--link-dir", default=None,
                    help="also create this folder of symlinks to the frames that pass the quality check")
    ARGS = ap.parse_args()
    Path(ARGS.out).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(ARGS.raw).glob("*.png"))
    if ARGS.only:
        keep = set(ARGS.only.split(","))
        files = [f for f in files if f.stem in keep]
    log = open(Path(ARGS.out) / "processing.jsonl", "a")
    n = err = 0
    with Pool(ARGS.workers) as pool:
        for r in pool.imap_unordered(work, files, chunksize=2):
            if r is None:
                continue
            log.write(json.dumps(r) + "\n"); log.flush()
            n += 1; err += "error" in r
            if n % 100 == 0:
                print(f"{n} processed, {err} errors", flush=True)
    print(f"done: {n} processed, {err} errors", flush=True)
    log.close()
    rows = [json.loads(l) for l in open(Path(ARGS.out) / "processing.jsonl")]
    latest = {r["frame"]: r for r in rows}                       # last record per frame wins
    ok = sorted(f for f, r in latest.items() if r.get("qc_ok"))
    flagged = sorted(f for f, r in latest.items() if not r.get("qc_ok"))
    json.dump(flagged, open(Path(ARGS.out) / "flagged.json", "w"))
    print(f"quality check: {len(ok)} pass, {len(flagged)} flagged (flagged.json)", flush=True)
    if ARGS.link_dir:
        ld = Path(ARGS.link_dir); ld.mkdir(parents=True, exist_ok=True)
        for f in ok:
            dst = ld / f"{f}HR.png"
            if not dst.exists():
                dst.symlink_to((Path(ARGS.out) / f"{f}HR.png").resolve())
        print(f"linked {len(ok)} frames into {ld}", flush=True)


if __name__ == "__main__":
    main()
