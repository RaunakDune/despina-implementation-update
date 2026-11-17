#!/usr/bin/env python
"""Download Apollo Hasselblad surface frames from ASU's March to the Moon archive
(https://tothemoon.im-ldi.com), which serves every frame as a processed film-strip scan.

Frame names come from DESPINA_dataset/dataset.csv (e.g. AS17-147-22551HR.png) or a text file
with one name per line. Resumable: finished files are skipped, partial files are retried.

  python tools/download_apollo.py --names DESPINA_dataset/dataset.csv --out data/apollo_raw --size med
Sizes: small (~1100 px wide), med (~3500 px), full (~14000 px, about 300 MB each).
"""
import argparse, csv, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://tothemoon.im-ldi.com/data_a70"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def frame_names(path):
    p = Path(path)
    if p.suffix == ".csv":
        names = [r["filename"] for r in csv.DictReader(open(p))]
    else:
        names = [l.strip() for l in open(p) if l.strip()]
    out = []
    for n in names:
        m = re.match(r"(AS\d+-\d+-\d+)", Path(n).stem)
        if m:
            out.append(m.group(1))
    return sorted(set(out))


def url_for(name, size):
    mission = name.split("-")[0]
    if size == "full":
        return f"{BASE}/{mission}/processed/{name}.png"
    return f"{BASE}/{mission}/extra/{name}.{size}.png"


def fetch(name, out, size, retries=4):
    dst = out / f"{name}.png"
    if dst.exists() and dst.stat().st_size > 1000:
        return name, "skip"
    tmp = dst.with_suffix(".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url_for(name, size), headers={"User-Agent": "DESPINA-research/1.0 (University of Houston)"})
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            with open(tmp, "rb") as f:
                if f.read(8) != PNG_MAGIC:
                    raise IOError("not a PNG")
            os.replace(tmp, dst)
            return name, "ok"
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(5 * (attempt + 1))
    return name, f"FAIL {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="med", choices=["small", "med", "full"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = frame_names(a.names)
    print(f"{len(names)} frames", flush=True)
    done = fail = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch, n, out, a.size) for n in names]
        for i, f in enumerate(as_completed(futs), 1):
            n, st = f.result()
            if st.startswith("FAIL"):
                fail += 1
                print(n, st, flush=True)
            else:
                done += 1
            if i % 50 == 0:
                print(f"{i}/{len(names)} done={done} fail={fail}", flush=True)
    print(f"finished: {done} ok, {fail} failed", flush=True)


if __name__ == "__main__":
    main()
