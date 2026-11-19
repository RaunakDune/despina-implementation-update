#!/usr/bin/env python
"""Tiny resumable GPU job queue for a scheduler-less server.

jobs file: one job per line, "<n_gpus> <shell command>"; "{GPUS}" in the command is
replaced by the assigned comma-separated GPU ids and CUDA_VISIBLE_DEVICES is NOT set
(so commands can use cuda:0,cuda:1 after we set it). Done jobs are recorded in
<state>/done/<hash>; rerunning the queue skips them. A GPU counts as free when the
queue has not assigned it and nvidia-smi reports < FREE_MB used.

  python tools/gpuq.py jobs.txt --state runs/queue --gpus 0,1,2,3
"""
import argparse, hashlib, os, subprocess, time
from pathlib import Path

FREE_MB = 1500


def used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--state", default="runs/queue")
    ap.add_argument("--gpus", default=None, help="comma-separated nvidia-smi indices (default: all)")
    ap.add_argument("--poll", type=float, default=20)
    a = ap.parse_args()
    st = Path(a.state); (st / "done").mkdir(parents=True, exist_ok=True); (st / "logs").mkdir(exist_ok=True)
    allowed = [int(g) for g in a.gpus.split(",")] if a.gpus else sorted(used_mb())
    jobs = []
    for line in open(a.jobs):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        n, cmd = line.split(" ", 1)
        h = hashlib.sha1(cmd.encode()).hexdigest()[:12]
        if not (st / "done" / h).exists():
            jobs.append((int(n), cmd, h))
    running = {}   # pid -> (proc, gpus, h, cmd)
    print(f"{len(jobs)} jobs pending", flush=True)
    while jobs or running:
        for pid, (p, gs, h, cmd) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[pid]
                if rc == 0:
                    (st / "done" / h).write_text(cmd)
                    print(f"[done] {h} on {gs}", flush=True)
                else:
                    print(f"[FAIL rc={rc}] {h}: see {st}/logs/{h}.log", flush=True)
        busy = {g for _, gs, _, _ in running.values() for g in gs}
        mem = used_mb()
        free = [g for g in allowed if g not in busy and mem.get(g, 1e9) < FREE_MB]
        i = 0
        while i < len(jobs):
            n, cmd, h = jobs[i]
            if n <= len(free):
                gs, free = free[:n], free[n:]
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(map(str, gs)))
                c = cmd.replace("{GPUS}", ",".join(str(k) for k in range(n)))
                log = open(st / "logs" / f"{h}.log", "a")
                p = subprocess.Popen(c, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                                     executable="/bin/bash")
                running[p.pid] = (p, gs, h, cmd)
                print(f"[start] {h} on GPUs {gs}: {cmd[:120]}", flush=True)
                jobs.pop(i)
            else:
                i += 1
        time.sleep(a.poll)
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
