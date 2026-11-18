#!/usr/bin/env python
"""Patch diffusers' train_text_to_image_lora_sdxl.py (v0.31.0) so the LR schedule spans the run.

Accelerate's prepared scheduler steps once per process for every optimizer step, but this
script sizes the schedule for one process. With six GPUs a cosine schedule ran three full
cycles and ended back at the peak learning rate. Scale warmup and length by num_processes,
as the ControlNet script does.

  python tools/patch_lora_schedule.py third_party/train_text_to_image_lora_sdxl.py third_party/train_text_to_image_lora_sdxl_fixed.py
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
old = """        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,"""
new = """        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps * accelerator.num_processes,"""
assert s.count(old) == 1, "scheduler anchor not found"
open(dst, "w").write(s.replace(old, new))
print("patched ->", dst)
