#!/usr/bin/env python
"""Patch diffusers' train_controlnet_sdxl.py (v0.31.0) so each training process keeps the
trainable ControlNet (weights + grads + Adam) on its own GPU and the frozen fp32 UNet on a
second GPU (cuda:local_rank + DESPINA_UNET_DEVICE_OFFSET). Autograd carries gradients across
devices. Without this, fp32 SDXL ControlNet training does not fit on a 24 GB P40.
Also moves the text encoders to CPU once prompt embeddings are cached, and computes those
embeddings 16 prompts at a time instead of 1000, and adds offset noise when
DESPINA_NOISE_OFFSET is set.

  python tools/patch_controlnet_split.py third_party/train_controlnet_sdxl.py third_party/train_controlnet_sdxl_split.py
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()

old_unet = "    unet.to(accelerator.device, dtype=weight_dtype)\n    text_encoder_one.to(accelerator.device, dtype=weight_dtype)"
new_unet = '''    _off = os.environ.get("DESPINA_UNET_DEVICE_OFFSET")
    if _off is not None:
        _ud = torch.device(f"cuda:{accelerator.local_process_index + int(_off)}")
        unet.to(_ud, dtype=weight_dtype)
        if args.gradient_checkpointing:
            unet.train()  # diffusers only checkpoints in train mode; SDXL UNet has no dropout
        _orig_unet_forward = unet.forward

        def _mv(x, d):
            if torch.is_tensor(x):
                return x.to(d)
            if isinstance(x, (list, tuple)):
                return type(x)(_mv(v, d) for v in x)
            if type(x) is dict:
                return {k: _mv(v, d) for k, v in x.items()}
            return x

        def _unet_forward(*a, **k):
            return _mv(_orig_unet_forward(*_mv(a, _ud), **_mv(k, _ud)), accelerator.device)

        unet.forward = _unet_forward
        logger.info(f"DESPINA split: controlnet on {accelerator.device}, unet on {_ud}")
    else:
        unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)'''
assert s.count(old_unet) == 1, "unet placement anchor not found"
s = s.replace(old_unet, new_unet)

old_del = "    del text_encoders, tokenizers\n"
assert s.count(old_del) == 1, "text encoder anchor not found"
s = s.replace(old_del, old_del + '    text_encoder_one.to("cpu")\n    text_encoder_two.to("cpu")\n')
old_map = "train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint)"
assert s.count(old_map) == 1, "embedding map anchor not found"
# the default map batch (1000 prompts) runs both text encoders at once and runs out of memory
s = s.replace(old_map, old_map.replace("batched=True,", "batched=True, batch_size=16,"))
# offset noise (as in the LoRA script): lets the model learn near-black regions like the lunar sky
old_noise = "                noise = torch.randn_like(latents)\n"
assert s.count(old_noise) == 1, "noise anchor not found"
s = s.replace(old_noise, old_noise + (
    "                _off = float(os.environ.get('DESPINA_NOISE_OFFSET', '0'))\n"
    "                if _off:\n"
    "                    noise += _off * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)\n"))
open(dst, "w").write(s)
print("patched ->", dst)
