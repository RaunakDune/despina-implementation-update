"""Image synthesis (DESPINA Group III, items 14-18): SDXL + lunar LoRA + ControlNets.

Pascal GPUs (P40/P5000) have very slow fp16 and no bf16, so on compute
capability < 7 everything runs in fp32. fp32 SDXL with two ControlNets does
not fit on one 24 GB card, so the ControlNets are placed on a second GPU and
bridged; pass two devices (e.g. "cuda:0,cuda:1") to enable this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from .constraints import NEGATIVE_PROMPT

SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNETS = {
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
    "softedge": "SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
}


@dataclass
class GenConfig:
    name: str = "full"
    use_lora: bool = True
    controls: List[str] = field(default_factory=lambda: ["depth", "softedge"])
    control_scales: List[float] = field(default_factory=lambda: [0.8, 0.5])
    control_end: float = 1.0
    steps: int = 30
    guidance: float = 7.0
    lora_scale: float = 0.9
    size: int = 1024


def pick_dtype(device: str) -> torch.dtype:
    d = torch.device(device)
    if d.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(d)
    return torch.float16 if major >= 7 else torch.float32


def _bridge(module: torch.nn.Module, home: torch.device):
    """Run `module` on its own device; move inputs there and outputs back to `home`."""
    orig = module.forward
    dev = next(module.parameters()).device

    def mv(x, d):
        if torch.is_tensor(x):
            return x.to(d)
        if isinstance(x, (list, tuple)):
            return type(x)(mv(v, d) for v in x)
        if isinstance(x, dict):
            return {k: mv(v, d) for k, v in x.items()}
        return x

    def fwd(*args, **kwargs):
        out = orig(*mv(args, dev), **mv(kwargs, dev))
        return mv(out, home)

    module.forward = fwd
    return module


class Despina:
    def __init__(self, devices: str = "cuda:0", lora_path: Optional[str] = None,
                 controlnet_paths: Optional[dict] = None, base: str = SDXL_BASE,
                 controls=("depth", "softedge"), cache_dir: Optional[str] = None):
        from diffusers import (ControlNetModel, StableDiffusionXLControlNetPipeline,
                               EulerDiscreteScheduler)
        devs = [torch.device(d.strip()) for d in devices.split(",")]
        self.home = devs[0]
        self.cn_dev = devs[1] if len(devs) > 1 else devs[0]
        self.dtype = pick_dtype(str(self.home))
        paths = {**CONTROLNETS, **(controlnet_paths or {})}
        self.control_names = list(controls)
        nets = [ControlNetModel.from_pretrained(paths[c], torch_dtype=self.dtype, cache_dir=cache_dir)
                for c in self.control_names]
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base, controlnet=nets if len(nets) > 1 else nets[0], torch_dtype=self.dtype,
            cache_dir=cache_dir, add_watermarker=False)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        for name in ("unet", "vae", "text_encoder", "text_encoder_2"):
            getattr(pipe, name).to(self.home)
        for n in nets:
            n.to(self.cn_dev)
            if self.cn_dev != self.home:
                _bridge(n, self.home)
        pipe.vae.enable_tiling()
        self.lora_loaded = False
        if lora_path:
            pipe.load_lora_weights(lora_path, adapter_name="lunar")
            pipe.unet.to(self.home)
            pipe.text_encoder.to(self.home)
            pipe.text_encoder_2.to(self.home)
            self.lora_loaded = True
        pipe.set_progress_bar_config(disable=True)
        # diffusers infers the execution device from whichever component it finds first, which
        # can be a bridged ControlNet on the second GPU. Pin it to the SDXL GPU.
        home = self.home
        pipe.__class__ = type("PinnedSDXLControlNetPipeline", (pipe.__class__,),
                              {"_execution_device": property(lambda _self: home)})
        self.pipe = pipe

    @torch.no_grad()
    def generate(self, prompt: str, conditions: dict, cfg: GenConfig, seed: int = 0,
                 negative_prompt: str = NEGATIVE_PROMPT) -> Image.Image:
        if self.lora_loaded:
            if cfg.use_lora:
                self.pipe.set_adapters(["lunar"], adapter_weights=[cfg.lora_scale])
            else:
                self.pipe.set_adapters(["lunar"], adapter_weights=[0.0])
        elif cfg.use_lora:
            raise RuntimeError(f"Config {cfg.name} needs a LoRA but none was loaded.")
        images, scales = [], []
        for name in self.control_names:
            if name in cfg.controls:
                im = conditions[name]
                scales.append(cfg.control_scales[cfg.controls.index(name)])
            else:  # control not used in this config: blank image, zero weight
                im = np.zeros((cfg.size, cfg.size), np.uint8)
                scales.append(0.0)
            im = Image.fromarray(np.asarray(im).astype(np.uint8)).convert("RGB").resize((cfg.size, cfg.size), Image.BICUBIC)
            images.append(im)
        g = torch.Generator(device="cpu").manual_seed(seed)
        out = self.pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            image=images if len(images) > 1 else images[0],
            controlnet_conditioning_scale=scales if len(scales) > 1 else scales[0],
            control_guidance_end=cfg.control_end,
            num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
            height=cfg.size, width=cfg.size, generator=g)
        return out.images[0]
