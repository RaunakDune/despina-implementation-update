"""Image-side constraint extractors (DESPINA Group I, items 2-7).

  * DPT monocular depth (item 2 -> item 3): training conditions and the
    image-conditioned baseline.
  * PiDiNet soft edges (item 4 -> items 5.1 / 5.2): on Apollo photos for
    training, on DEM simulated renders for synthesis, and for SoftEdge F1.
  * BLIP VQA annotations -> structured prompts (item 6 -> item 7).
Models load lazily so the DEM code never needs them.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict

import numpy as np
import torch
from PIL import Image

DPT_MODEL = "Intel/dpt-hybrid-midas"
PIDINET_REPO = "lllyasviel/Annotators"
BLIP_VQA_MODEL = "Salesforce/blip-vqa-base"


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return Image.fromarray(a.astype(np.uint8)).convert("RGB")


# ---- DPT ------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _dpt(model_id: str, device: str):
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    proc = DPTImageProcessor.from_pretrained(model_id)
    model = DPTForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return proc, model


@torch.no_grad()
def dpt_depth(img, device="cuda", model_id: str = DPT_MODEL) -> np.ndarray:
    """Relative inverse depth, min-max normalised to uint8 (near = white), same size as input."""
    pil = _to_pil(img)
    proc, model = _dpt(model_id, str(device))
    inp = proc(images=pil, return_tensors="pt").to(device)
    pred = model(**inp).predicted_depth[:, None]
    pred = torch.nn.functional.interpolate(pred, size=pil.size[::-1], mode="bicubic",
                                           align_corners=False)[0, 0]
    lo, hi = pred.min(), pred.max()
    out = ((pred - lo) / (hi - lo + 1e-8) * 255).clamp(0, 255)
    return out.byte().cpu().numpy()


# ---- PiDiNet --------------------------------------------------------------------
@lru_cache(maxsize=2)
def _pidinet(device: str):
    from controlnet_aux import PidiNetDetector
    det = PidiNetDetector.from_pretrained(PIDINET_REPO)
    det.netNetwork.to(device)
    return det


def soft_edges(img, device="cuda", safe: bool = False, detect_resolution: int = 512) -> np.ndarray:
    """PiDiNet soft-edge map, uint8, same size as the input."""
    pil = _to_pil(img)
    det = _pidinet(str(device))
    out = det(pil, detect_resolution=detect_resolution, image_resolution=max(pil.size),
              safe=safe, apply_filter=False, output_type="pil")
    out = out.convert("L").resize(pil.size, Image.BILINEAR)
    return np.array(out)


# ---- BLIP structured prompts ---------------------------------------------------
# Questions asked of BLIP-VQA and the sentence each "yes" adds. Sentences and
# order follow DESPINA_dataset/dataset.csv exactly so new captions match the
# released ones (including its "Darks shadows" wording). Questions were chosen by agreement
# with the released *_annotations.txt files: 61/64 answers match on the 8 annotated samples.
PROMPT_BASE = "A realistic image of the Moon's surface, from the perspective of a rover."
ANNOTATIONS = [
    ("hill", "Is there a hill in the image?",
     "A lunar landscape with hills in the background."),
    ("rocks", "Are there rocks in the image?",
     "Scattered rocks visible on the lunar surface."),
    ("crater", "Is there a crater in the image?", "A lunar crater is in view."),
    ("rover", "Is there a rover or vehicle in the image?", "A rover is present in the image."),
    ("shadow", "Are there shadows in the image?", "Darks shadows cast by the terrain."),
    ("horizon", "Is the horizon a straight flat line?", "A straight, flat horizon line on the surface."),
    ("astronaut", "Is there a person in the image?", "An astronaut is visible in the image."),
    ("equipment", "Is there equipment or machinery in the image?",
     "Visible NASA equipment or astronauts in the image."),
]


@lru_cache(maxsize=1)
def _blip(device: str, model_id: str = BLIP_VQA_MODEL):
    from transformers import BlipForQuestionAnswering, BlipProcessor
    proc = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id).to(device).eval()
    return proc, model


@torch.no_grad()
def blip_annotations(img, device="cuda") -> Dict[str, bool]:
    pil = _to_pil(img)
    proc, model = _blip(str(device))
    out = {}
    for key, q, _ in ANNOTATIONS:
        inp = proc(pil, q, return_tensors="pt").to(device)
        ans = proc.decode(model.generate(**inp, max_new_tokens=3)[0], skip_special_tokens=True)
        out[key] = ans.strip().lower().startswith("yes")
    return out


def annotations_to_prompt(ann: Dict[str, bool], base: str = PROMPT_BASE) -> str:
    parts = [base] + [s for k, _, s in ANNOTATIONS if ann.get(k)]
    return " ".join(parts)


def parse_annotation_file(path: str) -> Dict[str, bool]:
    ann = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1)
            ann[k.strip()] = v.strip().lower() == "yes"
    return ann


def terrain_category(ann: Dict[str, bool]) -> str:
    """FLAT / HILLY / ROCKY categories used in the perceptual study (Fig. 7)."""
    if ann.get("hill"):
        return "hilly"
    if ann.get("rocks"):
        return "rocky"
    return "flat"


def dem_prompt(category: str = "flat", extra: str = "") -> str:
    """Terrain-only prompt for DEM synthesis (no rover/astronaut/equipment)."""
    ann = {"hill": category == "hilly", "rocks": category == "rocky",
           "shadow": True, "horizon": category == "flat"}
    p = annotations_to_prompt(ann)
    return (p + " " + extra).strip()


NEGATIVE_PROMPT = ("clouds, blue sky, bright sky, sunset, atmosphere, haze, fog, vegetation, "
                   "water, trees, buildings, text, watermark, fiducial marks, crosshair, blurry")
