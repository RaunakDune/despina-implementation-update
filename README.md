# DESPINA

Implementation of **DESPINA: Synthesis of High-Fidelity Planetary Horizon Reconstructions
Using DEM-Guided Diffusion** (ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial
Information Sciences, XI-2-2026, 881-890).

DESPINA turns a digital elevation model (DEM) into realistic surface photographs. It
ray-casts the DEM into an inverse-depth map and a shaded view whose soft edges outline the
terrain, and uses both to condition Stable Diffusion XL through ControlNet. A LoRA trained
on Apollo Hasselblad photography supplies the lunar appearance. The result is a
photograph-like view of any position and direction on the DEM whose skyline matches the
terrain.

This work is under active research, and as such code and implementation may change
regularly. Please be aware that DESPINA is a partial synthetic data pipeline designed to be
slotted into a larger system, such as a Visual Place Recognition model.

## Authors

* **Adam Nelson-Archer**, University of Houston
* **Raunak Sarbajna**, Department of Computer Science, University of Houston
* **Christoph F. Eick**, Department of Computer Science, University of Houston

Companion repositories:

* [UH-DAIS/DESPINA_dataset](https://github.com/UH-DAIS/DESPINA_dataset): `dataset.csv` (the
  1,876 training frames and their prompts), annotation files, and sample images with their
  DPT depth maps and label maps
* [UH-DAIS/DESPINA_technical_details](https://github.com/UH-DAIS/DESPINA_technical_details):
  DEM-to-image tooling and a sample lunar DEM

## Citation

```bibtex
@article{nelsonarcher2026despina,
  title   = {{DESPINA}: Synthesis of High-Fidelity Planetary Horizon Reconstructions Using {DEM}-Guided Diffusion},
  author  = {Nelson-Archer, Adam and Sarbajna, Raunak and Eick, Christoph F.},
  journal = {ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume  = {XI-2-2026},
  pages   = {881--890},
  year    = {2026},
  doi     = {10.5194/isprs-annals-XI-2-2026-881-2026}
}
```

## License

The code is released under the [MIT License](LICENSE). The paper is published under
CC BY 4.0. The diffusers training scripts that `setup_env.sh` downloads into `third_party/`
are Apache 2.0 and keep their own license.

## Current state (November 2026)

| Item | State |
|---|---|
| Training data | 1,561 Apollo frames rebuilt from the NASA scans (1,277 train, 284 validation) |
| Lunar LoRA | trained, 3,000 steps, offset noise 0.1 |
| Depth and soft-edge ControlNets | trained, 3,000 steps each, no offset noise |
| Horizon accuracy on the sample DEM | 0.94 px mean Horizon RMSE over 16 images (8 views, 2 seeds), black sky in every image |
| Known weakness | foreground ground is smoother than in Apollo photographs |
| Paper Tables 1 and 2 | need the five evaluation sites (DEM, camera pose and reference photograph for each) in `configs/sites.yaml` |

**The trained weights and the training set are on Hugging Face:**

* [RonInDune/despina-lunar-sdxl](https://huggingface.co/RonInDune/despina-lunar-sdxl): the
  lunar LoRA (`lora/`), the depth ControlNet (`controlnet-depth/`) and the soft-edge
  ControlNet (`controlnet-softedge/`)
* [RonInDune/despina-apollo-surface](https://huggingface.co/datasets/RonInDune/despina-apollo-surface):
  the 1,561 processed Apollo frames with their depth maps, soft edges and prompts (private;
  access on request, because ASU's scans are for research and education use only)

Download the weights into the repository with:

```bash
huggingface-cli download RonInDune/despina-lunar-sdxl --local-dir weights
```

and pass `--lora weights/lora --cn-depth weights/controlnet-depth --cn-softedge
weights/controlnet-softedge` to the scripts below in place of the `runs/main/...` paths.

A training run writes the models to these paths inside the repository:

| Model | Path | Notes |
|---|---|---|
| Lunar LoRA (use this) | `runs/main/lora_noff/pytorch_lora_weights.safetensors` | offset noise 0.1 |
| Depth ControlNet (use this) | `runs/main/cn_depth/` | |
| Soft-edge ControlNet (use this) | `runs/main/cn_softedge/` | |
| LoRA without offset noise | `runs/main/lora/` | cannot render a black sky; kept for comparison |
| ControlNets with offset noise | `runs/main/cn_depth_noff/`, `runs/main/cn_softedge_noff/` | overexpose with the LoRA; not recommended |
| Training set | `runs/main/train_set/{train,val}/data.parquet` | images, DPT depth, soft edges, prompts |

To retrain from scratch anywhere, follow [Setup](#setup) and run `bash run_all.sh`; the
defaults reproduce the recommended models.

## Quick start: synthesize a view from a DEM

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 python scripts/synthesize_view.py \
    --dem DESPINA_technical_details/TIF-to-image/dem_data.tif --upscale 4 \
    --frac 0.2005,0.6348 --heading 270 --sun-az 120 --sun-el 12 \
    --lora runs/main/lora_noff --cn-depth runs/main/cn_depth --cn-softedge runs/main/cn_softedge \
    --seeds 0,1 --out runs/synth/example --devices cuda:0,cuda:1
```

The camera position can be given as `--lonlat`, `--xy` (DEM map coordinates), `--rowcol`
(DEM pixels) or `--frac` (fractions of the DEM grid). `--heading` is the view azimuth
(0 = north, clockwise); `--hfov` defaults to 49 degrees, the Apollo Hasselblad 60 mm lens.
The output folder receives `dem_depth.png`, `dem_sim.png`, `dem_softedge.png`,
`dem_skyline.npy` (the DEM skyline row for every image column), `meta.json` and
`image_seed<k>.png`. `--render-only` writes the DEM constraints without loading SDXL. With
only one free GPU, `--devices cuda:0,cpu` keeps SDXL on the GPU and runs the ControlNets on
the CPU (several times slower).

The same steps from Python:

```python
from despina.dem import load_dem
from despina.raycast import Camera, RenderConfig, render_view
from despina import constraints as C
from despina.pipeline import Despina, GenConfig

dem = load_dem("dem.tif", upscale=4)
cam = Camera(row=500, col=800, heading_deg=270, width=1024, height=1024)
r = render_view(dem, cam, RenderConfig(sun_az_deg=120, sun_el_deg=12), device="cuda:0")
cond = {"depth": r.inv_depth, "softedge": C.soft_edges(r.sim_image, "cuda:0")}
model = Despina("cuda:0,cuda:1", lora_path="runs/main/lora_noff",
                controlnet_paths={"depth": "runs/main/cn_depth", "softedge": "runs/main/cn_softedge"})
img = model.generate(C.dem_prompt("hilly"), cond, GenConfig(), seed=0)
```

Views that look across flat ground toward relief, as most Apollo surface photographs do,
work best. `tools/find_views.py` searches a DEM for such viewpoints and prints site entries.

## Setup

```bash
git clone https://github.com/RaunakDune/despina-implementation-update despina && cd despina
git clone https://github.com/UH-DAIS/DESPINA_dataset
git clone https://github.com/UH-DAIS/DESPINA_technical_details
bash setup_env.sh          # Python 3.11 venv, torch 2.5.1+cu124, packages from requirements.txt
```

`setup_env.sh` downloads the two diffusers v0.31.0 training scripts into `third_party/` and
patches them (see [Training](#training)). Model weights download from Hugging Face into
`hf_cache/` on first use (SDXL base, the SDXL depth and soft-edge ControlNets, DPT,
PiDiNet, BLIP; about 30 GB in fp32). The code was developed on a server with seven
24 GB Tesla P40s and no job scheduler; on GPUs with fast fp16 the pipeline switches to half
precision automatically.

Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID` (the runners do) so CUDA's GPU numbers match
`nvidia-smi`.

## Running the whole pipeline

```bash
bash run_all.sh                                   # all phases, resumable
PHASES="data prep" bash run_all.sh                # rebuild the training set
PHASES="train" TRAIN_GPUS=0,1,2,3,4,5 bash run_all.sh
PHASES="render gen eval" SITES=configs/sites.yaml bash run_all.sh
```

| Phase | What it does | Output |
|---|---|---|
| data | downloads the frames in `dataset.csv` and turns each film-strip scan into a training image | `data/apollo_raw/`, `data/apollo_processed/`, `data/apollo_train_images/` |
| prep | DPT depth, PiDiNet soft edges and prompts, packed to parquet | `runs/main/train_set/` |
| train | LoRA, then depth ControlNet, then soft-edge ControlNet | `runs/main/lora`, `runs/main/cn_depth`, `runs/main/cn_softedge` |
| render | DEM constraints for every site in `SITES` | `runs/main/sites/<site>/` |
| gen | images for every configuration in `EXP` | `runs/main/gen/<site>/<config>/seed<k>.png` |
| eval | SSIM, Horizon RMSE and SoftEdge F1 in terrain masks; Tables 1 and 2 | `runs/main/eval/tables.md`, `per_site.csv`, `per_image.csv` |

`tools/gpuq.py` schedules the parallel jobs on free GPUs and skips finished ones, so any
phase can be rerun after an interruption. Training resumes from the latest checkpoint.
`ALL_GPUS` (default: every GPU) and `TRAIN_GPUS` (default: an even number of them) take
comma-separated `nvidia-smi` indices; ControlNet training needs an even count.

### Sites and experiments

`configs/sites.yaml` lists the views to synthesize and evaluate. Each site has a DEM
(GeoTIFF), an optional `upscale`, a camera (`lon`/`lat`, `x`/`y`, `row`/`col` or
`row_frac`/`col_frac`, plus `heading_deg`, `pitch_deg`, `hfov_deg`, `eye_height_m`), the sun
direction, a terrain `category` (flat, hilly or rocky, which selects the prompt) and,
for SSIM and SoftEdge F1, the `reference` photograph of that view and an optional
`exclude_mask` of non-terrain pixels. When a photograph's pointing is unknown,
`scripts/fit_view.py` estimates heading, pitch and field of view by matching its skyline to
DEM skylines.

`configs/experiments.yaml` defines the generation configurations: the five rows of Table 2
(`base_softedge`, `base_depth`, `lora_softedge`, `lora_depth`, `full`) and the
image-conditioned baseline of Table 1 (`baseline_imgcond`, which needs a reference photo).

Other configs: `sites_ablation.yaml` (the eight views on the sample DEM used below),
`sites_check.yaml` (four diagnostic views), `experiments_ablation.yaml`,
`experiments_check.yaml`, `experiments_texture.yaml`, and `human_study_paper_counts.csv`
(the perceptual-study counts for `scripts/human_study_stats.py`, which produces Tables 3
and 4).

## Training data

The frames in `DESPINA_dataset/dataset.csv` are rebuilt from Arizona State University's
[March to the Moon](https://tothemoon.im-ldi.com) archive of the Apollo Hasselblad film
(the `data` phase, or by hand):

```bash
python tools/download_apollo.py --names DESPINA_dataset/dataset.csv --out data/apollo_raw --size med --workers 12
python scripts/process_apollo.py --raw data/apollo_raw --out data/apollo_processed --workers 24 \
    --link-dir data/apollo_train_images
```

`despina/apollo.py` turns each film-strip scan into a 1024 px training image:

* B&W magazines are 16-bit greyscale scans and are scaled to 8 bits explicitly.
* The frame is located from the bright terrain, which reaches the frame's left, right and
  bottom edges, checked against the frame-to-strip width ratio (0.758). Where the reseau
  crosses are visible, the cross lattice is located in the whole scan and the frame edges
  are placed 7.5 mm outside the outermost exposed cross columns. This handles bright ground
  close-ups and the magazines whose aperture exposes only four of the five columns.
* The reseau crosses are found with a matched filter, fitted as a rigid grid, and removed by
  filling each 1-2 px line from the pixels beside it, which keeps the film grain.
* Colour scans get a partial grey-world white balance against film-dye casts.
* A frame is flagged (`qc_ok: false` in `processing.jsonl`, listed in `flagged.json`) when a
  crop edge shows film border or neither frame-finding method succeeded. Only frames that
  pass are linked into `data/apollo_train_images/`.

Of the 1,876 frames, 1,839 are available from the archive (magazine AS14-66 and frame
AS14-67-9374 are not served), 1,561 pass the quality check and 278 are flagged. Most
flagged frames are downward-looking ground close-ups (AS15-84, AS15-89, AS16-105, AS16-112,
AS17-139, AS17-144) whose crops still show the printed frame number. Prompts come from
`dataset.csv`; DPT depth maps are computed on the new crops. Previews are in
`docs/dataset/`.

## Training

| Model | Script | Settings |
|---|---|---|
| Lunar LoRA | `scripts/train_lora.sh` | SDXL base, rank 16, LR 1e-4 cosine, 1024 px, offset noise 0.1 (`NOISE_OFFSET`), 3,000 steps, batch 6 on six GPUs |
| Depth ControlNet | `KIND=depth scripts/train_controlnet.sh` | from `diffusers/controlnet-depth-sdxl-1.0`, DPT depth, LR 1e-5, 8-bit AdamW, 3,000 steps |
| Soft-edge ControlNet | `KIND=softedge scripts/train_controlnet.sh` | from `SargeZT/controlnet-sd-xl-1.0-softedge-dexined`, PiDiNet soft edges, same settings |

Both wrappers run the diffusers v0.31.0 example scripts through small patches applied on
every run:

* `tools/patch_lora_schedule.py` sizes the LR schedule for multi-GPU runs (accelerate steps
  the scheduler once per process).
* `tools/patch_controlnet_split.py` gives each training process two GPUs, with the
  ControlNet and optimizer on one and the frozen UNet on the other; computes prompt
  embeddings 16 at a time; and adds optional offset noise (`CN_NOISE_OFFSET`, default 0).

**Offset noise matters for the sky.** Without it, the fine-tuned SDXL cannot render the
black lunar sky: it frames images to avoid the sky or greys it out
(`docs/results/lora_diagnostic_no_offset_noise.png`). Offset noise in LoRA training fixes
this (`docs/results/lora_offset_noise_comparison.png`). Adding it to the ControlNets as well
overexposes the images (`docs/results/controlnet_offset_noise_comparison.png`), so the
ControlNets are trained without it.

## Evaluation

Section 4's metrics are in `despina/metrics.py`, all computed inside terrain masks: SSIM
against the reference photograph, Horizon RMSE between the DEM skyline and the skyline
extracted from the generated image, and SoftEdge F1 between PiDiNet edge maps.
`scripts/evaluate.py` writes the per-image values and Tables 1 and 2.

The skyline extractor runs dynamic programming over a dark-above, bright-below boundary
cost, with the sky/ground threshold set from each image's own levels. On the 17 sample
images with label maps it is within 1.7 px (at 256 px) on 15, with a median error of
0.6 px.

### Horizon-only ablation on the sample DEM

Eight views on the sample DEM (`configs/sites_ablation.yaml`: one hillside view and seven
flat-ground views from `tools/find_views.py`), two seeds, the recommended models, the
conditioning settings of the paper (depth 0.8, soft edge 0.5). Horizon RMSE in pixels at
512 px:

| Configuration | Mean | Median | Images within 3 px | Published (Table 2) |
|---|---|---|---|---|
| Base + Soft Edge | 83.8 | 68.9 | 6% | 5.21 |
| Base + Depth Map | 66.9 | 13.3 | 50% | 3.18 |
| LoRA + Soft Edge | 66.2 | 5.0 | 50% | 2.87 |
| LoRA + Depth Map | 45.6 | 1.3 | 69% | 2.43 |
| Full DESPINA system | 0.94 | 0.83 | 100% | 2.06 |

The full system reproduces the DEM skyline on every image, with a black sky
(`docs/results/ablation_horizon.png`; green is the DEM skyline, red the extracted one). For
the other configurations the number mixes genuine failures (terrain drawn into the sky, a
missing hill) with skylines the extractor cannot find under a grey or bright sky, so their
values are upper bounds on the geometric error. These views have no photographs, so SSIM
and SoftEdge F1 are not computed.

The ground texture test (`configs/experiments_texture.yaml`,
`docs/results/texture_test.png`) found that stopping the ControlNets early or lowering the
depth weight does not add surface detail; the paper's settings give the most texture.

## Repository layout

```
despina/     library: DEM loading and normals, ray caster, Apollo frame processing,
             constraint extractors, SDXL pipeline, metrics, statistics, site configuration
scripts/     data preparation, training wrappers, site rendering, generation, single-view
             synthesis, evaluation, view fitting, LoRA diagnostics, perceptual-study statistics
tools/       Apollo downloader, GPU job queue, training-script patches, viewpoint finder
configs/     site and experiment definitions
docs/        dataset previews, result sheets and early test images
run_all.sh   resumable end-to-end runner
```

## Paper-to-code map

| Paper (Fig. 4 item) | Code |
|---|---|
| I. Apollo photographs, fiducial removal (1) | `tools/download_apollo.py`, `despina/apollo.py`, `scripts/process_apollo.py` |
| I. DPT depth (2 to 3) | `despina/constraints.py: dpt_depth` (Intel/dpt-hybrid-midas, min-max inverse depth) |
| I. PiDiNet soft edges (4 to 5.1) | `constraints.py: soft_edges` (lllyasviel/Annotators PiDiNet) |
| I. BLIP prompts (6 to 7) | `constraints.py: blip_annotations, annotations_to_prompt` (same sentences as `dataset.csv`) |
| I. Training set | `scripts/prep_training_data.py`, `scripts/build_parquet.py` (train/validation split by hash) |
| I. LoRA (8) | `scripts/train_lora.sh` |
| I. ControlNets (9) | `scripts/train_controlnet.sh` |
| II. DEM load and upscale (10) | `despina/dem.py: load_dem, upscale_dem` |
| II. Horizon-aware constraint generation (11) | `despina/raycast.py: render_view` |
| Eq. 1 normals | `dem.py: surface_normals` |
| Eq. 2 adaptive far plane | `raycast.py` pre-pass (`far_mode="global"` or `"column"`) |
| Eq. 3 inverse-depth normalisation | `raycast.py` step 3 |
| Stratified skyline sampling | `raycast.py` step 2 (`supersample`, `band_px`) |
| II. Simulated render (12) and soft edges (5.2) | `raycast.py` step 4 (Lommel-Seeliger shading, cast shadows, optional mosaic drape) |
| III. Synthesis (14 to 18) | `despina/pipeline.py: Despina`, `scripts/generate.py`, `scripts/synthesize_view.py` |
| Classical DEM drape baseline | `RenderResult.sim_image` with `albedo:` set in the site |
| Image-conditioned baseline | config `baseline_imgcond` (DPT depth of the real photo) |
| Sec. 4 metrics | `despina/metrics.py`, `scripts/evaluate.py` (Tables 1 and 2) |
| Tables 3 and 4 | `despina/stats.py`, `scripts/human_study_stats.py` |
| Camera pose for a reference photo | `scripts/fit_view.py` (skyline matching) |

## Implementation notes

* **Precision.** Pascal GPUs have very slow fp16 and no bf16, so everything runs in fp32
  there. For inference, SDXL and the text encoders sit on one GPU and the ControlNets on a
  second (`--devices cuda:0,cuda:1`); each generation job uses a GPU pair.
* **ControlNet training memory.** In fp32 a full SDXL ControlNet with Adam needs about
  20 GB on top of the 10 GB UNet, hence the two-GPU split and 8-bit AdamW. Six P40s run
  three data-parallel processes (effective batch 3; `ACC=2` gives 6).
* **Depth tone mapping.** Terrain maps to [0.06, 1] and the sky to 0
  (`RenderConfig.terrain_floor`), so distant ridges stay distinct from the sky as they are
  in DPT maps of photographs. Set it to 0 for the plain Eq. 3 mapping.
* **Normals.** Eq. 1 is implemented as normalize(-dh/dx, -dh/dy, 1), pointing up and out.
* **Far plane.** The default normalises with the largest Eq. 2 far plane in the view
  (`far_mode="global"`); `far_mode="column"` applies Eq. 2 per azimuth.
* **DEM upscaling.** Bicubic interpolation of the height field.
* **Curvature.** The ray caster subtracts r^2 / 2R (lunar R = 1737.4 km).
* **Viewpoints.** Apollo surface photographs mostly look across flat ground toward relief.
  Views from steep slopes are harder for the models; `tools/find_views.py` finds
  Apollo-like viewpoints.

## Measured performance (Tesla P40, fp32)

* Constraint rendering, 1024 px with the stratified skyline band: about 3 s per view.
* Generation with SDXL on one P40 and the ControlNets on a second: about 150 s per 1024 px
  image at 30 steps.
* LoRA training: 7.5 s per step with one image per GPU (18.5 GB per P40); 3,000 steps on
  six P40s take about 6.3 h.
* ControlNet training: 8.5 s per step per two-GPU process; 3,000 steps take about 7 h per
  ControlNet on six P40s.
* Apollo data rebuild: download about 45 frames per minute; processing 1,839 frames on 24
  CPU cores in about 3 minutes.

## Early tests

`docs/early_tests/` holds images from the first runs, with a LoRA and ControlNets trained on
15 sample images. They show the pipeline working end to end, the effect of viewpoint (a
camera on a steep hillside fills the sky with terrain; flat-ground views keep the DEM
skyline), and that prompt wording alone ("Pitch black sky", available as `--prompt-suffix`
in `scripts/generate.py`) does not darken the sky.

Smoke test on the 17 sample images:

```bash
python scripts/render_sites.py --sites configs/sites.yaml --out runs/smoke/sites
python scripts/prep_training_data.py --images DESPINA_dataset/processed_apollo_imgs --dataset-root DESPINA_dataset --out runs/smoke/train_set
python scripts/build_parquet.py --prep runs/smoke/train_set
GPUS=0,1 DATA=runs/smoke/train_set OUT=runs/smoke/lora STEPS=20 bash scripts/train_lora.sh
KIND=depth GPUS=0,1 DATA=runs/smoke/train_set OUT=runs/smoke/cn_depth STEPS=10 bash scripts/train_controlnet.sh
python tools/gpuq.py configs/jobs_smoke.txt --state runs/smoke/queue --gpus 0,1,2,3
python scripts/evaluate.py --sites-dir runs/smoke/sites --gen runs/smoke/gen --out runs/smoke/eval
```
