# DESPINA: implementation

Code for **DESPINA: Synthesis of High-Fidelity Planetary Horizon Reconstructions Using
DEM-Guided Diffusion** (ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial
Information Sciences, XI-2-2026, 881-890).
DESPINA ray-casts a digital elevation model (DEM) into inverse-depth and soft-edge maps and
uses them to condition Stable Diffusion XL through ControlNet. A LoRA trained on Apollo
surface photography supplies the lunar appearance.

This work under active research, and as such code and implementation may change regularly. Please
be aware that DESPINA is a partial synthetic data pipeline designed to be slotted into a larger
system, such a Visual Place Recognition model.

## Authors

DESPINA was designed and published by:

* **Adam Nelson-Archer**, University of Houston
* **Raunak Sarbajna**, Department of Computer Science, University of Houston
* **Christoph F. Eick**, Department of Computer Science, University of Houston

This repository reimplements the method from the paper. The original data and technical
material are in the authors' repositories:

* [UH-DAIS/DESPINA_dataset](https://github.com/UH-DAIS/DESPINA_dataset): processed Apollo
  photographs, DPT depth maps, label maps and BLIP prompts
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

## Status

The full pipeline runs end to end on the 17 sample images in `DESPINA_dataset` and the
sample DEM in `DESPINA_technical_details`. The tables in the paper have not been
reproduced yet: that needs the full training set (about 3,500 images in the paper; 1,876
captioned in `dataset.csv`), the original checkpoints, and the camera poses and reference
photographs for the five evaluation sites. See [Diagnostics so far](#diagnostics-so-far).

## Repository layout

```
despina/     library: DEM loading and normals, ray caster, constraint extractors,
             SDXL pipeline, metrics, statistics, site configuration
scripts/     data preparation, training wrappers, site rendering, generation,
             evaluation, view fitting, perceptual-study statistics
tools/       GPU job queue, ControlNet two-GPU patch, Apollo-like viewpoint finder
configs/     experiment and site definitions (paper configs and diagnostic configs)
docs/        contact sheets from the smoke tests
run_all.sh   resumable end-to-end runner
```

`setup_env.sh`, `run_all.sh` and the scripts expect the two data repositories cloned into
the repository root:

```bash
git clone https://github.com/UH-DAIS/DESPINA_dataset
git clone https://github.com/UH-DAIS/DESPINA_technical_details
```

## Paper-to-code map

| Paper (Fig. 4 item) | Code |
|---|---|
| I. DPT depth (2 to 3) | `despina/constraints.py: dpt_depth` (Intel/dpt-hybrid-midas, min-max inverse depth) |
| I. PiDiNet soft edges (4 to 5.1) | `constraints.py: soft_edges` (lllyasviel/Annotators PiDiNet) |
| I. BLIP prompts (6 to 7) | `constraints.py: blip_annotations, annotations_to_prompt` (same sentences as `dataset.csv`) |
| I. Training set | `scripts/prep_training_data.py`, `scripts/build_parquet.py` (2,900 / 600 split by hash) |
| I. LoRA (8) | `scripts/train_lora.sh` (diffusers v0.31.0 example script, unmodified) |
| I. ControlNets (9) | `scripts/train_controlnet.sh` (fine-tunes pretrained SDXL depth and soft-edge ControlNets) |
| II. DEM load and upscale (10) | `despina/dem.py: load_dem, upscale_dem` |
| II. Horizon-aware constraint generation (11) | `despina/raycast.py: render_view` |
| Eq. 1 normals | `dem.py: surface_normals` |
| Eq. 2 adaptive far plane | `raycast.py` pre-pass (`far_mode="global"` or `"column"`) |
| Eq. 3 inverse-depth normalisation | `raycast.py` step 3 |
| Stratified skyline sampling | `raycast.py` step 2 (`supersample`, `band_px`) |
| II. Simulated render (12) and soft edges (5.2) | `raycast.py` step 4 (Lommel-Seeliger shading, cast shadows, optional mosaic drape) |
| III. Synthesis (14 to 18) | `despina/pipeline.py: Despina`, `scripts/generate.py` |
| Classical DEM drape baseline | `RenderResult.sim_image` with `albedo:` set in `sites.yaml` |
| Image-conditioned baseline | config `baseline_imgcond` (DPT depth of the real photo) |
| Sec. 4 metrics | `despina/metrics.py`, `scripts/evaluate.py` (Tables 1 and 2) |
| Tables 3 and 4 | `despina/stats.py`, `scripts/human_study_stats.py` |
| Camera pose for a reference photo | `scripts/fit_view.py` (skyline matching) |

## Setup

```bash
git clone https://github.com/RaunakDune/despina-implementation-update despina && cd despina
git clone https://github.com/UH-DAIS/DESPINA_dataset
git clone https://github.com/UH-DAIS/DESPINA_technical_details
bash setup_env.sh          # Python 3.11 venv, torch 2.5.1+cu124, diffusers 0.31.0, bitsandbytes
```
`setup_env.sh` also downloads the two diffusers v0.31.0 training scripts into `third_party/`
and patches the ControlNet one for two-GPU training. Model weights are cached in
`hf_cache/` inside the repository. The code was developed on a server with seven Tesla P40s (7x Tesla P40, 1x Quadro
P5000, no job scheduler); on GPUs with fast fp16 the pipeline switches to half precision.

## Running

```bash
# everything, resumable
bash run_all.sh
# or one phase at a time
PHASES="prep" bash run_all.sh
PHASES="train" LORA_STEPS=3000 CN_STEPS=3000 bash run_all.sh
PHASES="render gen eval" bash run_all.sh
```
Results: `runs/main/eval/tables.md`, `per_site.csv`, `per_image.csv`; images in `runs/main/gen/<site>/<config>/seed*.png`.

Generation configs (`configs/experiments.yaml`) are the five ablation rows of Table 2 plus
the image-conditioned baseline of Table 1. Sites (`configs/sites.yaml`) take a DEM, a camera
position (lon/lat, map x/y or DEM row/col), heading, pitch, field of view, sun direction and,
for SSIM and SoftEdge F1, the reference photo of that view.

## Implementation decisions

* **Precision.** Pascal GPUs have very slow fp16 and no bf16, so everything runs in fp32.
  For inference, SDXL and the text encoders sit on one GPU and the ControlNets on a second
  (`--devices cuda:0,cuda:1`). Each generation job uses a GPU pair.
* **ControlNet training memory.** In fp32 a full SDXL ControlNet with Adam needs about 20 GB
  on top of the 10 GB UNet. `scripts/train_controlnet.sh` therefore gives every training
  process two GPUs (ControlNet and optimizer on one, frozen UNet on the other, via
  `tools/patch_controlnet_split.py`) and uses 8-bit AdamW (bitsandbytes). Six P40s run three
  data-parallel processes, so the effective batch is 3 (`ACC=2` gives the paper's 6).
* **Depth tone mapping.** Exact Eq. 3 sends terrain beyond a few hundred metres to black, the
  same value as the sky. The ControlNet is trained on DPT maps, where distant ridges sit just
  above the sky. `RenderConfig.terrain_floor` (default 0.06) maps terrain to [0.06, 1] and
  the sky to 0. Set it to 0 for the literal Eq. 3.
* **Eq. 1 sign.** Implemented as normalize(-dh/dx, -dh/dy, 1), so normals point up and out.
* **Far plane.** Eq. 2 is defined per azimuth. Normalising each column separately produces
  vertical banding, so the default uses the maximum over columns (`far_mode="global"`).
  The per-column version is available.
* **DEM upscaling.** Bicubic interpolation of the height field. The paper cites Liu et al.
  (2020b), which evaluates DEM accuracy and is not an upscaling method.
* **Curvature.** The ray caster subtracts r^2 / 2R (lunar R = 1737.4 km).
* **Skyline extraction** for Horizon RMSE uses dynamic programming over a dark-sky boundary
  cost. On the 17 released samples it is within 1.7 px (at 256 px) of the label-map skyline
  on 16 of 17 images. The miss has a hill in deep shadow against the black sky.

## Findings while reproducing the paper

* **Table 2 "Relative Improvement".** Four of the five published values are the SSIM gain
  alone (0.462/0.378 = +22.2%, 0.489/0.378 = +29.4%, 0.534/0.378 = +41.3%), not the
  composite the footnote describes. The full system's +47.1% matches neither reading
  (SSIM alone gives +49.2%). `evaluate.py` reports both readings.
* **Tables 3 and 4** are reproduced exactly with 36 judgements per method and category, using
  Wilson intervals, an exact binomial test against chance and a pooled two-proportion z-test.
  The text says 35 participants.
* **Dataset size.** `DESPINA_dataset/dataset.csv` lists 1,876 captioned images (AS11 to AS17).
  The paper says about 3,500, and the repository contains 17 images.

## Measured performance

* Constraint rendering (1024 px, 3x3 stratified skyline band): about 3 s per view on a P40.
* Generation, fp32, SDXL on one P40 and ControlNets on a second: about 100 s per 1024 px
  image at 20 steps (about 150 s at 30). 6 configs x 3 seeds x 5 sites = 90 images, about
  1.3 h on three GPU pairs.
* LoRA training: 7.6 s/step with one image per GPU (18.5 GB per P40). 3,000 steps on six
  P40s (batch 6) is about 6.5 h.
* ControlNet training (split mode, 8-bit AdamW): 8.4 s/step per two-GPU process. 3,000 steps
  on six P40s (batch 3) is about 7 h per ControlNet.
* Without the lunar LoRA, SDXL treats the black sky of the DEM depth map as distant terrain
  and fills it in (rows "Base + ..."), which is the failure mode the paper attributes to
  base models. The image-conditioned baseline keeps a black sky.

## Smoke test

```bash
python scripts/render_sites.py --sites configs/sites.yaml --out runs/smoke/sites
python scripts/prep_training_data.py --images DESPINA_dataset/processed_apollo_imgs --dataset-root DESPINA_dataset --out runs/smoke/train_set
python scripts/build_parquet.py --prep runs/smoke/train_set
GPUS=1,2 DATA=runs/smoke/train_set OUT=runs/smoke/lora STEPS=20 bash scripts/train_lora.sh
KIND=depth GPUS=1,2 DATA=runs/smoke/train_set OUT=runs/smoke/cn_depth STEPS=10 bash scripts/train_controlnet.sh
python tools/gpuq.py configs/jobs_smoke.txt --state runs/smoke/queue --gpus 1,2,6,7
python scripts/evaluate.py --sites-dir runs/smoke/sites --gen runs/smoke/gen --out runs/smoke/eval
```

## Rebuilding the training images from the NASA scans

The 17 sample images in `DESPINA_dataset` are all that is published of the training set, so
the frames listed in its `dataset.csv` can be rebuilt from Arizona State University's
[March to the Moon](https://tothemoon.im-ldi.com) archive of the Apollo Hasselblad film:

```bash
python tools/download_apollo.py --names DESPINA_dataset/dataset.csv --out data/apollo_raw --size med --workers 12
python scripts/process_apollo.py --raw data/apollo_raw --out data/apollo_processed --workers 24
python scripts/prep_training_data.py --images data/apollo_processed --dataset-root DESPINA_dataset \
    --out data/train_set --recompute-depth           # one process per GPU with --shard/--num-shards
python scripts/build_parquet.py --prep data/train_set
```

`despina/apollo.py` turns each film-strip scan into a training image:

* B&W magazines are 16-bit greyscale scans and are scaled to 8 bits explicitly.
* The 55 mm frame is located from the bright terrain, which reaches the frame's left, right
  and bottom edges, and checked against the frame-to-strip width ratio (0.758).
* The 5 x 5 reseau crosses are found with a matched filter, fitted as a rigid grid, and
  removed by filling each 1-2 px line from the pixels beside it, which keeps film grain.
* Colour scans get a grey-world white balance, since the scans carry film-dye casts.

`processing.jsonl` records the crop and two quality flags per frame; frames where the crop
fell back to the size prior (`qc_ok: false`) should be checked by eye. ASU does not serve
magazine AS14-66 (36 frames). Depth maps are recomputed because the crops differ from the
released ones; prompts come from `dataset.csv`.

## Diagnostics so far

These runs used a LoRA and ControlNets trained on 15 of the 17 sample images (300 steps
each), so they test the pipeline rather than the method.

* `docs/smoke_test/contact_base_vs_lora20.png`, `lora17_contact.png`: on a camera standing on
  a steep hillside, every configuration fills the sky with terrain. The LoRA changes the
  colour and texture to match Apollo Hasselblad frames but does not restore the sky.
* `contact_pretrainedCN.png`: from flat ground looking toward a hill (the framing of most
  Apollo surface photos, found with `tools/find_views.py`), the generated hill follows the
  DEM skyline. The sky is pale grey-teal rather than black. Weaker soft-edge conditioning
  or stopping the ControlNets at 60% of the steps changes little.
* `contact_finetunedCN.png`: ControlNets fine-tuned on the samples darken the sky to slate
  grey and make the terrain closer to the Apollo photographs. The hillside view still fails.
* `contact_blacksky.png`: adding "Pitch black sky, no atmosphere." to the prompt does not
  darken the sky, and on one seed it fills the sky with terrain again. The option remains
  as `--prompt-suffix` in `scripts/generate.py`.

Viewpoint matters most: Apollo-like framing carries the DEM geometry into the image. Sky
colour moved only with ControlNet training, and full-scale training is expected but not yet
shown to make it black.
