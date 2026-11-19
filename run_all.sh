#!/usr/bin/env bash
# DESPINA end-to-end (multi-GPU server, no scheduler). Every phase is resumable:
# rerun the same command after an interruption and finished work is skipped.
#
#   bash run_all.sh                          # all phases
#   PHASES="render gen eval" bash run_all.sh
#
# Phases: data (download and process the Apollo frames), prep (DPT depth, PiDiNet soft edges,
# prompts, parquet), train (LoRA, depth ControlNet, soft-edge ControlNet), render (DEM
# constraints for the sites), gen (images), eval (metrics and tables).
#
# Inputs (override with env vars):
#   IMAGES   training images (default: the frames that pass the quality check in the data phase)
#   DROOT    folder with dataset.csv and prompts/ (DESPINA_dataset)
#   SITES    sites yaml          EXP  experiments yaml
#   TRAIN_GPUS / ALL_GPUS   GPU ids by nvidia-smi numbering
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # match nvidia-smi numbering
export HF_HOME=${HF_HOME:-$PWD/hf_cache}
IMAGES=${IMAGES:-data/apollo_train_images}
DROOT=${DROOT:-DESPINA_dataset}
SITES=${SITES:-configs/sites.yaml}
EXP=${EXP:-configs/experiments.yaml}
RUN=${RUN:-runs/main}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,5,6}     # six P40s (ControlNet training needs an even count)
ALL_GPUS=${ALL_GPUS:-0,1,2,3,5,6,7}
LORA_STEPS=${LORA_STEPS:-3000}
CN_STEPS=${CN_STEPS:-3000}
PHASES=${PHASES:-"data prep train render gen eval"}
mkdir -p "$RUN"

has() { [[ " $PHASES " == *" $1 "* ]]; }

if has data; then
  python tools/download_apollo.py --names "$DROOT/dataset.csv" --out data/apollo_raw --size med --workers 12
  python scripts/process_apollo.py --raw data/apollo_raw --out data/apollo_processed --workers 24 \
    --link-dir data/apollo_train_images
fi

if has prep; then
  N=$(echo "$ALL_GPUS" | tr ',' '\n' | wc -l)
  : > "$RUN/jobs_prep.txt"
  for i in $(seq 0 $((N-1))); do
    echo "1 python scripts/prep_training_data.py --images $IMAGES --dataset-root $DROOT --out $RUN/train_set --device cuda:0 --recompute-depth --shard $i --num-shards $N" >> "$RUN/jobs_prep.txt"
  done
  python tools/gpuq.py "$RUN/jobs_prep.txt" --state "$RUN/queue" --gpus "$ALL_GPUS"
  python scripts/build_parquet.py --prep "$RUN/train_set"
fi

if has train; then
  [ -d "$RUN/lora/pytorch_lora_weights.safetensors" ] || [ -f "$RUN/lora/pytorch_lora_weights.safetensors" ] || \
    GPUS=$TRAIN_GPUS DATA=$RUN/train_set OUT=$RUN/lora STEPS=$LORA_STEPS bash scripts/train_lora.sh
  for K in depth softedge; do
    [ -f "$RUN/cn_$K/diffusion_pytorch_model.safetensors" ] || \
      KIND=$K GPUS=$TRAIN_GPUS DATA=$RUN/train_set OUT=$RUN/cn_$K STEPS=$CN_STEPS bash scripts/train_controlnet.sh
  done
fi

if has render; then
  python scripts/render_sites.py --sites "$SITES" --out "$RUN/sites" --device cuda:${ALL_GPUS%%,*}
fi

if has gen; then
  LORA_ARG=""; [ -f "$RUN/lora/pytorch_lora_weights.safetensors" ] && LORA_ARG="--lora $RUN/lora"
  CN_ARG=""
  [ -f "$RUN/cn_depth/diffusion_pytorch_model.safetensors" ] && CN_ARG="$CN_ARG --cn-depth $RUN/cn_depth"
  [ -f "$RUN/cn_softedge/diffusion_pytorch_model.safetensors" ] && CN_ARG="$CN_ARG --cn-softedge $RUN/cn_softedge"
  : > "$RUN/jobs_gen.txt"
  for C in $(python -c "import yaml;print(' '.join(c['name'] for c in yaml.safe_load(open('$EXP'))['configs']))"); do
    echo "2 python scripts/generate.py --sites-dir $RUN/sites --experiments $EXP --out $RUN/gen --devices cuda:0,cuda:1 --configs $C $LORA_ARG $CN_ARG" >> "$RUN/jobs_gen.txt"
  done
  python tools/gpuq.py "$RUN/jobs_gen.txt" --state "$RUN/queue" --gpus "$ALL_GPUS"
fi

if has eval; then
  python scripts/evaluate.py --sites-dir "$RUN/sites" --gen "$RUN/gen" --out "$RUN/eval" --device cuda:${ALL_GPUS%%,*}
fi
echo "run_all finished: $PHASES"
