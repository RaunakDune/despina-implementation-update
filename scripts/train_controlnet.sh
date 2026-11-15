#!/usr/bin/env bash
# Group I, item 9: fine-tune an SDXL ControlNet on Apollo photos + their conditions.
#   KIND=depth    starts from diffusers/controlnet-depth-sdxl-1.0, conditions = DPT depth
#   KIND=softedge starts from SargeZT/controlnet-sd-xl-1.0-softedge-dexined, conditions = PiDiNet
#   KIND=depth GPUS=0,1,2,3,5,6 DATA=data/train_set OUT=runs/cn_depth bash scripts/train_controlnet.sh
set -euo pipefail
cd "$(dirname "$0")/.."
KIND=${KIND:-depth}; GPUS=${GPUS:-0}; DATA=${DATA:-data/train_set}; OUT=${OUT:-runs/cn_$KIND}
STEPS=${STEPS:-3000}; RES=${RES:-1024}; LR=${LR:-1e-5}; BS=${BS:-1}; ACC=${ACC:-1}
case $KIND in
  depth) INIT=diffusers/controlnet-depth-sdxl-1.0 ;;
  softedge) INIT=SargeZT/controlnet-sd-xl-1.0-softedge-dexined ;;
  *) echo "KIND must be depth or softedge"; exit 1 ;;
esac
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
source venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # match nvidia-smi numbering
export HF_HOME=${HF_HOME:-$PWD/hf_cache}
# SPLIT=1 (default): each process uses two GPUs, ControlNet + optimizer on one and the frozen
# UNet on the other (tools/patch_controlnet_split.py), so GPUS must list an even number of GPUs.
SPLIT=${SPLIT:-1}
if [ "$SPLIT" = 1 ]; then
  [ $((N % 2)) -eq 0 ] || { echo "SPLIT=1 needs an even number of GPUs"; exit 1; }
  NP=$((N / 2))
  [ -f third_party/train_controlnet_sdxl_split.py ] || \
    python tools/patch_controlnet_split.py third_party/train_controlnet_sdxl.py third_party/train_controlnet_sdxl_split.py
  export CUDA_VISIBLE_DEVICES=$GPUS DESPINA_UNET_DEVICE_OFFSET=$NP
  SCRIPT=third_party/train_controlnet_sdxl_split.py; GPU_ARG=""
else
  NP=$N; SCRIPT=third_party/train_controlnet_sdxl.py; GPU_ARG="--gpu_ids $GPUS"
fi
MP=""; [ "$NP" -gt 1 ] && MP="--multi_gpu"
accelerate launch $MP --num_processes "$NP" $GPU_ARG --mixed_precision no ${ACCEL_EXTRA:-} \
  "$SCRIPT" \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --controlnet_model_name_or_path "$INIT" \
  --train_data_dir "$DATA/train" --image_column image --conditioning_image_column "$KIND" --caption_column text \
  --resolution "$RES" --train_batch_size "$BS" --gradient_accumulation_steps "$ACC" \
  --gradient_checkpointing --set_grads_to_none --max_train_steps "$STEPS" --learning_rate "$LR" \
  --lr_scheduler constant_with_warmup --lr_warmup_steps 100 --mixed_precision no \
  --dataloader_num_workers 4 --proportion_empty_prompts 0.1 \
  --checkpointing_steps 500 --checkpoints_total_limit 3 --resume_from_checkpoint latest \
  --use_8bit_adam --seed 42 --output_dir "$OUT" ${EXTRA:-}
