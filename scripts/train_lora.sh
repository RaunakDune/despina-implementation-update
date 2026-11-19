#!/usr/bin/env bash
# Group I, item 8: lunar LoRA on SDXL (diffusers v0.31.0 example script; tools/patch_lora_schedule.py
# fixes its LR schedule for multi-GPU runs).
# fp32 because Pascal GPUs have no fast fp16/bf16. Paper: batch 6 on 6 GPUs.
#   GPUS=0,1,2,3,4,5 DATA=data/train_set OUT=runs/lora STEPS=3000 bash scripts/train_lora.sh
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS=${GPUS:-0}; DATA=${DATA:-data/train_set}; OUT=${OUT:-runs/lora}
STEPS=${STEPS:-3000}; RES=${RES:-1024}; RANK=${RANK:-16}; LR=${LR:-1e-4}; BS=${BS:-1}; ACC=${ACC:-1}
# Offset noise lets SDXL learn large near-black regions (the lunar sky). Without it the
# fine-tuned model avoids or greys out the sky. Set NOISE_OFFSET=0 to disable.
NOISE_OFFSET=${NOISE_OFFSET:-0.1}
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MP=""; [ "$N" -gt 1 ] && MP="--multi_gpu"
source venv/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # match nvidia-smi numbering
export HF_HOME=${HF_HOME:-$PWD/hf_cache}
python tools/patch_lora_schedule.py third_party/train_text_to_image_lora_sdxl.py third_party/train_text_to_image_lora_sdxl_fixed.py
accelerate launch $MP --num_processes "$N" --gpu_ids "$GPUS" --mixed_precision no \
  third_party/train_text_to_image_lora_sdxl_fixed.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --dataset_name "$DATA/train" --image_column image --caption_column text \
  --resolution "$RES" --center_crop --random_flip \
  --train_batch_size "$BS" --gradient_accumulation_steps "$ACC" --gradient_checkpointing \
  --max_train_steps "$STEPS" --learning_rate "$LR" --lr_scheduler cosine --lr_warmup_steps 100 \
  --rank "$RANK" --noise_offset "$NOISE_OFFSET" --mixed_precision no --dataloader_num_workers 4 \
  --checkpointing_steps 500 --checkpoints_total_limit 3 --resume_from_checkpoint latest \
  --seed 42 --output_dir "$OUT" ${EXTRA:-}
