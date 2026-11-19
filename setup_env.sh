set -x
cd "$(dirname "$0")"
python3.11 -m venv venv
source venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -q -r requirements.txt
mkdir -p third_party && cd third_party
for f in examples/text_to_image/train_text_to_image_lora_sdxl.py examples/controlnet/train_controlnet_sdxl.py; do
  [ -f "$(basename $f)" ] || curl -sSfLO https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/$f
done
cd .. && python tools/patch_controlnet_split.py third_party/train_controlnet_sdxl.py third_party/train_controlnet_sdxl_split.py
python -c "import torch,diffusers;print(torch.__version__,diffusers.__version__,torch.cuda.is_available())"
echo SETUP_DONE
