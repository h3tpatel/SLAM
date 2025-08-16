#!/usr/bin/env bash
set -e

ENV_DIR=${1:-mast3r-slam}
CUDA_VERSION=${2:-cu118}

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed. Visit https://astral.sh/uv to install."
  exit 1
fi

uv venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
uv pip install -e thirdparty/mast3r
uv pip install -e thirdparty/in3d
uv pip install --no-build-isolation -e .

# Uncomment for faster mp4 loading
# uv pip install torchcodec==0.1

mkdir -p checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
