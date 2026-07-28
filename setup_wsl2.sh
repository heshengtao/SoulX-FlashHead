#!/bin/bash
# SoulX-FlashHead WSL2 / Linux one-click setup
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[*]${NC} $*"; }
ok()   { echo -e "${GREEN}[+]${NC} $*"; }
err()  { echo -e "${RED}[!]${NC} $*"; exit 1; }

ENV_NAME="flashhead"
PYTHON_VER="3.10"
PORT="${FLASHHEAD_PORT:-8765}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  SoulX-FlashHead WSL2 Installer"
echo "========================================"
echo ""

# ---- 0. Check system ----
log "Checking system..."
if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi not found. Is this WSL2 with NVIDIA driver?"
fi
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
ok "GPU OK, VRAM: ${VRAM} MiB"

# ---- 1. Conda env ----
log "Step 1/6: Conda environment"
if conda info --envs 2>/dev/null | grep -q "^${ENV_NAME} "; then
    ok "Conda env '${ENV_NAME}' already exists, skipping."
else
    conda create -n "${ENV_NAME}" python="${PYTHON_VER}" -y
    ok "Conda env '${ENV_NAME}' created."
fi

# ---- 2. PyTorch ----
log "Step 2/6: PyTorch with CUDA"
CONDA_PYTHON="$(conda run -n "${ENV_NAME}" which python)"
if conda run -n "${ENV_NAME}" python -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "2.7.1"; then
    ok "PyTorch 2.7.1 already installed, skipping."
else
    conda run -n "${ENV_NAME}" pip install torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128
    ok "PyTorch installed."
fi

# ---- 3. Requirements ----
log "Step 3/6: Python dependencies"
# Remove nccl line (not needed for single-GPU)
TMP_REQ="/tmp/flashhead_req_$$.txt"
sed '/nvidia-nccl-cu12/d' requirements.txt > "$TMP_REQ"
conda run -n "${ENV_NAME}" pip install -r "$TMP_REQ" 2>&1 | tail -3
rm -f "$TMP_REQ"
ok "Dependencies installed."

# ---- 4. FlashAttention + Triton ----
log "Step 4/6: FlashAttention + Triton (compile backend)"
if conda run -n "${ENV_NAME}" python -c "import flash_attn" 2>/dev/null; then
    ok "FlashAttention already installed, skipping."
else
    FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    conda run -n "${ENV_NAME}" pip install "$FA_URL"
    ok "FlashAttention 2.8.0 installed (prebuilt wheel)."
fi

if conda run -n "${ENV_NAME}" python -c "import triton" 2>/dev/null; then
    ok "Triton already installed, skipping."
else
    conda run -n "${ENV_NAME}" pip install triton
    ok "Triton installed (required for torch.compile)."
fi

# ---- 5. FFmpeg ----
log "Step 5/6: FFmpeg"
if command -v ffmpeg &>/dev/null; then
    ok "FFmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    conda install -n "${ENV_NAME}" -c conda-forge ffmpeg -y 2>/dev/null || \
        { log "conda ffmpeg failed, trying apt..."; sudo apt-get install -y ffmpeg; }
    ok "FFmpeg installed."
fi

# ---- 6. Models ----
log "Step 6/6: Models"
mkdir -p models

MODEL_DIR="models/SoulX-FlashHead-1_3B"
if [ -f "${MODEL_DIR}/Model_Lite/diffusion_pytorch_model.safetensors" ]; then
    ok "Model SoulX-FlashHead-1_3B already downloaded."
else
    log "Downloading SoulX-FlashHead-1_3B (~8GB, this may take a while)..."
    conda run -n "${ENV_NAME}" huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
        --local-dir "${MODEL_DIR}"
    ok "Model downloaded."
fi

WAV2VEC_DIR="models/wav2vec2-base-960h"
if [ -f "${WAV2VEC_DIR}/model.safetensors" ] || [ -f "${WAV2VEC_DIR}/pytorch_model.bin" ]; then
    ok "Wav2Vec2 already downloaded."
else
    log "Downloading wav2vec2-base-960h..."
    conda run -n "${ENV_NAME}" huggingface-cli download facebook/wav2vec2-base-960h \
        --local-dir "${WAV2VEC_DIR}"
    ok "Wav2Vec2 downloaded."
fi

# ---- Done ----
echo ""
echo "========================================"
echo -e "  ${GREEN}Setup complete!${NC}"
echo "========================================"
echo ""
echo "  Run the server:"
echo "    conda activate ${ENV_NAME}"
echo "    python streaming_server.py"
echo ""
echo "  Or with a custom port:"
echo "    FLASHHEAD_PORT=8080 python streaming_server.py"
echo ""
echo "  Quick test (another terminal):"
echo "    conda activate ${ENV_NAME}"
echo "    python streaming_client.py --audio examples/podcast_sichuan_16k.wav"
echo ""
echo "  Open browser: http://localhost:${PORT}"
echo ""
