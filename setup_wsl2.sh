#!/bin/bash
# SoulX-FlashHead WSL2 / Linux one-click setup
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[0;33m'; NC='\033[0m'; BOLD='\033[1m'

log()    { echo -e "${CYAN}  [·]${NC} $*"; }
ok()     { echo -e "${GREEN}  [✓]${NC} $*"; }
warn()   { echo -e "${YELLOW}  [!]${NC} $*"; }
err()    { echo -e "${RED}  [✗]${NC} $*"; exit 1; }
step()   { echo -e "\n${BOLD}${CYAN}═══ $* ═══${NC}"; }
header() { echo -e "${BOLD}${GREEN}──▶ $*${NC}"; }

SECONDS=0
_step_start=0

step_begin() { _step_start=$SECONDS; echo ""; }
step_end()   { local d=$((SECONDS - _step_start)); echo -e "        ${GREEN}⏱ took ${d}s${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="flashhead"
PYTHON_VER="3.10"
export CONDA_PYTHON=""

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║  SoulX-FlashHead WSL2 / Linux Setup     ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ============================================================
# 0 — System checks
# ============================================================
step "Step 0/7 — System Check"
step_begin

if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi not found. Are you in WSL2 with NVIDIA driver installed?"
fi
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
echo -e "  ${BOLD}GPU:${NC}  ${GPU}"
echo -e "  ${BOLD}VRAM:${NC} ${VRAM} MiB"
ok "GPU check passed"

# Install conda if missing
if ! command -v conda &>/dev/null; then
    warn "Conda not found, installing Miniconda..."
    URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    TMP="/tmp/miniconda_$$.sh"
    log "Downloading Miniconda..."
    wget -q --show-progress -O "$TMP" "$URL" || curl -#L -o "$TMP" "$URL"
    log "Installing to ~/miniconda3..."
    bash "$TMP" -b -u -p "$HOME/miniconda3" >/dev/null 2>&1
    rm -f "$TMP"
    "$HOME/miniconda3/bin/conda" init bash >/dev/null 2>&1
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    ok "Miniconda installed. Restart shell and re-run this script."
    exit 0
fi
ok "Conda found: $(conda --version 2>/dev/null)"

eval "$(conda shell.bash hook)" 2>/dev/null || true
# Accept ToS for newer conda
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r   2>/dev/null || true
ok "Conda ready"

step_end

# ============================================================
# 1 — Conda environment
# ============================================================
step "Step 1/7 — Conda Environment (${ENV_NAME}, Python ${PYTHON_VER})"
step_begin

if conda info --envs 2>/dev/null | grep -q "^${ENV_NAME} "; then
    ok "Environment '${ENV_NAME}' already exists"
else
    log "Creating environment..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VER}" -y 2>&1 | while IFS= read -r line; do
        case "$line" in
            *"Downloading"*|*"Extracting"*) echo -e "        ${line}" ;;
        esac
    done
    ok "Environment created"
fi

CONDA_PYTHON="$(conda run -n "${ENV_NAME}" which python 2>/dev/null)"
echo -e "  ${BOLD}Python:${NC} ${CONDA_PYTHON}"
echo -e "  ${BOLD}Version:${NC} $(conda run -n "${ENV_NAME}" python --version 2>&1)"

# Helper to run pip in the env with live output
pip_run() {
    conda run -n "${ENV_NAME}" --no-capture-output python -m pip "$@" 2>&1 | while IFS= read -r line; do
        if [[ "$line" =~ Downloading|Installing|Successfully|ERROR|Collecting ]]; then
            echo -e "        ${line}"
        fi
    done
    return ${PIPESTATUS[0]}
}

pip_run_quiet() {
    conda run -n "${ENV_NAME}" --no-capture-output python -m pip "$@" 2>&1 | tail -5
    return ${PIPESTATUS[0]}
}

step_end

# ============================================================
# 2 — PyTorch
# ============================================================
step "Step 2/7 — PyTorch 2.7.1 + CUDA 12.8"
step_begin

TORCH_OK=$(conda run -n "${ENV_NAME}" python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [[ "$TORCH_OK" == "2.7.1"* ]]; then
    ok "PyTorch 2.7.1 already installed"
else
    header "Downloading PyTorch (~3.3 GB)"
    if pip_run install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128; then
        ok "PyTorch installed"
    else
        warn "Official index failed, trying tsinghua mirror..."
        pip_run install torch==2.7.1 torchvision==0.22.1 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu128
        ok "PyTorch installed (mirror)"
    fi
fi

# Verify CUDA
CUDA_OK=$(conda run -n "${ENV_NAME}" python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [[ "$CUDA_OK" == "True" ]]; then
    ok "CUDA available: $(conda run -n "${ENV_NAME}" python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)"
else
    err "CUDA NOT available! Check nvidia-container-toolkit / WSL2 setup."
fi

step_end

# ============================================================
# 3 — Python dependencies
# ============================================================
step "Step 3/7 — Python Dependencies (requirements.txt)"
step_begin

TMP_REQ="/tmp/flashhead_req_$$.txt"
sed '/nvidia-nccl-cu12/d' requirements.txt > "$TMP_REQ"

DEPS_COUNT=$(grep -vc '^\s*#' "$TMP_REQ" || echo "?")
log "Installing ~${DEPS_COUNT} packages..."

if pip_run_quiet install -r "$TMP_REQ"; then
    ok "Dependencies installed"
else
    warn "Some packages failed, retrying one by one..."
    while IFS= read -r pkg; do
        [[ -z "$pkg" || "$pkg" == \#* ]] && continue
        conda run -n "${ENV_NAME}" --no-capture-output python -m pip install "$pkg" 2>&1 | tail -1 || warn "  Failed: $pkg"
    done < "$TMP_REQ"
fi
rm -f "$TMP_REQ"

# Verify key packages
for mod in diffusers transformers xformers accelerate; do
    conda run -n "${ENV_NAME}" python -c "import ${mod}" 2>/dev/null && ok "  ${mod}" || warn "  ${mod} MISSING"
done

step_end

# ============================================================
# 4 — FlashAttention + Triton
# ============================================================
step "Step 4/7 — FlashAttention + Triton (compile)"
step_begin

# FlashAttention
if conda run -n "${ENV_NAME}" python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null; then
    ok "FlashAttention already installed"
else
    FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    log "Installing FlashAttention (prebuilt wheel)..."
    if pip_run_quiet install "$FA_URL"; then
        ok "FlashAttention 2.8.0 installed"
    else
        warn "Prebuilt wheel failed, trying build from source (slower)..."
        conda run -n "${ENV_NAME}" --no-capture-output pip install ninja
        conda run -n "${ENV_NAME}" --no-capture-output pip install flash_attn==2.8.0.post2 --no-build-isolation 2>&1 | tail -10
        ok "FlashAttention built from source"
    fi
fi

# Triton (for torch.compile)
if conda run -n "${ENV_NAME}" python -c "import triton; print(triton.__version__)" 2>/dev/null; then
    ok "Triton already installed"
else
    log "Installing Triton..."
    if pip_run_quiet install triton; then
        ok "Triton installed"
    else
        warn "Triton install failed (torch.compile will use fallback)"
    fi
fi

step_end

# ============================================================
# 5 — FFmpeg
# ============================================================
step "Step 5/7 — FFmpeg"
step_begin

if command -v ffmpeg &>/dev/null; then
    ok "FFmpeg: $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f3)"
else
    log "Installing FFmpeg via apt..."
    sudo apt-get update -qq 2>/dev/null
    sudo apt-get install -y -qq ffmpeg 2>&1 | tail -3
    ok "FFmpeg installed"
fi

step_end

# ============================================================
# 6 — Download models
# ============================================================
step "Step 6/7 — Download Models"
step_begin

mkdir -p models

# FlashHead model
MODEL_DIR="models/SoulX-FlashHead-1_3B"
LITE_FILE="${MODEL_DIR}/Model_Lite/diffusion_pytorch_model.safetensors"
if [ -f "$LITE_FILE" ]; then
    SIZE=$(du -sh "$LITE_FILE" 2>/dev/null | cut -f1)
    ok "SoulX-FlashHead-1_3B already downloaded (${SIZE})"
else
    header "Downloading SoulX-FlashHead-1_3B (~8 GB)"
    log "This may take 10-30 minutes depending on network..."
    conda run -n "${ENV_NAME}" --no-capture-output \
        huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
        --local-dir "${MODEL_DIR}" 2>&1 | while IFS= read -r line; do
        if [[ "$line" =~ Downloading|Fetching|Download\ complete ]]; then
            echo -e "        ${line}"
        fi
    done
    SIZE=$(du -sh "${MODEL_DIR}" 2>/dev/null | cut -f1)
    ok "Model downloaded (${SIZE})"
fi

# Wav2Vec2
WAV2VEC_DIR="models/wav2vec2-base-960h"
if [ -f "${WAV2VEC_DIR}/model.safetensors" ] || [ -f "${WAV2VEC_DIR}/pytorch_model.bin" ]; then
    ok "Wav2Vec2 already downloaded"
else
    header "Downloading wav2vec2-base-960h (~360 MB)"
    conda run -n "${ENV_NAME}" --no-capture-output \
        huggingface-cli download facebook/wav2vec2-base-960h \
        --local-dir "${WAV2VEC_DIR}" 2>&1 | tail -5
    ok "Wav2Vec2 downloaded"
fi

step_end

# ============================================================
# 7 — Verify
# ============================================================
step "Step 7/7 — Verification"
step_begin

echo ""
conda run -n "${ENV_NAME}" python -c "
import torch, flash_attn, xformers
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA:          {torch.version.cuda}')
print(f'  GPU:           {torch.cuda.get_device_name(0)}')
print(f'  FlashAttn:     {flash_attn.__version__}')
print(f'  xFormers:      {xformers.__version__}')
" 2>/dev/null || warn "Some imports failed — check logs above"

# Quick VRAM check (load a small tensor)
conda run -n "${ENV_NAME}" python -c "
import torch
t = torch.zeros(1, device='cuda')
print(f'  CUDA works:    True')
del t
" 2>/dev/null && ok "CUDA compute test passed"

step_end

# ============================================================
# Done
# ============================================================
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║         Setup Complete!  🎉              ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Total time: ${BOLD}${SECONDS}s${NC}"
echo ""
echo -e "  Start the server:"
echo -e "    ${CYAN}conda activate ${ENV_NAME}${NC}"
echo -e "    ${CYAN}python streaming_server.py${NC}"
echo ""
echo -e "  Quick benchmark:"
echo -e "    ${CYAN}conda activate ${ENV_NAME}${NC}"
echo -e "    ${CYAN}python benchmark_lite.py${NC}"
echo ""
echo -e "  Web UI:  ${BOLD}http://localhost:8765${NC}"
echo ""
