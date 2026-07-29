# SoulX-FlashHead — AI Agent Automated Deployment Guide

> This document is intended for **AI Agents** (OpenCode / Claude / Copilot, etc.) and automation scripts, with the goal of **unattended deployment and verification of this project**.
> Agent execution requirements: follow steps in strict order; each step has a "verification" checkpoint; if verification fails, consult [Troubleshooting](#-troubleshooting) before retrying — do not blindly retry.

---

## 0. Project Overview

- **What it is**: SoulX-FlashHead real-time streaming talking-head lip-sync service. FastAPI + WebSocket, listening on `ws://0.0.0.0:8765/ws/stream`.
- **Input**: One reference image + 16kHz audio stream (base64, float32/int16/wav).
- **Output**: 24 frames of 512×512 video per chunk (JPEG batch binary packets; WebP RGBA when matting is enabled).
- **Entry point**: `streaming_server.py` (dev/personal use), `start_server.bat` (Windows double-click launch, using conda env `flashhead`).
- **Models**: `SoulX-FlashHead-1_3B` (Lite variant achieves real-time on single RTX 4090) + `wav2vec2-base-960h` (audio encoder).
- **Optional**: Matting model (transparent background; MODNet 6.5M lightweight model by default, auto-downloaded on first use; switchable to RVM via env var).

### Hardware / System Requirements

| Item | Requirement |
|---|---|
| GPU | NVIDIA, Lite variant recommends ≥ 12GB VRAM (RTX 4090-class supports 3 concurrent streams) |
| CUDA | 12.8 (corresponding to torch 2.7.1+cu128) |
| Python | 3.10 (managed via conda, env name fixed as `flashhead`) |
| OS | Windows 10/11 or Linux |
| Disk | Models ~10GB+, reserve 20GB |

---

## 1. Environment Setup

### 1.1 Create conda environment

```bash
conda create -n flashhead python=3.10 -y
conda activate flashhead
```

> On Windows, `start_server.bat` hardcodes the path `D:\anaconda\envs\flashhead\python.exe`.
> If your Anaconda installation path differs, update the `PYTHON` variable on line 14 of `start_server.bat` accordingly.

### 1.2 Install PyTorch (CUDA 12.8)

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.7.1+cu128 True
```

### 1.3 Install other dependencies

```bash
pip install -r requirements.txt
```

### 1.4 Install FlashAttention

```bash
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation
```

> Building from source on Windows is highly prone to failure. **Strongly recommended to install the prebuilt wheel**:
> Download the `.whl` matching `cu128 + torch2.7 + python3.10` from
> https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2,
> then `pip install <downloaded.whl>`.

### 1.5 SageAttention (optional, for acceleration)

```bash
pip install sageattention==2.2.0 --no-build-isolation
```

### 1.6 FFmpeg

```bash
# Windows (conda)
conda install -c conda-forge ffmpeg==7 -y
# Ubuntu/Debian
apt-get install ffmpeg
```

---

## 2. Download Model Weights

```bash
# For users in mainland China, set the mirror first:
#   Windows PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"
#   Linux/bash:         export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub[cli]"
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h
```

Verify directory structure:

```
models/
├── SoulX-FlashHead-1_3B/
│   ├── Model_Lite/      # config.json + weights
│   └── Model_Pro/
└── wav2vec2-base-960h/
```

---

## 3. Launch & Verification

### 3.1 Start the server

```bash
# Option 1 (recommended for Windows): double-click or run
start_server.bat

# Option 2: manual
python streaming_server.py
```

### 3.2 Health check

```bash
curl http://127.0.0.1:8765/health
# Expected: {"status":"ok", ...}
```

Open `http://127.0.0.1:8765/` in a browser to see the built-in test page. Upload a reference image + audio for end-to-end verification.

### 3.3 Environment variables (set as needed)

| Variable | Default | Description |
|---|---|---|
| `FLASHHEAD_CKPT_DIR` | `models/SoulX-FlashHead-1_3B` | Model directory |
| `FLASHHEAD_WAV2VEC_DIR` | `models/wav2vec2-base-960h` | wav2vec directory |
| `FLASHHEAD_MODEL_TYPE` | `lite` | `lite` / `pro` |
| `FLASHHEAD_MAX_SESSIONS` | `2` | Max concurrent sessions |
| `FLASHHEAD_SEED` | `42` | Random seed |
| `FLASHHEAD_MATTING` | `0` | `1` to force matting for all sessions (defaults to MODNet) |
| `FLASHHEAD_MATTING_MODEL` | `modnet` | `modnet` / `rvm` |
| `FLASHHEAD_MATTING_RVM_DS` | `0.5` | RVM downsample ratio (lower = faster) |
| `FLASHHEAD_MATTING_RVM_KI` | `5` | RVM keyframe interval (higher = faster) |
| `FLASHHEAD_SUB_BATCH` | `6` | Frames per sub-batch (higher = faster encoding, slightly slower first frame) |
| `FLASHHEAD_WEBP_QUALITY` | `65` | WebP encoding quality (lower = faster) |

### 3.4 Pre-download matting model (optional but recommended)

Transparent background (`transparent_bg`) requires downloading the matting model on first use:

- **MODNet** (default, recommended): ~25 MB, downloaded from HuggingFace (`DavG25/modnet-pretrained-models`).
  Downloaded automatically on first server start, or pre-download manually:
  ```bash
  python -c "from flash_head.utils.matting import MODNetMatting; MODNetMatting(device='cuda'); print('MODNet ready')"
  ```
  Requires internet access to HuggingFace; configure `HTTPS_PROXY` if behind a firewall.

- **RVM** (fallback): ~14.5 MB, downloaded from GitHub (`PeterL1n/RobustVideoMatting`).
  Switch via `FLASHHEAD_MATTING_MODEL=rvm`, or pre-download:
  ```bash
  python -c "import torch; torch.hub.load('PeterL1n/RobustVideoMatting', 'mobilenetv3', trust_repo=True); print('RVM ready')"
  ```

- Cache location: `C:\Users\<user>\.cache\torch\hub\checkpoints\` (Linux: `~/.cache/torch/hub/checkpoints/`).
- If MODNet fails to load, it auto-falls-back to RVM; if RVM also fails, falls back to JPEG (no transparent background).
- Manual alternative (RVM): download `https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth` into `<hub>\checkpoints\`.

---

## 4. WebSocket Protocol Overview (for integrators)

Endpoint: `ws://<host>:8765/ws/stream`. Control messages are JSON text frames; video frames are binary frames.

### 4.1 Client → Server

**init** (first message after connecting):

```json
{
  "type": "init",
  "cond_image": "<base64 image bytes or server-side local path/directory>",
  "cond_is_path": false,
  "base_seed": 42,
  "use_face_crop": false,
  "transparent_bg": false
}
```

- When `cond_is_path: true`, `cond_image` is a **server-side local path**; passing a **directory** loads all `*.png` files as multiple persons (person_name = filename without extension).
- `transparent_bg: true` enables MODNet/RVM matting, outputting WebP RGBA frames.

**audio_chunk**: `{"type":"audio_chunk","audio":"<base64>","audio_format":"float32"}` (float32/int16/wav, 16kHz mono)

**flush**: Pad and emit frames when buffer is below one chunk (session kept alive).
**clear**: Discard buffered audio.
**reset**: `{"type":"reset","person_name":"<image name in directory mode>"}` hot-switch person.
**finish**: Flush remaining audio and end session.

### 4.2 Server → Client

- `ready`: `{type, session_id, frame_num:33, motion_frames_num:9, slice_len:24, chunk_audio_samples:15360, tgt_fps:25, sample_rate:16000, model_load_time_s}` — **width/height not included**, those are in `frames_meta`.
- `frames_meta`: `{type, chunk_idx, frames_count, height, width, processing_time_ms, fmt}` — `fmt` is `jpeg` or `webp` (when matting is active).
  Immediately followed by one **binary frame** message, batch format: `[4B count LE][4B len LE][encoded frame]...`
- `flushed` / `cleared` / `reset_ok` / `finished` (summary stats) / `error`.

---

## 5. Integration with super-agent-party

1. Complete deployment via chapters 1-3 and start `streaming_server.py`.
2. super-agent-party main UI → Settings → **SoulX Desktop Pet Bot**:
   - Default service address: `ws://127.0.0.1:8765`;
   - Upload one or more reference images (stored in party user data directory `uploaded_files/soulx_images/`);
   - Select the persona to use from the dropdown;
   - The "Transparent Background" toggle corresponds to `transparent_bg` (requires completing 3.4).
3. Click the start button to open the SoulX desktop pet window.
   - **Local service**: party uses directory-mode init, auto-`reset`s to the selected persona after ready; **remote service**: auto-falls-back to base64 single-image mode.
   - Window control panel includes an Anime4K ×2 upscaling toggle (client-side WebGPU, no server-side support needed).

---

## 6. ✅ Deployment Verification Checklist (Agent must check)

- [ ] `python -c "import torch; print(torch.cuda.is_available())"` outputs `True`
- [ ] `python -c "import flash_attn"` runs without error
- [ ] `models/SoulX-FlashHead-1_3B` and `models/wav2vec2-base-960h` exist and are non-empty
- [ ] `curl http://127.0.0.1:8765/health` returns ok
- [ ] (If matting is enabled) `~/.cache/torch/hub/checkpoints/modnet_mobilenetv2.pth` or `rvm_mobilenetv3.pth` exists
- [ ] Built-in test page (`http://127.0.0.1:8765/`) can generate video

## 🔧 Troubleshooting

| Symptom | Cause / Resolution |
|---|---|
| `torch.cuda.is_available()` returns False | torch version mismatch with CUDA; reinstall cu128 version |
| flash_attn build fails (Windows) | Use the official release prebuilt wheel (see 1.4) |
| Server starts with OOM / session gets evicted | Lower `FLASHHEAD_MAX_SESSIONS`; ensure no other processes are consuming VRAM |
| First-time matting hangs for a long time | Matting model is downloading (MODNet ~25MB from HuggingFace; RVM ~14.5MB from GitHub), see 3.4 for pre-download |
| Matting has no effect, output is still JPEG | Check server log for `Matting model loaded`; if MODNet fails it auto-falls-back to RVM, JPEG only if both fail. Force with `FLASHHEAD_MATTING=1` |
| MODNet download fails | Confirm `HTTPS_PROXY` proxy is configured (e.g., `http://127.0.0.1:7892`); or switch to RVM via `FLASHHEAD_MATTING_MODEL=rvm` |
| Directory-mode person name doesn't work | person_name must match the PNG filename (without extension); avoid `.` and spaces in filenames |
| HuggingFace download is slow | Set `HF_ENDPOINT=https://hf-mirror.com` |
| Client upscaling button shows unavailable | Upscaling runs on client WebGPU; requires Chrome/Edge/Electron with WebGPU support; unrelated to server |
