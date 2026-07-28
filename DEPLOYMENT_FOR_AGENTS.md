# SoulX-FlashHead — AI Agent 自动部署指南

> 本文档面向 **AI Agent**（OpenCode / Claude / Copilot 等）与自动化脚本，目标是**无人值守地完成本项目的部署与验证**。
> Agent 执行要求：严格按顺序执行；每一步都有"验证"环节，验证失败时先查[故障排查](#-故障排查)再继续，不要盲目重试。

---

## 0. 项目概况

- **是什么**：SoulX-FlashHead 实时流式数字人口型同步服务。FastAPI + WebSocket，监听 `ws://0.0.0.0:8765/ws/stream`。
- **输入**：一张人物参考图 + 16kHz 音频流（base64，float32/int16/wav）。
- **输出**：每块（chunk）24 帧 512×512 视频帧（JPEG 批量二进制包；启用抠图时为 WebP RGBA）。
- **入口**：`streaming_server.py`（开发/自用）、`start_server.bat`（Windows 双击启动，使用 conda 环境 `flashhead`）。
- **模型**：`SoulX-FlashHead-1_3B`（Lite 版单卡 RTX4090 可实时）+ `wav2vec2-base-960h`（音频编码器）。
- **可选**：RVM 抠图模型（透明背景，首次启用时自动下载，可提前预下载，见 3.4）。

### 硬件/系统要求

| 项 | 要求 |
|---|---|
| GPU | NVIDIA，Lite 版建议 ≥ 12GB VRAM（RTX 4090 级可 3 路并发） |
| CUDA | 12.8（对应 torch 2.7.1+cu128） |
| Python | 3.10（用 conda 管理，环境名固定为 `flashhead`） |
| OS | Windows 10/11 或 Linux |
| 磁盘 | 模型约 10GB+，预留 20GB |

---

## 1. 环境准备

### 1.1 创建 conda 环境

```bash
conda create -n flashhead python=3.10 -y
conda activate flashhead
```

> Windows 下 `start_server.bat` 写死路径 `D:\anaconda\envs\flashhead\python.exe`。
> 若你的 Anaconda 安装位置不同，请同步修改 `start_server.bat` 第 14 行的 `PYTHON` 变量。

### 1.2 安装 PyTorch (CUDA 12.8)

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.7.1+cu128 True
```

### 1.3 安装其他依赖

```bash
pip install -r requirements.txt
```

### 1.4 安装 FlashAttention

```bash
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation
```

> Windows 下源码编译极易失败。**推荐直接装预编译 wheel**：从
> https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2
> 下载与 `cu128 + torch2.7 + python3.10` 匹配的 `.whl`，然后 `pip install <下载的.whl>`。

### 1.5 SageAttention（可选，加速）

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

## 2. 下载模型权重

```bash
# 中国大陆先设置镜像：
#   Windows PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"
#   Linux/bash:         export HF_ENDPOINT=https://hf-mirror.com
pip install "huggingface_hub[cli]"
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h
```

验证目录结构：

```
models/
├── SoulX-FlashHead-1_3B/
│   ├── Model_Lite/      # config.json + 权重
│   └── Model_Pro/
└── wav2vec2-base-960h/
```

---

## 3. 启动与验证

### 3.1 启动服务

```bash
# 方式一（推荐 Windows）：双击或执行
start_server.bat

# 方式二：手动
python streaming_server.py
```

### 3.2 健康检查

```bash
curl http://127.0.0.1:8765/health
# 期望: {"status":"ok", ...}
```

浏览器打开 `http://127.0.0.1:8765/` 可见内置测试页，上传参考图+音频即可端到端验证。

### 3.3 环境变量（按需设置）

| 变量 | 默认 | 说明 |
|---|---|---|
| `FLASHHEAD_CKPT_DIR` | `models/SoulX-FlashHead-1_3B` | 模型目录 |
| `FLASHHEAD_WAV2VEC_DIR` | `models/wav2vec2-base-960h` | wav2vec 目录 |
| `FLASHHEAD_MODEL_TYPE` | `lite` | `lite` / `pro` |
| `FLASHHEAD_MAX_SESSIONS` | `2` | 最大并发会话 |
| `FLASHHEAD_SEED` | `42` | 随机种子 |
| `FLASHHEAD_MATTING` | `0` | `1` 时对所有会话强制开启 RVM 抠图 |

### 3.4 预下载 RVM 抠图模型（可选但推荐）

透明背景（`transparent_bg`）首次启用时会通过 torch.hub 自动下载 RVM 代码与权重（约 14.5MB，需访问 GitHub）。
建议部署时**提前预下载**，避免首次使用时等待：

```bash
# conda 环境已激活时：
python -c "import torch; torch.hub.load('PeterL1n/RobustVideoMatting', 'mobilenetv3', trust_repo=True); print('RVM ready')"

# Windows 未激活环境时可直接调用解释器（按实际 conda 路径调整）：
D:\anaconda\envs\flashhead\python.exe -c "import torch; torch.hub.load('PeterL1n/RobustVideoMatting', 'mobilenetv3', trust_repo=True); print('RVM ready')"
```

- 下载位置：`C:\Users\<用户>\.cache\torch\hub\`（Linux 为 `~/.cache/torch/hub/`）。代码与权重都缓存在此，之后完全离线可用。
- 该下载发生在**本项目的 Python 环境/当前 OS 用户的全局 torch 缓存**中，不会改动任何其他项目（例如 super-agent-party）的文件或依赖。
- GitHub 较慢时可先配置代理，或多试几次（checkpoint 不支持断点续传，失败会重新下载，属正常现象）。
- 手动备选：下载 `https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth` 放入 `<hub>\checkpoints\`，再执行上面的命令让它完成 repo 检出。

---

## 4. WebSocket 协议速览（供集成方使用）

端点：`ws://<host>:8765/ws/stream`。控制消息为 JSON 文本帧；视频帧为二进制帧。

### 4.1 客户端 → 服务端

**init**（连接后首条消息）：

```json
{
  "type": "init",
  "cond_image": "<base64 图像字节 或 服务器本地路径/目录>",
  "cond_is_path": false,
  "base_seed": 42,
  "use_face_crop": false,
  "transparent_bg": false
}
```

- `cond_is_path: true` 时 `cond_image` 是**服务器本地路径**；传**目录**则加载目录下全部 `*.png` 作为多个人物（person_name = 文件名去扩展名）。
- `transparent_bg: true` 启用 RVM 抠图，输出 WebP RGBA 帧。

**audio_chunk**：`{"type":"audio_chunk","audio":"<base64>","audio_format":"float32"}`（float32/int16/wav，16kHz 单声道）

**flush**：缓冲不足一块时补齐出帧（会话保持）。
**clear**：丢弃缓冲音频。
**reset**：`{"type":"reset","person_name":"<目录模式下的图片名>"}` 热切换人物。
**finish**：冲洗剩余音频并结束会话。

### 4.2 服务端 → 客户端

- `ready`：`{type, session_id, frame_num:33, motion_frames_num:9, slice_len:24, chunk_audio_samples:15360, tgt_fps:25, sample_rate:16000, model_load_time_s}` — **不含宽高**，宽高在 `frames_meta` 里。
- `frames_meta`：`{type, chunk_idx, frames_count, height, width, processing_time_ms, fmt}` — `fmt` 为 `jpeg` 或 `webp`（抠图启用时）。
  紧随其后一条**二进制帧**，批量格式：`[4B count LE][4B len LE][encoded frame]...`
- `flushed` / `cleared` / `reset_ok` / `finished`（汇总统计）/ `error`。

---

## 5. 与 super-agent-party 集成

1. 先按第 1-3 章完成部署并启动 `streaming_server.py`。
2. super-agent-party 主界面 → 设置 → **SoulX 桌宠机器人**：
   - 服务地址默认 `ws://127.0.0.1:8765`；
   - 上传一张或多张参考图（存于 party 用户数据目录 `uploaded_files/soulx_images/`）；
   - 下拉选择要使用的形象；
   - "透明背景"开关对应 `transparent_bg`（需已完成 3.4）。
3. 点击启动按钮打开 SoulX 桌宠窗口。
   - **本机服务**时 party 使用目录模式 init，ready 后自动 `reset` 到选中形象；**远程服务**自动回退为 base64 单图模式。
   - 窗口控制面板含 Anime4K ×2 超分开关（客户端 WebGPU，无需服务端配合）。

---

## 6. ✅ 部署验证清单（Agent 必查）

- [ ] `python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`
- [ ] `python -c "import flash_attn"` 无异常
- [ ] `models/SoulX-FlashHead-1_3B` 与 `models/wav2vec2-base-960h` 存在且非空
- [ ] `curl http://127.0.0.1:8765/health` 返回 ok
- [ ] （启用抠图时）`~/.cache/torch/hub/checkpoints/rvm_mobilenetv3.pth` 存在
- [ ] 内置测试页 (`http://127.0.0.1:8765/`) 可生成视频

## 🔧 故障排查

| 症状 | 原因/处理 |
|---|---|
| `torch.cuda.is_available()` 为 False | torch 版本与 CUDA 不匹配，重装 cu128 版本 |
| flash_attn 编译失败（Windows） | 改用官方 release 的预编译 wheel（见 1.4） |
| 启动报 OOM / 会话被回收 | 降低 `FLASHHEAD_MAX_SESSIONS`；确认无其他进程占显存 |
| 首次抠图卡住很久 | RVM 正在从 GitHub 下载（约 14.5MB），见 3.4 预下载 |
| 抠图无效输出仍为 JPEG | 查看服务端日志 `RVM matting unavailable`；确认权重下载成功；也可用 `FLASHHEAD_MATTING=1` 强制开启 |
| 目录模式人物名不生效 | person_name 必须等于 PNG 文件名（不含扩展名），文件名不要含 `.` 与空格 |
| huggingface 下载慢 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| 客户端超分按钮提示不可用 | 超分在客户端 WebGPU 执行，需 Chrome/Edge/Electron 支持 WebGPU；与服务端无关 |
