# SoulX-FlashHead Streaming API

WebSocket-based real-time talking-head streaming service.

## Quick Start

```bash
# 1. Install dependencies (see main README)
conda create -n flashhead python=3.10
conda activate flashhead
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. Download models
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h

# 3. Start server
python streaming_server.py
# or double-click start_server.bat (Windows)
```

Server runs at `http://localhost:8765`. Open it in browser for the test UI.

## Architecture

```
Client (Browser / Python)           Server (FastAPI + WebSocket)
     │                                      │
     │── ws://host:8765/ws/stream ─────────►│  Session created
     │                                      │
     │── {type:"init", cond_image, ...} ───►│  Load model + prep params
     │◄─ {type:"ready", session_id, ...} ──│
     │                                      │
     │── {type:"audio_chunk", audio, ...} ─►│  Buffer → encode → generate
     │◄─ {type:"frames_meta", ...} ────────│  (JSON metadata)
     │◄─ [binary: raw uint8 frames] ──────│  (N×H×W×3 bytes)
     │                                      │
     │── {type:"finish"} ──────────────────►│  Flush buffer + summary
     │◄─ {type:"finished", ...} ───────────│
```

## WebSocket Protocol

**Endpoint:** `ws://host:8765/ws/stream`

All control messages are JSON. Frame data is binary.

### 1. Initialize

Send:
```json
{
  "type": "init",
  "cond_image": "<base64-encoded image> or \"examples/girl.png\"",
  "cond_is_path": false,
  "model_type": "lite",
  "base_seed": 42,
  "use_face_crop": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cond_image` | string | required | Base64-encoded JPEG/PNG, or local file path if `cond_is_path=true`. May also be a **directory** containing multiple `*.png` images (multi-person mode; person_name = filename without extension) |
| `cond_is_path` | bool | `false` | Whether `cond_image` is a server-side file path |
| `model_type` | string | `"lite"` | `"lite"` (RTX 4080+) or `"pro"` (2× RTX 5090) |
| `base_seed` | int | `42` | Random seed for deterministic output |
| `use_face_crop` | bool | `false` | Enable face detection and auto-crop |
| `transparent_bg` | bool | `false` | Enable server-side RVM matting; frames are then encoded as WebP with alpha (see `fmt` in `frames_meta`). Can also be forced via env `FLASHHEAD_MATTING=1` |

Receive:
```json
{
  "type": "ready",
  "session_id": "a1b2c3d4e5f6",
  "frame_num": 33,
  "motion_frames_num": 9,
  "slice_len": 24,
  "chunk_audio_samples": 15360,
  "tgt_fps": 25,
  "sample_rate": 16000,
  "model_load_time_s": 3.9
}
```

| Field | Description |
|-------|-------------|
| `slice_len` | New frames generated per audio chunk |
| `chunk_audio_samples` | Audio samples expected per chunk (float32 @ 16kHz) |
| `tgt_fps` | Output video frame rate |
| `motion_frames_num` | Overlap frames carried forward between chunks |

### 2. Send Audio Chunk

Send:
```json
{
  "type": "audio_chunk",
  "audio": "<base64-encoded float32 samples>",
  "audio_format": "float32"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `audio` | string | Base64-encoded audio data |
| `audio_format` | string | `"float32"` | `"int16"` | `"wav"` |

The server buffers incoming audio internally. A processing chunk is triggered automatically when `chunk_audio_samples` samples have accumulated. You can send arbitrary-sized chunks.

### 3. Receive Frame Results

For each processed chunk, two messages are sent back-to-back:

**First:** JSON metadata:
```json
{
  "type": "frames_meta",
  "chunk_idx": 0,
  "frames_count": 24,
  "height": 512,
  "width": 512,
  "processing_time_ms": 408.5,
  "fmt": "jpeg"
}
```

| Field | Description |
|-------|-------------|
| `fmt` | Frame encoding of the following binary payload: `jpeg` (opaque, default) or `webp` (RGBA with alpha, when `transparent_bg` is active) |

**Then:** Binary WebSocket frame — image batch payload:

```
[4 bytes: frame count N (uint32 LE)]
[4 bytes: frame_1 length (uint32 LE)][frame_1 bytes]
[4 bytes: frame_2 length (uint32 LE)][frame_2 bytes]
...
```

Each entry is one video frame (`height`×`width` from the metadata), encoded per `fmt`. Clients should decode with `createImageBitmap` (or equivalent) which keeps decoding off the main thread.

### 4. Finish

Send:
```json
{"type": "finish"}
```

The server flushes any remaining buffered audio, sends the final chunk frames, then responds with:

```json
{
  "type": "finished",
  "total_frames": 1656,
  "total_time_s": 28.3,
  "avg_fps": 58.6,
  "steady_state_fps": 58.8,
  "first_chunk_ms": 498.0,
  "avg_chunk_ms": 408.2,
  "num_chunks": 69
}
```

**Note:** After `finish`, the server destroys the session and closes the connection. Use `flush` instead if you want to keep the session alive for subsequent audio.

### 4b. Flush (keep-alive)

Send:
```json
{"type": "flush"}
```

Flushes any remaining buffered audio (pads with silence to a full chunk), generates the final frames for the current speech segment, but **keeps the session alive**. Ideal for persistent talking-head windows that handle multiple separate speech segments over one connection.

Response:
```json
{"type": "flushed"}
```

### 4c. Clear (discard buffer)

Send:
```json
{"type": "clear"}
```

Discards any buffered (unprocessed) audio without generating frames, keeping the session alive. Use when speech is interrupted (e.g., user barge-in) and the buffered tail audio should not be rendered.

Response:
```json
{"type": "cleared"}
```

### 5. Reset Person (multi-person)

```json
{"type": "reset", "person_name": "alice"}
```
Response:
```json
{"type": "reset_ok", "person": "alice"}
```

Multi-person mode: pass a **directory** as `cond_image` (with `cond_is_path: true`) at init — all `*.png` files inside are loaded as separate persons, and `person_name` is the PNG filename without extension. `reset` hot-switches the active person without re-initializing the model. If `person_name` does not exist, the current person is kept.

### 6. Error

```json
{"type": "error", "message": "..."}
```

## REST Endpoints

### GET /health

```json
{"status":"ok","sessions":0,"gpu":"NVIDIA GeForce RTX 4080","vram_gb":5.05}
```

### GET /

Web-based test UI (Vue 3 SPA). Upload image + WAV audio, view real-time streaming output.

## Python Client Example

```python
import asyncio, base64, json, numpy as np, websockets

async def main():
    # Load audio
    audio = np.random.randn(16000 * 5).astype(np.float32)  # 5s of noise

    async with websockets.connect("ws://localhost:8765/ws/stream", max_size=50*1024*1024) as ws:
        # Init
        await ws.send(json.dumps({
            "type": "init",
            "cond_image": "examples/girl.png",
            "cond_is_path": True,
            "model_type": "lite"
        }))
        resp = json.loads(await ws.recv())
        assert resp["type"] == "ready"
        chunk_samples = resp["chunk_audio_samples"]

        # Stream audio
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i+chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            b64 = base64.b64encode(chunk.tobytes()).decode()
            await ws.send(json.dumps({"type": "audio_chunk", "audio": b64, "audio_format": "float32"}))

            # Receive meta + binary frames
            meta = json.loads(await ws.recv())
            frames = await ws.recv()  # raw bytes
            print(f"Chunk {meta['chunk_idx']}: {meta['frames_count']} frames, {meta['processing_time_ms']}ms")

        # Finish
        await ws.send(json.dumps({"type": "finish"}))
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "finished":
                print(f"Done. FPS: {msg['avg_fps']}")
                break

asyncio.run(main())
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASHHEAD_CKPT_DIR` | `models/SoulX-FlashHead-1_3B` | Model checkpoint path |
| `FLASHHEAD_WAV2VEC_DIR` | `models/wav2vec2-base-960h` | Wav2Vec2 model path |
| `FLASHHEAD_MODEL_TYPE` | `lite` | `lite` or `pro` |
| `FLASHHEAD_MAX_SESSIONS` | `2` | Max concurrent WebSocket sessions |
| `FLASHHEAD_SEED` | `42` | Default random seed |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID |

## Benchmark Script

```bash
python benchmark_lite.py
```
Runs offline inference with 66s sample audio and prints detailed metrics: FPS, VRAM, first-frame latency.

## Performance (RTX 4080, 512×512, Lite model)

| Metric | Value |
|--------|-------|
| Model load time | ~3.8s |
| Peak VRAM | ~5.0 GB |
| First-frame latency | ~500ms |
| Steady-state FPS | ~58.8 (2.3× real-time) |
| Chunk processing time | ~408ms (24 frames/chunk) |
