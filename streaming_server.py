"""
SoulX-FlashHead Streaming API Server
WebSocket-based streaming inference service.

Protocol:
  1. Connect: ws://host:port/ws/stream
  2. Send init (JSON) -> receives session config
  3. Send audio chunks (JSON/binary) -> receives video frames
  4. Send finish -> receives summary

Run:
  python streaming_server.py
  # or
  uvicorn streaming_server:app --host 0.0.0.0 --port 8765
"""

import asyncio
import base64
import io
import json
import os
import sys
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# Windows: disable torch.compile before importing flash_head (Triton not available on win32)
if sys.platform == "win32":
    import flash_head.src.pipeline.flash_head_pipeline as _pipe_cfg
    _pipe_cfg.COMPILE_MODEL = False
    _pipe_cfg.COMPILE_VAE = False

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CKPT_DIR = os.environ.get("FLASHHEAD_CKPT_DIR", "models/SoulX-FlashHead-1_3B")
WAV2VEC_DIR = os.environ.get("FLASHHEAD_WAV2VEC_DIR", "models/wav2vec2-base-960h")
DEFAULT_MODEL_TYPE = os.environ.get("FLASHHEAD_MODEL_TYPE", "lite")
MAX_SESSIONS = int(os.environ.get("FLASHHEAD_MAX_SESSIONS", "2"))
BASE_SEED = int(os.environ.get("FLASHHEAD_SEED", "42"))
MATTING_ENABLED = os.environ.get("FLASHHEAD_MATTING", "0") == "1"
SUB_BATCH = 3  # 每个子批次打包的帧数, 越小首帧到达越快

def _resolve_path(path: str) -> str:
    """将 Windows 路径转为 WSL 路径（当在 Linux 下运行时）。"""
    if sys.platform == "linux" and re.match(r'^[A-Z]:[\\/]', path):
        drive = path[0].lower()
        rest = path[2:].replace('\\', '/')
        return f'/mnt/{drive}/{rest}'
    return path

_MATTING = None

def get_matting():
    """进程级单例：RVM 抠图模型只在首次需要时加载"""
    global _MATTING
    if _MATTING is None:
        from flash_head.utils.matting import RVMMatting
        _MATTING = RVMMatting(device="cuda")
    return _MATTING

# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class Session:
    id: str
    pipeline: object = None
    infer_params: dict = None
    person_name: str = ""

    # Audio accumulator: buffers incoming chunks until we have enough for processing
    audio_buffer: deque = field(default_factory=lambda: deque())
    audio_dq: deque = None
    audio_start_idx: int = 0
    audio_end_idx: int = 0
    audio_accum_count: int = 0

    # Metrics
    created_at: float = field(default_factory=time.time)
    model_load_time: float = 0
    chunk_times: list = field(default_factory=list)
    total_frames: int = 0
    finished: bool = False

    # 人像抠图（透明背景），RVMMatting 实例或 None
    matting: object = None

    @property
    def slice_len(self) -> int:
        return self.infer_params['frame_num'] - self.infer_params['motion_frames_num']

    @property
    def chunk_audio_len(self) -> int:
        """Samples per processing chunk: slice_len * sample_rate / tgt_fps"""
        return self.slice_len * self.infer_params['sample_rate'] // self.infer_params['tgt_fps']

    @property
    def initialized(self) -> bool:
        return self.pipeline is not None


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> Session:
        async with self._lock:
            if len(self._sessions) >= MAX_SESSIONS:
                oldest = min(self._sessions, key=lambda k: self._sessions[k].created_at)
                await self._destroy(oldest)
            sid = uuid.uuid4().hex[:12]
            s = Session(id=sid)
            self._sessions[sid] = s
            return s

    async def get(self, sid: str) -> Optional[Session]:
        return self._sessions.get(sid)

    async def destroy(self, sid: str):
        async with self._lock:
            await self._destroy(sid)

    async def _destroy(self, sid: str):
        s = self._sessions.pop(sid, None)
        if s is not None:
            if s.pipeline is not None:
                del s.pipeline
            torch.cuda.empty_cache()
            logger.info(f"Session {sid} destroyed, VRAM freed")

    @property
    def count(self) -> int:
        return len(self._sessions)


sessions_mgr = SessionManager()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class InitMsg(BaseModel):
    type: str = "init"
    cond_image: str       # base64 bytes or local file path
    cond_is_path: bool = False
    base_seed: int = BASE_SEED
    use_face_crop: bool = False
    transparent_bg: bool = False

class InitResponse(BaseModel):
    type: str = "ready"
    session_id: str
    frame_num: int
    motion_frames_num: int
    slice_len: int
    chunk_audio_samples: int
    tgt_fps: int
    sample_rate: int
    model_load_time_s: float
    cond_preview: str = ""    # 透明背景开启时，返回抠图后的参考图 base64 PNG

class AudioMsg(BaseModel):
    type: str = "audio_chunk"
    audio: str            # base64-encoded audio data
    audio_format: str = "float32"  # float32 | int16 | wav

class FrameMeta(BaseModel):
    type: str = "frames_meta"
    chunk_idx: int
    frames_count: int
    height: int
    width: int
    processing_time_ms: float
    fmt: str = "jpeg"     # jpeg | webp（webp 时帧为 RGBA 带 alpha）


class FinishMsg(BaseModel):
    type: str = "finish"


class SummaryMsg(BaseModel):
    type: str = "finished"
    total_frames: int
    total_time_s: float
    avg_fps: float
    steady_state_fps: float
    first_chunk_ms: float
    avg_chunk_ms: float
    num_chunks: int


class ErrorMsg(BaseModel):
    type: str = "error"
    message: str


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def decode_audio(b64_data: str, fmt: str) -> np.ndarray:
    raw = base64.b64decode(b64_data)
    if fmt == "float32":
        return np.frombuffer(raw, dtype=np.float32).copy()
    elif fmt == "int16":
        return (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0).copy()
    elif fmt == "wav":
        with wave.open(io.BytesIO(raw), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            return (np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0).copy()
    raise ValueError(f"Unknown audio_format: {fmt}")


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pipeline runner (runs in thread pool to avoid blocking event loop)
# ---------------------------------------------------------------------------

def _do_init(msg: InitMsg, ckpt_dir: str, wav2vec_dir: str) -> tuple[Session, InitResponse]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()

    # Resolve image
    if msg.cond_is_path:
        cond = _resolve_path(msg.cond_image)
    else:
        raw_img = base64.b64decode(msg.cond_image)
        img_path = os.path.join(os.environ.get("TEMP", "/tmp"), f"_flashhead_{uuid.uuid4().hex[:8]}.png")
        with open(img_path, "wb") as f:
            f.write(raw_img)
        cond = img_path

    pipeline = get_pipeline(
        world_size=1,
        ckpt_dir=ckpt_dir,
        model_type=DEFAULT_MODEL_TYPE,
        wav2vec_dir=wav2vec_dir,
    )
    get_base_data(pipeline, cond, base_seed=msg.base_seed, use_face_crop=msg.use_face_crop)
    params = get_infer_params()

    # Init streaming state
    sr = params['sample_rate']
    dur = params['cached_audio_duration']
    fn = params['frame_num']
    mfn = params['motion_frames_num']
    fps = params['tgt_fps']
    sl = fn - mfn

    audio_dq = deque([0.0] * (sr * dur), maxlen=sr * dur)
    audio_start = dur * fps - fn
    audio_end = dur * fps

    torch.cuda.synchronize()
    load_time = time.time() - t0

    session = Session(id="pending")
    session.pipeline = pipeline
    session.infer_params = params
    session.person_name = list(pipeline.cond_image_dict.keys())[0]
    session.audio_dq = audio_dq
    session.audio_start_idx = audio_start
    session.audio_end_idx = audio_end

    if msg.transparent_bg or MATTING_ENABLED:
        try:
            session.matting = get_matting()
            gb = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"RVM matting enabled, GPU memory allocated: {gb:.1f} GB")
        except Exception as e:
            logger.error(f"RVM matting unavailable, fallback to JPEG: {e}")
            session.matting = None

    cond_preview_b64 = ""
    if session.matting is not None:
        try:
            import glob
            first = list(pipeline.cond_image_dict.keys())[0]
            pil_img = pipeline.cond_image_dict[first]
            arr = np.array(pil_img)
            matted = session.matting.apply(arr[np.newaxis, ...])[0]  # (H, W, 4)
            buf = io.BytesIO()
            Image.fromarray(matted).save(buf, format='PNG')
            cond_preview_b64 = base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            logger.warning(f"cond_preview matting failed: {e}")

    resp = InitResponse(
        session_id=session.id,
        frame_num=fn,
        motion_frames_num=mfn,
        slice_len=sl,
        chunk_audio_samples=sl * sr // fps,
        tgt_fps=fps,
        sample_rate=sr,
        model_load_time_s=round(load_time, 3),
        cond_preview=cond_preview_b64,
    )
    return session, resp


def _do_infer(session: Session, chunk: np.ndarray, chunk_idx: int) -> tuple[FrameMeta, list]:
    """GPU 推理 + 抠图 + 编码，全部在 thread pool 中完成，返回 meta + 预编码子批次列表"""
    p = session.pipeline
    ip = session.infer_params
    mfn = ip['motion_frames_num']

    torch.cuda.synchronize()
    t0 = time.time()

    session.audio_dq.extend(chunk.tolist())
    audio_arr = np.array(session.audio_dq)

    emb = get_audio_embedding(p, audio_arr, session.audio_start_idx, session.audio_end_idx)
    video = run_pipeline(p, emb)
    video = video[mfn:]  # (N, H, W, C) uint8

    torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000

    session.total_frames += video.shape[0]
    session.chunk_times.append(ms / 1000)

    n, h, w, c = video.shape
    frames_np = video.cpu().numpy().astype(np.uint8)
    fmt = 'jpeg'
    if session.matting is not None:
        try:
            t_mat = time.time()
            session.matting.reset()
            frames_np = session.matting.apply(frames_np)
            dt = time.time() - t_mat
            vram = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"[matting] {n}f in {dt:.3f}s ({dt*1000/n:.1f}ms/f) GPU={vram:.1f}GB")
            fmt = 'webp'
        except Exception as e:
            logger.error(f"matting.apply FAILED, disabling matting for session: {e}")
            session.matting = None
    meta = FrameMeta(
        chunk_idx=chunk_idx,
        frames_count=n,
        height=h,
        width=w,
        processing_time_ms=round(ms, 1),
        fmt=fmt,
    )
    # 在 thread pool 中编码子批次——不阻塞事件循环接收音频
    sub_batches = []
    for i in range(0, n, SUB_BATCH):
        parts = []
        for j in range(i, min(i + SUB_BATCH, n)):
            img = Image.fromarray(frames_np[j])
            buf = io.BytesIO()
            if fmt == 'webp':
                img.save(buf, format='WEBP', quality=80)
            else:
                img.save(buf, format='JPEG', quality=75)
            data = buf.getvalue()
            parts.append(len(data).to_bytes(4, 'little') + data)
        sub_batches.append(
            len(parts).to_bytes(4, 'little') + b''.join(parts)
        )
    return meta, sub_batches


def _encode_sub_batch(frames: np.ndarray, start: int, end: int, fmt: str) -> bytes:
    parts = []
    for j in range(start, end):
        img = Image.fromarray(frames[j])
        buf = io.BytesIO()
        if fmt == 'webp':
            img.save(buf, format='WEBP', quality=80)
        else:
            img.save(buf, format='JPEG', quality=75)
        data = buf.getvalue()
        parts.append(len(data).to_bytes(4, 'little') + data)
    return len(parts).to_bytes(4, 'little') + b''.join(parts)


def _do_finish(session: Session) -> SummaryMsg:
    if not session.chunk_times:
        return SummaryMsg(total_frames=0, total_time_s=0, avg_fps=0,
                          steady_state_fps=0, first_chunk_ms=0, avg_chunk_ms=0, num_chunks=0)
    total = sum(session.chunk_times)
    avg_fps = session.total_frames / total if total > 0 else 0
    if len(session.chunk_times) > 1:
        ss = sum(session.chunk_times[1:])
        # steady state: slice_len frames per chunk, for all chunks except first
        sf = (session.slice_len * (len(session.chunk_times) - 1)) / ss if ss > 0 else 0
    else:
        sf = avg_fps
    return SummaryMsg(
        total_frames=session.total_frames,
        total_time_s=round(total, 3),
        avg_fps=round(avg_fps, 1),
        steady_state_fps=round(sf, 1),
        first_chunk_ms=round(session.chunk_times[0] * 1000, 1),
        avg_chunk_ms=round(
            sum(session.chunk_times[1:]) / (len(session.chunk_times) - 1) * 1000
            if len(session.chunk_times) > 1 else session.chunk_times[0] * 1000, 1
        ),
        num_chunks=len(session.chunk_times),
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="FlashHead Streaming API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sessions": sessions_mgr.count,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "vram_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2) if torch.cuda.is_available() else 0,
    }


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    session = await sessions_mgr.create()
    logger.info(f"WS connected: {session.id}")

    try:
        chunk_idx = 0

        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type", "")

            # --- INIT ---
            if mtype == "init":
                im = InitMsg(**msg)
                loop = asyncio.get_event_loop()
                new_session, resp = await loop.run_in_executor(None, _do_init, im, CKPT_DIR, WAV2VEC_DIR)

                # Transfer identity
                old_id = session.id
                session.pipeline = new_session.pipeline
                session.infer_params = new_session.infer_params
                session.person_name = new_session.person_name
                session.audio_dq = new_session.audio_dq
                session.audio_start_idx = new_session.audio_start_idx
                session.audio_end_idx = new_session.audio_end_idx
                session.model_load_time = new_session.model_load_time
                session.matting = new_session.matting
                sessions_mgr._sessions.pop(new_session.id, None)
                resp.session_id = old_id
                await ws.send_text(resp.model_dump_json())
                logger.info(f"Sess {session.id} init OK, load={session.model_load_time:.1f}s")

            # --- AUDIO CHUNK ---
            elif mtype == "audio_chunk":
                if not session.initialized:
                    await ws.send_text(ErrorMsg(message="Not initialized. Send init first.").model_dump_json())
                    continue

                am = AudioMsg(**msg)
                arr = decode_audio(am.audio, am.audio_format)

                # Buffer: accumulate audio, emit frames for each full chunk
                session.audio_accum_count += len(arr)
                session.audio_buffer.extend(arr)

                chunk_size = session.chunk_audio_len
                while len(session.audio_buffer) >= chunk_size:
                    chunk = np.array([session.audio_buffer.popleft() for _ in range(chunk_size)], dtype=np.float32)
                    loop = asyncio.get_event_loop()
                    meta, sub_batches = await loop.run_in_executor(None, _do_infer, session, chunk, chunk_idx)
                    chunk_idx += 1
                    await ws.send_text(meta.model_dump_json())
                    for sub in sub_batches:
                        await ws.send_bytes(sub)

            # --- FINISH ---
            elif mtype == "finish":
                # Flush remaining audio buffer (pad to full chunk)
                remaining = len(session.audio_buffer)
                if remaining > 0:
                    chunk_size = session.chunk_audio_len
                    pad_needed = chunk_size - remaining
                    chunk = np.zeros(chunk_size, dtype=np.float32)
                    for i in range(remaining):
                        chunk[i] = session.audio_buffer.popleft()
                    logger.info(f"Sess {session.id} flushing {remaining} samples (+{pad_needed} pad)")
                    loop = asyncio.get_event_loop()
                    meta, sub_batches = await loop.run_in_executor(None, _do_infer, session, chunk, chunk_idx)
                    chunk_idx += 1
                    await ws.send_text(meta.model_dump_json())
                    for sub in sub_batches:
                        await ws.send_bytes(sub)

                summary = _do_finish(session)
                await ws.send_text(summary.model_dump_json())
                session.finished = True
                logger.info(f"Sess {session.id} finished: {summary.total_frames}f, {summary.avg_fps}fps")
                break

            # --- FLUSH (pad buffer + generate, keep session alive) ---
            elif mtype == "flush":
                if session.initialized:
                    remaining = len(session.audio_buffer)
                    if remaining > 0:
                        chunk_size = session.chunk_audio_len
                        chunk = np.zeros(chunk_size, dtype=np.float32)
                        for i in range(remaining):
                            chunk[i] = session.audio_buffer.popleft()
                        loop = asyncio.get_event_loop()
                        meta, sub_batches = await loop.run_in_executor(None, _do_infer, session, chunk, chunk_idx)
                        chunk_idx += 1
                        await ws.send_text(meta.model_dump_json())
                        for sub in sub_batches:
                            await ws.send_bytes(sub)
                    await ws.send_text(json.dumps({"type": "flushed"}))

            # --- CLEAR (discard buffered audio, keep session alive) ---
            elif mtype == "clear":
                session.audio_buffer.clear()
                if session.matting is not None:
                    session.matting.reset()
                await ws.send_text(json.dumps({"type": "cleared"}))

            # --- RESET (switch person or re-seed) ---
            elif mtype == "reset":
                if session.initialized:
                    pn = msg.get("person_name", session.person_name)
                    session.pipeline.reset_person_name(pn)
                    if session.matting is not None:
                        session.matting.reset()
                    await ws.send_text(json.dumps({"type": "reset_ok", "person": pn}))

            else:
                await ws.send_text(ErrorMsg(message=f"Unknown type: {mtype}").model_dump_json())

    except WebSocketDisconnect:
        logger.info(f"WS disconnect: {session.id}")
    except Exception as e:
        logger.error(f"WS error {session.id}: {e}")
        try:
            await ws.send_text(ErrorMsg(message=str(e)).model_dump_json())
        except Exception:
            pass
    finally:
        await sessions_mgr.destroy(session.id)


# ---------------------------------------------------------------------------
# Web UI (Vue 3 SPA)
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return HTMLResponse(HTML_UI)


HTML_UI = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SoulX-FlashHead - AI Talking Head</title>
<script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #14141f;
    --border: #1e1e32;
    --primary: #7c5cfc;
    --primary-glow: rgba(124, 92, 252, 0.25);
    --text: #e0e0f0;
    --text-dim: #8888aa;
    --danger: #ff4d6a;
    --success: #2ed8a3;
    --warning: #f0a050;
    --radius: 12px;
    --radius-sm: 8px;
    --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: var(--font); min-height: 100vh; }
  #app { display: flex; min-height: 100vh; }
  /* ── Sidebar ── */
  .sidebar {
    width: 360px; min-width: 360px; background: var(--surface);
    border-right: 1px solid var(--border); display: flex; flex-direction: column;
    padding: 24px; gap: 20px; overflow-y: auto;
  }
  .logo { font-size: 1.2rem; font-weight: 700; letter-spacing: -0.02em; }
  .logo span { color: var(--primary); }
  .label { font-size: 0.78rem; font-weight: 600; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; }
  /* upload zones */
  .upload-zone {
    border: 2px dashed var(--border); border-radius: var(--radius);
    padding: 20px; text-align: center; cursor: pointer; transition: all .15s;
    background: rgba(255,255,255,0.015); min-height: 140px; display: flex;
    flex-direction: column; align-items: center; justify-content: center; gap: 8px;
  }
  .upload-zone:hover { border-color: var(--primary); background: rgba(124,92,252,0.06); }
  .upload-zone.has-file { border-color: var(--success); border-style: solid; }
  .upload-zone img { max-width: 100%; max-height: 200px; border-radius: var(--radius-sm); object-fit: cover; }
  .upload-zone audio { width: 100%; margin-top: 4px; }
  .upload-icon { font-size: 2rem; opacity: 0.5; }
  .file-name { font-size: 0.82rem; color: var(--text-dim); word-break: break-all; }
  /* buttons */
  .btn {
    display: inline-flex; align-items: center; justify-content: center; gap: 6px;
    padding: 10px 24px; border: none; border-radius: var(--radius-sm);
    font-size: 0.9rem; font-weight: 600; cursor: pointer; transition: all .15s;
    font-family: var(--font);
  }
  .btn-primary { background: var(--primary); color: #fff; }
  .btn-primary:hover:not(:disabled) { box-shadow: 0 0 24px var(--primary-glow); transform: translateY(-1px); }
  .btn-danger { background: transparent; border: 1px solid var(--danger); color: var(--danger); }
  .btn-danger:hover:not(:disabled) { background: rgba(255,77,106,0.12); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-row { display: flex; gap: 10px; }
  .btn-row .btn { flex: 1; }
  /* select & inputs */
  select, input[type=number] {
    width: 100%; padding: 8px 12px; border-radius: var(--radius-sm);
    border: 1px solid var(--border); background: var(--bg); color: var(--text);
    font-size: 0.85rem; font-family: var(--font);
  }
  select:focus, input:focus { outline: none; border-color: var(--primary); }
  .field { margin-bottom: 8px; }
  /* progress */
  .progress-bar {
    height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-top: 6px;
  }
  .progress-fill { height: 100%; background: var(--primary); border-radius: 2px; transition: width .3s; }
  .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .metric-card {
    background: rgba(255,255,255,0.03); border-radius: var(--radius-sm);
    padding: 10px 12px; text-align: center;
  }
  .metric-val { font-size: 1.4rem; font-weight: 700; color: var(--primary); }
  .metric-label { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; }
  /* ── Main ── */
  .main {
    flex: 1; display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 32px; position: relative;
  }
  .video-container {
    position: relative; border-radius: var(--radius); overflow: hidden;
    box-shadow: 0 0 60px rgba(124,92,252,0.15), 0 0 120px rgba(0,0,0,0.5);
    background: #000;
  }
  #videoCanvas { display: block; width: 512px; height: 512px; background: #000; }
  .placeholder {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 12px; pointer-events: none;
  }
  .placeholder-icon { font-size: 4rem; opacity: 0.3; }
  .placeholder-text { font-size: 0.9rem; color: var(--text-dim); }
  .status-badge {
    position: absolute; bottom: -36px; left: 50%; transform: translateX(-50%);
    padding: 4px 16px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;
    white-space: nowrap;
  }
  .status-idle { background: rgba(255,255,255,0.06); color: var(--text-dim); }
  .status-loading { background: rgba(240,160,80,0.15); color: var(--warning); }
  .status-streaming { background: rgba(46,216,163,0.15); color: var(--success); }
  .status-error { background: rgba(255,77,106,0.15); color: var(--danger); }
  .fps-overlay {
    position: absolute; top: 12px; right: 12px; background: rgba(0,0,0,0.7);
    padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 700;
    color: var(--success); font-variant-numeric: tabular-nums;
  }
  @media (max-width: 900px) {
    #app { flex-direction: column; }
    .sidebar { width: 100%; min-width: 0; }
    #videoCanvas { width: min(512px, 90vw); height: min(512px, 90vw); }
  }
</style>
</head>
<body>
<div id="app">
  <div class="sidebar">
    <div class="logo">SoulX <span>FlashHead</span></div>

    <div>
      <div class="label">Source Image</div>
      <div class="upload-zone" :class="{ 'has-file': imageFile }" @click="$refs.imgInput.click()" @dragover.prevent @drop.prevent="onImgDrop">
        <template v-if="imagePreview">
          <img :src="imagePreview" alt="Preview">
        </template>
        <template v-else>
          <div class="upload-icon">🖼️</div>
          <div style="font-size:0.82rem">Click or drag image here</div>
        </template>
        <div class="file-name" v-if="imageFile">{{ imageFile.name }}</div>
      </div>
      <input type="file" ref="imgInput" accept="image/*" @change="onImgChange" hidden>
    </div>

    <div>
      <div class="label">Audio Track</div>
      <div class="upload-zone" :class="{ 'has-file': audioFile }" @click="$refs.audioInput.click()" @dragover.prevent @drop.prevent="onAudioDrop">
        <template v-if="audioFile">
          <audio :src="audioPreview" controls style="width:100%"></audio>
          <div class="file-name">{{ audioFile.name }}</div>
        </template>
        <template v-else>
          <div class="upload-icon">🎙️</div>
          <div style="font-size:0.82rem">Click or drag WAV file</div>
        </template>
      </div>
      <input type="file" ref="audioInput" accept="audio/wav,.wav" @change="onAudioChange" hidden>
    </div>

    <div style="margin-top:auto;">
      <div class="btn-row">
        <button class="btn btn-primary" style="flex:2" @click="start" :disabled="running || !audioFile">
          {{ running ? 'Generating…' : '▶ Start' }}
        </button>
        <button class="btn btn-danger" @click="stop" :disabled="!running" style="flex:1">⏹ Stop</button>
      </div>
      <div class="progress-bar" v-if="running">
        <div class="progress-fill" :style="{ width: progress + '%' }"></div>
      </div>
    </div>

    <div class="metrics-grid" v-if="metrics">
      <div class="metric-card"><div class="metric-val">{{ metrics.avgFps }}</div><div class="metric-label">Avg FPS</div></div>
      <div class="metric-card"><div class="metric-val">{{ metrics.firstMs }}ms</div><div class="metric-label">First Frame</div></div>
      <div class="metric-card"><div class="metric-val">{{ metrics.chunks }}</div><div class="metric-label">Chunks</div></div>
      <div class="metric-card"><div class="metric-val">{{ metrics.totalFrames }}</div><div class="metric-label">Frames</div></div>
    </div>
  </div>

  <div class="main">
    <div class="video-container">
      <canvas id="videoCanvas" width="512" height="512"></canvas>
      <div class="placeholder" v-if="!running && !hasResult">
        <div class="placeholder-icon">🎬</div>
        <div class="placeholder-text">Upload image + audio and press Start</div>
      </div>
      <div class="fps-overlay" v-if="currentFps">~{{ currentFps }} fps</div>
    </div>
    <div class="status-badge" :class="'status-' + statusClass">{{ statusText }}</div>
  </div>
</div>
<script>
const { createApp, ref, reactive, nextTick } = Vue

createApp({
  setup() {
    const imgInput = ref(null)
    const audioInput = ref(null)
    const imageFile = ref(null)
    const imagePreview = ref(null)
    const audioFile = ref(null)
    const audioPreview = ref(null)
    const running = ref(false)
    const hasResult = ref(false)
    const statusText = ref('Ready')
    const statusClass = ref('idle')
    const progress = ref(0)
    const currentFps = ref('')
    const metrics = ref(null)
    const ws = ref(null)
    const audioCtx = ref(null)

    function reset() {
      progress.value = 0; currentFps.value = ''; metrics.value = null; hasResult.value = false
      const c = document.getElementById('videoCanvas')
      if (c) { const ctx = c.getContext('2d'); ctx.clearRect(0, 0, 512, 512) }
    }

    function previewFile(file, cb) {
      const r = new FileReader()
      r.onload = e => cb(e.target.result)
      r.readAsDataURL(file)
    }

    function onImgChange(e) {
      const f = e.target.files[0]
      if (!f) return
      imageFile.value = f; cachedImg.value = null
      previewFile(f, v => imagePreview.value = v)
      reset()
    }

    function onAudioChange(e) {
      const f = e.target.files[0]
      if (!f) return
      audioFile.value = f
      audioPreview.value = URL.createObjectURL(f)
      reset()
    }

    function onImgDrop(e) {
      const f = e.dataTransfer.files[0]
      if (f && f.type.startsWith('image/')) { imageFile.value = f; cachedImg.value = null; previewFile(f, v => imagePreview.value = v); reset() }
    }

    function onAudioDrop(e) {
      const f = e.dataTransfer.files[0]
      if (f && (f.type.includes('wav') || f.name.endsWith('.wav'))) { audioFile.value = f; audioPreview.value = URL.createObjectURL(f); reset() }
    }

    function setStatus(cls, txt) { statusClass.value = cls; statusText.value = txt }

    // ---- smooth playback ----
    let frameBuffer = [], playbackTimer = null, playbackIdx = 0, canvasCtx = null
    const TARGET_FPS = 25

    function initPlayback() {
      const c = document.getElementById('videoCanvas')
      canvasCtx = c.getContext('2d')
      frameBuffer = []; playbackIdx = 0
      if (playbackTimer) { clearInterval(playbackTimer); playbackTimer = null }
    }

    function startPlayback() {
      if (playbackTimer) return
      const interval = 1000 / TARGET_FPS
      playbackTimer = setInterval(() => {
        if (!running.value && frameBuffer.length === 0) { stopPlayback(); return }
        if (playbackIdx >= frameBuffer.length) return // wait for more frames
        const { bitmap, w, h } = frameBuffer[playbackIdx]
        canvasCtx.canvas.width = w; canvasCtx.canvas.height = h
        canvasCtx.drawImage(bitmap, 0, 0)
        bitmap.close()
        playbackIdx++
        currentFps.value = TARGET_FPS
      }, interval)
    }

    function stopPlayback() {
      if (playbackTimer) { clearInterval(playbackTimer); playbackTimer = null }
      currentFps.value = ''
    }

    // JPEG batch: [4B count][4B len][jpeg]...
    async function pushFrames(rawBytes, n, h, w) {
      const view = new DataView(rawBytes)
      const count = view.getUint32(0, true)
      let offset = 4
      const blobs = []
      for (let i = 0; i < count; i++) {
        const len = view.getUint32(offset, true)
        offset += 4
        blobs.push(new Blob([new Uint8Array(rawBytes, offset, len)], { type: 'image/jpeg' }))
        offset += len
      }
      const bitmaps = await Promise.all(blobs.map(b => createImageBitmap(b).catch(() => null)))
      for (let i = 0; i < bitmaps.length; i++) {
        if (bitmaps[i]) frameBuffer.push({ bitmap: bitmaps[i], w, h })
      }
    }

    const cachedImg = ref(null)  // { name, b64 } — avoid re-encoding same file

    async function getImageB64() {
      const f = imageFile.value
      if (!f) return ''
      if (cachedImg.value && cachedImg.value.name === f.name && cachedImg.value.size === f.size) {
        return cachedImg.value.b64
      }
      const buf = await f.arrayBuffer()
      const bytes = new Uint8Array(buf)
      let bin = ''; for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i])
      const b64 = btoa(bin)
      cachedImg.value = { name: f.name, size: f.size, b64 }
      return b64
    }

    function reset() {
      progress.value = 0; currentFps.value = ''; metrics.value = null; hasResult.value = false
      stopPlayback()
      frameBuffer = []; playbackIdx = 0; canvasCtx = null
      const c = document.getElementById('videoCanvas')
      if (c) { const ctx = c.getContext('2d'); ctx.clearRect(0, 0, 512, 512) }
    }

    async function start() {
      if (!audioFile.value) return
      reset(); running.value = true; setStatus('loading', 'Initializing model...')
      initPlayback()

      // Read image as base64 (cached, only re-encodes when file changes)
      const imgB64 = await getImageB64()

      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
      const sock = new WebSocket(proto + '//' + location.host + '/ws/stream')
      sock.binaryType = 'arraybuffer'
      ws.value = sock

      let lastMeta = null, chunkIdx = 0, totalFrames = 0, totalMs = 0

      sock.onopen = () => {
        sock.send(JSON.stringify({
          type: 'init',
          cond_image: imgB64 || 'examples/girl.png',
          cond_is_path: !imgB64,
          base_seed: Math.floor(Math.random() * 1000)
        }))
      }

      sock.onmessage = async (e) => {
        if (e.data instanceof ArrayBuffer) {
          if (!lastMeta) return
          const { frames_count, height, width, chunk_idx, processing_time_ms } = lastMeta
          totalFrames += frames_count; totalMs += processing_time_ms
          progress.value = Math.min(99, Math.round((chunk_idx / (audioFile.value.size / 40000)) * 100))
          pushFrames(e.data, frames_count, height, width)
          hasResult.value = true
          if (chunk_idx === 0) startPlayback() // start on first chunk
          lastMeta = null
        } else {
          const m = JSON.parse(e.data)
          if (m.type === 'ready') {
            setStatus('streaming', 'Streaming…')
            streamAudioFile(audioFile.value, m.slice_len, m.sample_rate, m.chunk_audio_samples)
          } else if (m.type === 'frames_meta') {
            lastMeta = m
          } else if (m.type === 'finished') {
            progress.value = 100
            metrics.value = {
              avgFps: m.avg_fps, firstMs: m.first_chunk_ms,
              chunks: m.num_chunks, totalFrames: m.total_frames
            }
            setStatus('idle', 'Done')
            running.value = false; sock.close()
            // playback loop will drain remaining frames naturally
          } else if (m.type === 'error') {
            setStatus('error', m.message); running.value = false
          }
        }
      }

      sock.onerror = () => { setStatus('error', 'Connection error'); running.value = false }
      sock.onclose = () => { if (running.value) { setStatus('idle', 'Disconnected'); running.value = false } }
    }

    async function streamAudioFile(file, sliceLen, sampleRate, chunkSamples) {
      const ab = await file.arrayBuffer()
      const ctx = new OfflineAudioContext(1, ab.byteLength / (file.type === 'audio/wav' ? 2 : 1), sampleRate)
      const buf = await ctx.decodeAudioData(ab)
      const samples = buf.getChannelData(0)

      for (let i = 0; i < samples.length; i += chunkSamples) {
        if (!running.value) break
        let chunk = samples.slice(i, i + chunkSamples)
        if (chunk.length < chunkSamples) {
          const p = new Float32Array(chunkSamples); p.set(chunk); chunk = p
        }
        const raw = new Uint8Array(chunk.buffer)
        let b64 = ''
        for (let j = 0; j < raw.length; j++) b64 += String.fromCharCode(raw[j])
        ws.value.send(JSON.stringify({ type: 'audio_chunk', audio: btoa(b64), audio_format: 'float32' }))
        await new Promise(r => setTimeout(r, 1))
      }
      if (running.value) ws.value.send(JSON.stringify({ type: 'finish' }))
    }

    function stop() {
      running.value = false; stopPlayback()
      if (ws.value) { ws.value.close(); ws.value = null }
      setStatus('idle', 'Stopped')
    }

    return {
      imgInput, audioInput, imageFile, imagePreview, audioFile, audioPreview,
      running, hasResult, statusText, statusClass,
      progress, currentFps, metrics,
      onImgChange, onAudioChange, onImgDrop, onAudioDrop,
      start, stop
    }
  }
}).mount('#app')
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
