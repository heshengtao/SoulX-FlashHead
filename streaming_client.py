"""
Python client for FlashHead streaming server (binary frames protocol).

Protocol (per chunk):
  1. Server sends JSON: {"type": "frames_meta", "chunk_idx": N, "frames_count": N, "height": H, "width": W, "processing_time_ms": M}
  2. Server sends binary: raw uint8 bytes of shape (N, H, W, C) packed C-contiguous

Usage:
  python streaming_client.py --audio examples/podcast_sichuan_16k.wav
"""

import argparse
import asyncio
import base64
import io
import json
import os
import struct
import sys
import time
import wave

import numpy as np
import websockets
from PIL import Image


async def load_wav_as_float32(path: str) -> np.ndarray:
    with wave.open(path, 'rb') as wf:
        n = wf.getnframes()
        data = wf.readframes(n)
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def save_frames_as_mp4(frames_list: list, path: str, fps: int = 25):
    """Save accumulated frames as MP4."""
    import imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = imageio.get_writer(path, format='mp4', mode='I', fps=fps, codec='h264',
                                ffmpeg_params=['-bf', '0'])
    for frames in frames_list:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])
    writer.close()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="examples/podcast_sichuan_16k.wav")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--url", type=str, default="ws://127.0.0.1:8765/ws/stream")
    parser.add_argument("--model", type=str, default="lite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None, help="Save output video to path")
    args = parser.parse_args()

    audio = await load_wav_as_float32(args.audio)
    print(f"Loaded audio: {len(audio)} samples ({len(audio) / 16000:.1f}s)")

    all_frames = []  # accumulate for optional save

    # Connect with larger max_size for binary frames
    async with websockets.connect(
        args.url, ping_interval=30, max_size=50 * 1024 * 1024  # 50MB
    ) as ws:

        # ---- INIT ----
        if args.image:
            with open(args.image, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
            init_msg = json.dumps({
                "type": "init", "cond_image": img_b64,
                "cond_is_path": False, "model_type": args.model, "base_seed": args.seed,
            })
        else:
            init_msg = json.dumps({
                "type": "init", "cond_image": "examples/girl.png",
                "cond_is_path": True, "model_type": args.model, "base_seed": args.seed,
            })

        print("Sending init...")
        t0 = time.time()
        await ws.send(init_msg)

        resp = json.loads(await ws.recv())
        if resp["type"] == "error":
            print(f"ERROR: {resp['message']}")
            return

        assert resp["type"] == "ready"
        slice_len = resp["slice_len"]
        sample_rate = resp["sample_rate"]
        tgt_fps = resp["tgt_fps"]
        chunk_audio_samples = resp["chunk_audio_samples"]
        print(f"Init OK ({resp['model_load_time_s']}s)")
        print(f"  slice_len={slice_len}, chunk_samples={chunk_audio_samples}, fps={tgt_fps}")

        # ---- STREAM AUDIO CHUNKS ----
        total_frames = 0
        chunk_idx = 0
        audio_sent = 0

        for i in range(0, len(audio), chunk_audio_samples):
            raw = audio[i:i + chunk_audio_samples]
            if len(raw) < chunk_audio_samples:
                padded = np.zeros(chunk_audio_samples, dtype=np.float32)
                padded[:len(raw)] = raw
                raw = padded

            b64 = base64.b64encode(raw.tobytes()).decode()
            await ws.send(json.dumps({
                "type": "audio_chunk", "audio": b64, "audio_format": "float32",
            }))
            audio_sent += chunk_audio_samples

            # Receive: JSON meta + binary frames
            meta_raw = await ws.recv()
            meta = json.loads(meta_raw)

            if meta["type"] == "error":
                print(f"  ERROR: {meta['message']}")
                return
            elif meta["type"] == "frames_meta":
                bin_data = await ws.recv()
                n, h, w = meta["frames_count"], meta["height"], meta["width"]
                frames = np.frombuffer(bin_data, dtype=np.uint8).reshape(n, h, w, 3).copy()
                all_frames.append(frames)
                total_frames += n
                fps = n / (meta["processing_time_ms"] / 1000) if meta["processing_time_ms"] > 0 else 0
                print(f"  Chunk {meta['chunk_idx']}: {n} frames ({h}x{w}), "
                      f"{meta['processing_time_ms']:.0f}ms, {fps:.1f} fps")
                chunk_idx += 1
            else:
                print(f"  Unexpected meta type: {meta.get('type')}")

        # ---- FINISH ----
        await ws.send(json.dumps({"type": "finish"}))

        # Consume remaining (flushed chunk + summary)
        while True:
            resp = await ws.recv()
            if isinstance(resp, bytes):
                print(f"  [Flush] binary data, {len(resp)} bytes")
                continue
            resp = json.loads(resp)
            if resp["type"] == "frames_meta":
                bin_data = await ws.recv()
                n, h, w = resp["frames_count"], resp["height"], resp["width"]
                frames = np.frombuffer(bin_data, dtype=np.uint8).reshape(n, h, w, 3).copy()
                all_frames.append(frames)
                total_frames += n
                print(f"  [Flush] Chunk {resp['chunk_idx']}: {n} frames, {resp['processing_time_ms']:.0f}ms")
            elif resp["type"] == "finished":
                break
            else:
                print(f"  Unexpected: {resp.get('type')}")

        total_elapsed = time.time() - t0
        print()
        print("=" * 60)
        print("SUMMARY")
        print(f"  Total frames: {resp['total_frames']}")
        print(f"  Inference time: {resp['total_time_s']}s")
        print(f"  Wall-clock time: {total_elapsed:.1f}s")
        print(f"  Avg FPS: {resp['avg_fps']}")
        print(f"  Steady-state FPS: {resp['steady_state_fps']}")
        print(f"  First chunk: {resp['first_chunk_ms']}ms")
        print(f"  Avg chunk: {resp['avg_chunk_ms']}ms")
        print(f"  Chunks: {resp['num_chunks']}")
        print("=" * 60)

        # Optional save
        if args.save and all_frames:
            print(f"Saving video to {args.save}...")
            save_frames_as_mp4(all_frames, args.save, fps=tgt_fps)
            print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
