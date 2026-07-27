import argparse
import os
import sys
import time

# Windows: disable torch.compile (Triton not available)
if sys.platform == "win32":
    import flash_head.src.pipeline.flash_head_pipeline as _pipe_cfg
    _pipe_cfg.COMPILE_MODEL = False
    _pipe_cfg.COMPILE_VAE = False

import torch
import numpy as np
import librosa
from collections import deque
from datetime import datetime
from loguru import logger

from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline


def format_bytes(n_bytes):
    if n_bytes < 1024:
        return f"{n_bytes:.0f} B"
    elif n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    elif n_bytes < 1024 ** 3:
        return f"{n_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{n_bytes / 1024 ** 3:.2f} GB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="models/SoulX-FlashHead-1_3B")
    parser.add_argument("--wav2vec_dir", type=str, default="models/wav2vec2-base-960h")
    parser.add_argument("--model_type", type=str, default="lite")
    parser.add_argument("--cond_image", type=str, default="examples/girl.png")
    parser.add_argument("--audio_path", type=str, default="examples/podcast_sichuan_16k.wav")
    parser.add_argument("--save_file", type=str, default=None)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SoulX-FlashHead Lite Model Benchmark on RTX 4080")
    logger.info("=" * 60)

    # --- 1. Model loading time ---
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()

    pipeline = get_pipeline(
        world_size=1,
        ckpt_dir=args.ckpt_dir,
        wav2vec_dir=args.wav2vec_dir,
        model_type=args.model_type,
    )

    torch.cuda.synchronize()
    model_load_time = time.time() - t0
    vram_after_load = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    logger.info(f"Model load time: {model_load_time:.2f}s")
    logger.info(f"VRAM after load: {format_bytes(vram_after_load)}")

    # --- 2. Base data preparation ---
    get_base_data(pipeline, args.cond_image, base_seed=42, use_face_crop=False)
    infer_params = get_infer_params()

    sample_rate = infer_params['sample_rate']
    tgt_fps = infer_params['tgt_fps']
    cached_audio_duration = infer_params['cached_audio_duration']
    frame_num = infer_params['frame_num']
    motion_frames_num = infer_params['motion_frames_num']
    slice_len = frame_num - motion_frames_num

    logger.info(f"Target FPS: {tgt_fps}, Frame num per chunk: {frame_num}, Motion frames: {motion_frames_num}")

    human_speech_array_all, _ = librosa.load(args.audio_path, sr=sample_rate, mono=True)
    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
    human_speech_array_frame_num = frame_num * sample_rate // tgt_fps

    # Pad audio
    remainder = len(human_speech_array_all) % human_speech_array_slice_len
    if remainder > 0:
        pad_length = human_speech_array_slice_len - remainder
        human_speech_array_all = np.concatenate(
            [human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)]
        )

    # --- 3. Inference with metrics ---
    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num

    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
    human_speech_array_slices = human_speech_array_all.reshape(-1, human_speech_array_slice_len)

    num_chunks = human_speech_array_slices.shape[0]
    total_frames_generated = 0
    total_inference_time = 0
    chunk_times = []
    first_frame_latency = None

    logger.info(f"Audio duration: {len(human_speech_array_all) / sample_rate:.1f}s")
    logger.info(f"Total chunks: {num_chunks}")
    logger.info(f"Starting inference...")

    torch.cuda.reset_peak_memory_stats()

    for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
        torch.cuda.synchronize()
        start_time = time.time()

        audio_dq.extend(human_speech_array.tolist())
        audio_array = np.array(audio_dq)
        audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)

        video = run_pipeline(pipeline, audio_embedding)
        video = video[motion_frames_num:]

        torch.cuda.synchronize()
        end_time = time.time()
        chunk_time = end_time - start_time
        chunk_times.append(chunk_time)
        total_inference_time += chunk_time

        frames_in_chunk = video.shape[0]
        total_frames_generated += frames_in_chunk

        if chunk_idx == 0:
            first_frame_latency = chunk_time

        logger.info(
            f"Chunk {chunk_idx + 1}/{num_chunks}: "
            f"{chunk_time:.3f}s | {frames_in_chunk} frames | "
            f"{frames_in_chunk / chunk_time:.1f} FPS"
        )

    peak_vram = torch.cuda.max_memory_allocated()

    # --- 4. Print summary ---
    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    logger.info(f"GPU: NVIDIA GeForce RTX 4080 (16GB)")
    logger.info(f"Model: SoulX-FlashHead Lite (1.3B)")
    logger.info(f"Resolution: {infer_params['width']}x{infer_params['height']} @ {tgt_fps} FPS target")
    logger.info(f"Sampling steps: {infer_params['sample_steps']}")
    logger.info(f"FlashAttention: {getattr(pipeline.model, 'flash_attn_available', 'N/A (fallback to SDPA)')}")
    logger.info("-" * 60)
    logger.info(f"Total audio duration: {len(human_speech_array_all) / sample_rate:.1f}s")
    logger.info(f"Total chunks processed: {num_chunks}")
    logger.info(f"Total frames generated: {total_frames_generated}")
    logger.info(f"Total inference time: {total_inference_time:.2f}s")
    logger.info("-" * 60)
    logger.info(f"First-frame latency: {first_frame_latency:.3f}s")
    logger.info(f"Average chunk time: {np.mean(chunk_times[1:]):.3f}s" if len(chunk_times) > 1 else "N/A")
    logger.info(f"Average FPS (all chunks): {total_frames_generated / total_inference_time:.1f}")
    logger.info(f"Max single-chunk FPS: {max([(slice_len / t) for t in chunk_times]):.1f}")
    logger.info(f"Steady-state FPS (excl. first): {sum([slice_len for _ in chunk_times[1:]]) / sum(chunk_times[1:]):.1f}" if len(chunk_times) > 1 else "N/A")
    logger.info("-" * 60)
    logger.info(f"Peak VRAM usage: {format_bytes(peak_vram)}")
    logger.info(f"VRAM after model load: {format_bytes(vram_after_load)}")
    logger.info("=" * 60)

    saved = getattr(args, 'save_file', None)
    logger.info(f"Output saved to: {saved if saved else 'N/A (not saved)'}")


if __name__ == "__main__":
    main()
