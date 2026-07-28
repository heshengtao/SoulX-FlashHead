import sys, os, time, numpy as np, torch, librosa
sys.path.insert(0, '.')
os.environ['FLASHHEAD_CKPT_DIR'] = 'models/SoulX-FlashHead-1_3B'
os.environ['FLASHHEAD_WAV2VEC_DIR'] = 'models/wav2vec2-base-960h'

import flash_head.src.pipeline.flash_head_pipeline as _cfg
_cfg.COMPILE_MODEL = False
_cfg.COMPILE_VAE = False

from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline

print('Loading pipeline...')
t0 = time.time()
pipeline = get_pipeline(world_size=1, ckpt_dir='models/SoulX-FlashHead-1_3B', model_type='lite', wav2vec_dir='models/wav2vec2-base-960h')
get_base_data(pipeline, 'examples/girl.png', base_seed=42, use_face_crop=False)
params = get_infer_params()
fn = params['frame_num']
mfn_latent = params['motion_frames_latent_num']
# motion_frames_num = (mfn_latent - 1) * vae_stride + 1; for lite VAE stride[0]=8
mfn = (mfn_latent - 1) * 8 + 1
sl = fn - mfn
chunk_samples = sl * params['sample_rate'] // params['tgt_fps']
print(f'Init: {time.time()-t0:.1f}s (res={params["height"]}x{params["width"]}, slice_len={sl}, chunk_samples={chunk_samples})')

audio, sr = librosa.load('examples/podcast_sichuan_16k.wav', sr=16000)
total_chunks = len(audio) // chunk_samples
print(f'Audio: {len(audio)/16000:.1f}s, {total_chunks} chunks of {chunk_samples} samples')

# warmup
chunk = audio[:chunk_samples].astype(np.float32)
from collections import deque
audio_dq = deque(audio[:params['cached_audio_duration']*16000].tolist(), maxlen=params['cached_audio_duration']*16000)
emb = get_audio_embedding(pipeline, np.array(audio_dq), params['cached_audio_duration']*params['tgt_fps']-fn, params['cached_audio_duration']*params['tgt_fps'])
_ = run_pipeline(pipeline, emb)
torch.cuda.synchronize()

# benchmark
times = []
for ci in range(min(total_chunks - 1, 20)):
    start = ci * chunk_samples
    chunk = audio[start:start+chunk_samples].astype(np.float32)
    audio_dq.extend(chunk.tolist())
    t1 = time.time()
    emb = get_audio_embedding(pipeline, np.array(audio_dq), 8*25-fn, 8*25)
    video = run_pipeline(pipeline, emb)
    video = video[mfn:]  # (N, H, W, C)
    torch.cuda.synchronize()
    dt = time.time() - t1
    times.append(dt)
    n_frames = video.shape[0]
    print(f'  chunk {ci+1}: {dt*1000:.0f}ms ({n_frames}f, {n_frames/dt:.1f}fps)')

avg = sum(times) / len(times) if times else 0
need = 0.96
print(f'Average: {avg*1000:.0f}ms/chunk, throughput: {sl/avg:.1f}fps (need 25fps)')
if avg <= need:
    print('REAL-TIME: YES')
else:
    print(f'REAL-TIME: NO ({avg/need:.1f}x slower than needed)')
