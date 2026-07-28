import sys, os, time, numpy as np, torch, librosa
sys.path.insert(0, '.')
os.environ['FLASHHEAD_CKPT_DIR'] = 'models/SoulX-FlashHead-1_3B'
os.environ['FLASHHEAD_WAV2VEC_DIR'] = 'models/wav2vec2-base-960h'

import flash_head.src.pipeline.flash_head_pipeline as _cfg
_cfg.COMPILE_MODEL = False
_cfg.COMPILE_VAE = False

from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline
from flash_head.utils.matting import RVMMatting
from collections import deque

print('Loading pipeline...')
t0 = time.time()
pipeline = get_pipeline(world_size=1, ckpt_dir='models/SoulX-FlashHead-1_3B', model_type='lite', wav2vec_dir='models/wav2vec2-base-960h')
get_base_data(pipeline, 'examples/girl.png', base_seed=42, use_face_crop=False)
params = get_infer_params()
fn = params['frame_num']
mfn = (params['motion_frames_latent_num'] - 1) * 8 + 1
sl = fn - mfn
chunk_samples = sl * params['sample_rate'] // params['tgt_fps']
print(f'Init: {time.time()-t0:.1f}s (res={params["height"]}x{params["width"]})')

print('Loading RVM...')
matting = RVMMatting(device='cuda')

audio, sr = librosa.load('examples/podcast_sichuan_16k.wav', sr=16000)
total_chunks = len(audio) // chunk_samples
print(f'Audio: {len(audio)/16000:.1f}s, {total_chunks} chunks')

aud_dur = params['cached_audio_duration']
audio_dq = deque(audio[:aud_dur*16000].tolist(), maxlen=aud_dur*16000)
emb = get_audio_embedding(pipeline, np.array(audio_dq), aud_dur*25-fn, aud_dur*25)
_ = run_pipeline(pipeline, emb)
torch.cuda.synchronize()

times = []
for ci in range(min(total_chunks-1, 12)):
    start = ci * chunk_samples
    chunk = audio[start:start+chunk_samples].astype(np.float32)
    audio_dq.extend(chunk.tolist())
    t1 = time.time()
    emb = get_audio_embedding(pipeline, np.array(audio_dq), aud_dur*25-fn, aud_dur*25)
    video = run_pipeline(pipeline, emb); video = video[mfn:]
    pipe_t = time.time() - t1
    frames_np = video.cpu().numpy().astype(np.uint8)
    matting.reset()
    t2 = time.time()
    frames_np = matting.apply(frames_np)
    mat_t = time.time() - t2
    total = pipe_t + mat_t
    times.append(total)
    print(f'  chunk {ci+1}: pipe={pipe_t*1000:.0f}ms mat={mat_t*1000:.0f}ms total={total*1000:.0f}ms')

avg = sum(times)/len(times)
print(f'Average: {avg*1000:.0f}ms/chunk (need <960ms for real-time)')
print(f'REAL-TIME: {"YES" if avg < 0.96 else "NO"}')
