import torch
import numpy as np
from loguru import logger


class RVMMatting:
    def __init__(self, device="cuda", downsample_ratio=0.5, dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.downsample_ratio = downsample_ratio
        self.model = torch.hub.load(
            "PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True
        )
        self.model = self.model.to(device=device, dtype=dtype).eval()
        self.reset()
        logger.info(f"RVM matting loaded (device={device}, dtype={dtype}, ds={downsample_ratio})")

    def reset(self):
        self._rec = [None, None, None, None]

    @torch.no_grad()
    def apply(self, frames: np.ndarray) -> np.ndarray:
        """抠图 + alpha 关键帧插值：每 KEY_INTERVAL 帧跑一次 RVM，中间帧 alpha 线性过渡"""
        src_tensor = (
            torch.from_numpy(frames)
            .to(device=self.device, dtype=self.dtype)
            .permute(0, 3, 1, 2)
            .div_(255.0)
        )
        T = frames.shape[0]
        KEY_INTERVAL = 5  # 每 5 帧一次完整抠图

        # 收集关键帧的 alpha (pha)
        alpha_keys = {}  # frame_index -> pha_tensor
        for idx in range(0, T, KEY_INTERVAL):
            fgr, pha, *self._rec = self.model(
                src_tensor[idx:idx + 1], *self._rec, self.downsample_ratio
            )
            alpha_keys[idx] = pha  # (1, 1, H, W)

        # 确保最后一帧是关键帧
        if (T - 1) not in alpha_keys:
            fgr, pha, *self._rec = self.model(
                src_tensor[T - 1:T], *self._rec, self.downsample_ratio
            )
            alpha_keys[T - 1] = pha

        # 为每帧构造 RGBA：RGB 用原始帧，alpha 用关键帧插值
        all_rgba = torch.cat([src_tensor, torch.zeros_like(src_tensor[:, :1])], dim=1)  # (T, 4, H, W)
        key_idxs = sorted(alpha_keys.keys())
        for i in range(T):
            if i in alpha_keys:
                all_rgba[i, 3:4] = alpha_keys[i]
            else:
                # 找前后关键帧，线性插值 alpha
                prev = max(k for k in key_idxs if k < i)
                nxt = min(k for k in key_idxs if k > i)
                w = (i - prev) / (nxt - prev)
                all_rgba[i, 3:4] = (1 - w) * alpha_keys[prev] + w * alpha_keys[nxt]

        all_rgba = all_rgba.float().clamp_(0, 1).mul_(255).byte()
        all_rgba = all_rgba.permute(0, 2, 3, 1)
        return all_rgba.cpu().numpy()
