import torch
import numpy as np
from loguru import logger
import os
import sys
import urllib.request


def _get_modnet_cache_dir():
    """Return path to cached MODNet source, downloading if needed."""
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "torch", "hub", "ZHKKKe_MODNet_master"
    )
    if not os.path.isdir(cache_dir):
        torch.hub.load("ZHKKKe/MODNet", "mobilenetv2", trust_repo=True)
    # torch.hub.load will have downloaded the repo even if hubconf is missing
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "torch", "hub", "ZHKKKe_MODNet_master"
    )
    return cache_dir


_MODNET_SRC = None


def _import_modnet():
    """Import MODNet model from cached source (no hubconf.py needed)."""
    global _MODNET_SRC
    if _MODNET_SRC is not None:
        return _MODNET_SRC
    cache_dir = _get_modnet_cache_dir()
    src_dir = os.path.join(cache_dir, "src", "models")
    if src_dir not in sys.path:
        sys.path.insert(0, os.path.join(cache_dir, "src"))
    from models.modnet import MODNet
    from models.backbones.wrapper import MobileNetV2Backbone
    _MODNET_SRC = (MODNet, MobileNetV2Backbone)
    return _MODNET_SRC


_MODNET_WEIGHT_URL = (
    "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/"
    "models/modnet_photographic_portrait_matting.ckpt"
)


def _download_modnet_weights():
    """Download MODNet weights to torch hub checkpoints directory.
    Requires HTTPS_PROXY or https_proxy environment variable to be set
    if behind a firewall."""
    hub_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints"
    )
    os.makedirs(hub_dir, exist_ok=True)
    dst = os.path.join(hub_dir, "modnet_mobilenetv2.pth")

    if os.path.isfile(dst):
        return dst

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""
    if proxy:
        handler = urllib.request.ProxyHandler({"https": proxy, "http": proxy})
        opener = urllib.request.build_opener(handler)
        urllib.request.install_opener(opener)

    logger.info(
        f"Downloading MODNet weights from HuggingFace..."
        f"{' (proxy: ' + proxy + ')' if proxy else ''}"
    )
    torch.hub.download_url_to_file(_MODNET_WEIGHT_URL, dst, progress=True)
    return dst


class RVMMatting:
    def __init__(self, device="cuda", downsample_ratio=0.5, dtype=torch.float16,
                 key_interval=5):
        self.device = device
        self.dtype = dtype
        self.downsample_ratio = downsample_ratio
        self.key_interval = key_interval
        self.model = torch.hub.load(
            "PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True
        )
        self.model = self.model.to(device=device, dtype=dtype).eval()
        self.reset()
        logger.info(
            f"RVM matting loaded (device={device}, dtype={dtype}, "
            f"ds={downsample_ratio}, key_int={key_interval})"
        )

    def reset(self):
        self._rec = [None, None, None, None]

    @torch.no_grad()
    def apply(self, frames: np.ndarray) -> np.ndarray:
        src_tensor = (
            torch.from_numpy(frames)
            .to(device=self.device, dtype=self.dtype)
            .permute(0, 3, 1, 2)
            .div_(255.0)
        )
        T = frames.shape[0]
        KI = self.key_interval

        key_indices = list(range(0, T, KI))
        if key_indices[-1] != T - 1:
            key_indices.append(T - 1)

        key_src = src_tensor[key_indices]  # (K, 3, H, W)

        fgr, pha, *self._rec = self.model(
            key_src, *self._rec, self.downsample_ratio
        )
        pha = pha.float()  # (K, 1, H, W)

        alpha_keys = {}
        for i, idx in enumerate(key_indices):
            alpha_keys[idx] = pha[i:i + 1]

        all_rgba = torch.cat(
            [src_tensor, torch.zeros_like(src_tensor[:, :1])], dim=1
        )  # (T, 4, H, W)
        key_idxs = sorted(alpha_keys.keys())
        for i in range(T):
            if i in alpha_keys:
                all_rgba[i, 3:4] = alpha_keys[i]
            else:
                prev = max(k for k in key_idxs if k < i)
                nxt = min(k for k in key_idxs if k > i)
                w = (i - prev) / (nxt - prev)
                all_rgba[i, 3:4] = (
                    (1 - w) * alpha_keys[prev] + w * alpha_keys[nxt]
                )

        all_rgba = all_rgba.float().clamp_(0, 1).mul_(255).byte()
        all_rgba = all_rgba.permute(0, 2, 3, 1)
        return all_rgba.cpu().numpy()


class MODNetMatting:
    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype

        MODNet, _ = _import_modnet()
        self.model = MODNet(backbone_arch='mobilenetv2', backbone_pretrained=False)
        self.model = self.model.to(device=device, dtype=dtype).eval()

        ckpt_path = _download_modnet_weights()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)

        logger.info(
            f"MODNet matting loaded (device={device}, dtype={dtype})"
        )

    def reset(self):
        pass

    @torch.no_grad()
    def apply(self, frames: np.ndarray) -> np.ndarray:
        src_tensor = (
            torch.from_numpy(frames)
            .to(device=self.device, dtype=self.dtype)
            .permute(0, 3, 1, 2)
            .div_(255.0)
        )  # (T, 3, H, W)
        T = src_tensor.shape[0]

        # MODNet can batch — pass all frames at once, then split alpha
        _, _, all_pha = self.model(src_tensor, inference=True)
        all_pha = all_pha.float()  # (T, 1, H, W)

        all_rgba = torch.cat([src_tensor, all_pha], dim=1)  # (T, 4, H, W)
        all_rgba = all_rgba.float().clamp_(0, 1).mul_(255).byte()
        all_rgba = all_rgba.permute(0, 2, 3, 1)
        return all_rgba.cpu().numpy()
