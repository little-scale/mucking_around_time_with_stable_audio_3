from __future__ import annotations

import importlib.util
import math
import platform
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np

EventFn = Callable[[dict], None]
CONTRACT_VERSION = "1.0"
SAMPLE_RATE = 44_100


class BackendUnavailable(RuntimeError):
    """A requested backend cannot run in the current environment."""


@dataclass(frozen=True)
class BackendInfo:
    backend: str
    device: str
    device_name: str
    dtype: str
    vram_total_bytes: int = 0
    vram_free_bytes: int = 0
    framework: str = ""
    framework_version: str = ""
    contract_version: str = CONTRACT_VERSION

    def as_dict(self) -> dict:
        return asdict(self)


class MonitorBackend(ABC):
    name: str
    dit_choices: tuple[str, ...]
    decoder_choices: tuple[str, ...]
    max_seconds: float = 380.0
    sample_rate: int = SAMPLE_RATE

    @abstractmethod
    def diagnostics(self) -> BackendInfo: ...

    @abstractmethod
    def generate(self, *, output_path: str, emit: EventFn, **config) -> dict: ...

    @property
    def same_codec_choices(self) -> tuple[str, ...]:
        return tuple(self.decoder_choices)

    def same_encode(self, input_path: str, codec_name: str) -> dict:
        """Encode host-local audio into a [256, frames] NumPy latent."""
        raise BackendUnavailable(f"The {self.name} backend does not expose SAME encoding")

    def same_decode(self, latent: np.ndarray, codec_name: str, target_samples: int) -> np.ndarray:
        """Decode a [256, frames] NumPy latent into [2, samples] float audio."""
        raise BackendUnavailable(f"The {self.name} backend does not expose SAME decoding")


def event(kind: str, **payload) -> dict:
    return {"type": kind, "time": time.time(), "contract_version": CONTRACT_VERSION, **payload}


def array_stats(value) -> dict:
    x = np.asarray(value, dtype=np.float32)
    if x.size == 0:
        return {"shape": list(x.shape), "mean": 0.0, "std": 0.0, "rms": 0.0, "min": 0.0, "max": 0.0}
    return {
        "shape": list(x.shape),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def heatmap_preview(value, rows: int = 64, cols: int = 96) -> dict:
    a = np.asarray(value, dtype=np.float32)
    while a.ndim > 2:
        a = a[0]
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a[None, :]
    r_idx = np.linspace(0, max(a.shape[0] - 1, 0), min(rows, a.shape[0]), dtype=int)
    c_idx = np.linspace(0, max(a.shape[1] - 1, 0), min(cols, a.shape[1]), dtype=int)
    view = a[np.ix_(r_idx, c_idx)]
    if not view.size:
        view = np.zeros((1, 1), dtype=np.float32)
    finite = np.nan_to_num(view, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(finite, [2.0, 98.0])
    q = np.zeros_like(finite, dtype=np.uint8) if abs(hi - lo) < 1e-8 else np.rint(np.clip((finite - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return {"w": int(q.shape[1]), "h": int(q.shape[0]), "data": q.reshape(-1).tolist(), "lo": float(lo), "hi": float(hi)}


def waveform_preview(audio, points: int = 1200) -> dict:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim > 2:
        a = a.reshape((-1, a.shape[-1]))
    if a.shape[0] > 2 and a.shape[1] <= 2:
        a = a.T
    a = a[:2]
    n = a.shape[-1]
    if not n:
        return {"points": [], "channels": int(a.shape[0]), "samples": 0}
    edges = np.linspace(0, n, min(points, n) + 1, dtype=int)
    packed = []
    for channel in a:
        packed.append([[float(channel[edges[i]:edges[i + 1]].min()), float(channel[edges[i]:edges[i + 1]].max())] for i in range(len(edges) - 1)])
    return {"points": packed, "channels": int(a.shape[0]), "samples": int(n)}


def auto_backend_name() -> str:
    """Conservative detection: usable CUDA first, then Apple Silicon MLX."""
    if platform.system() == "Linux" and importlib.util.find_spec("torch") is not None:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    if platform.system() == "Darwin" and platform.machine() == "arm64" and importlib.util.find_spec("mlx") is not None:
        return "mlx"
    raise BackendUnavailable(
        "Could not auto-detect a usable backend. Use --backend cuda on Linux/NVIDIA or "
        "--backend mlx on an Apple-silicon Mac; run ./sa3-monitor --diagnose for details."
    )
