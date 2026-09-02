from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

from backends.base import BackendUnavailable, auto_backend_name


@dataclass
class Probe:
    name: str
    available: bool
    reason: str


def probe_backends(sa3_root: str | None = None) -> list[Probe]:
    probes: list[Probe] = []
    if platform.system() != "Linux":
        probes.append(Probe("cuda", False, "CUDA mode is supported on Linux/NVIDIA hosts"))
    elif importlib.util.find_spec("torch") is None:
        probes.append(Probe("cuda", False, "PyTorch is not installed in this Python environment"))
    else:
        try:
            import torch
            ok = bool(torch.cuda.is_available())
            reason = f"{torch.cuda.get_device_name(0)}; torch {torch.__version__}; CUDA {torch.version.cuda}" if ok else "torch.cuda.is_available() is false"
            if ok and importlib.util.find_spec("stable_audio_3") is None:
                ok, reason = False, "CUDA works, but the official stable_audio_3 package is not installed"
            probes.append(Probe("cuda", ok, reason))
        except Exception as exc:
            probes.append(Probe("cuda", False, f"PyTorch probe failed: {type(exc).__name__}: {exc}"))

    root = Path(sa3_root or os.environ.get("SA3_MLX_ROOT") or os.environ.get("SA3_ROOT") or ".").expanduser().resolve()
    mlx_root = root / "optimized" / "mlx" if (root / "optimized" / "mlx").exists() else root
    mlx_source = mlx_root / "scripts" / "sa3_mlx.py"
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        probes.append(Probe("mlx", False, "MLX mode requires an Apple-silicon Mac"))
    elif importlib.util.find_spec("mlx") is None:
        probes.append(Probe("mlx", False, "The mlx package is not installed in this Python environment"))
    elif not mlx_source.exists():
        probes.append(Probe("mlx", False, f"SA3 MLX source not found at {mlx_source}; set --sa3-root"))
    else:
        probes.append(Probe("mlx", True, f"Apple Silicon + MLX; source {mlx_root}"))
    return probes


def create_backend(name: str, sa3_root: str | None = None):
    if sa3_root:
        os.environ["SA3_ROOT"] = str(Path(sa3_root).expanduser().resolve())
    selected = auto_backend_name() if name == "auto" else name
    if selected == "cuda":
        try:
            from backends.cuda_engine import CUDAMonitorEngine
            return CUDAMonitorEngine()
        except BackendUnavailable:
            raise
        except Exception as exc:
            raise BackendUnavailable(f"CUDA backend startup failed: {type(exc).__name__}: {exc}") from exc
    if selected == "mlx":
        try:
            from backends.mlx_engine import SA3MonitorEngine
            return SA3MonitorEngine()
        except BackendUnavailable:
            raise
        except Exception as exc:
            raise BackendUnavailable(
                "MLX backend startup failed. Use an Apple-silicon MLX environment and set "
                f"--sa3-root to the official checkout. Original error: {type(exc).__name__}: {exc}"
            ) from exc
    raise BackendUnavailable(f"Unknown backend {selected!r}; expected auto, cuda or mlx")


def diagnostics_json(sa3_root: str | None = None) -> list[dict]:
    return [asdict(item) for item in probe_backends(sa3_root)]
