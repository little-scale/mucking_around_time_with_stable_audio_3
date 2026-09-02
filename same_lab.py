"""Backend-neutral SAME latent editing for the integrated browser lab.

The accelerator-specific encode/decode calls live on the selected monitor
backend.  Everything in this module operates on NumPy arrays shaped
``[256, latent_frames]`` so the HTTP and browser contract is identical on MLX
and CUDA.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


SAMPLE_RATE = 44_100
SAMPLES_PER_LATENT = 4096
LATENT_FRAMES_PER_SECOND = SAMPLE_RATE / SAMPLES_PER_LATENT
PROFILE_CHANNELS = 256
PROFILE_MIN = -3.0
PROFILE_MAX = 3.0
OPERATIONS = (
    "Zero / ablate selected",
    "Scale selected",
    "Add offset (channel sigma)",
    "Add Gaussian noise (channel sigma)",
    "Invert sign",
    "Replace with channel mean",
    "Freeze in time",
    "Shuffle selected time",
    "Keep only selected (ablate all others)",
    "Frame displacement (circular)",
    "LFO - Additive offset",
    "LFO - Replace around channel mean",
    "LFO - Multiplicative",
)
WAVEFORMS = ("Sine", "Triangle", "Square", "Sawtooth")


@dataclass
class LatentSession:
    session_id: str
    directory: Path
    client_ip: str
    codec: str
    original: np.ndarray
    current: np.ndarray
    baseline_audio: np.ndarray
    target_samples: int
    source_a_name: str
    source_b: np.ndarray | None = None
    source_b_audio: np.ndarray | None = None
    source_b_name: str | None = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def duration(self) -> float:
        return self.target_samples / SAMPLE_RATE


def new_session_id() -> str:
    return uuid.uuid4().hex[:16]


def parse_channels(spec: str, n_channels: int = PROFILE_CHANNELS) -> list[int]:
    text = (spec or "").strip().lower()
    if not text:
        raise ValueError("Enter channels such as 12, 3-8, or all.")
    if text in {"all", "*"}:
        return list(range(n_channels))
    channels: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        try:
            if "-" in part:
                start, end = (int(value) for value in part.split("-", 1))
                start, end = sorted((start, end))
                channels.update(range(start, end + 1))
            else:
                channels.add(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid channel selection: {part}") from exc
    bad = [channel for channel in channels if not 0 <= channel < n_channels]
    if bad:
        raise ValueError(f"Channels must be between 0 and {n_channels - 1}: {bad[:10]}")
    if not channels:
        raise ValueError("No channels selected.")
    return sorted(channels)


def seconds_to_frames(start_seconds: float, end_seconds: float, frames: int, duration: float) -> tuple[int, int]:
    start_seconds = min(max(float(start_seconds), 0.0), duration)
    end_seconds = min(max(float(end_seconds), 0.0), duration)
    if end_seconds <= start_seconds:
        raise ValueError("End time must be greater than start time.")
    hop = SAMPLES_PER_LATENT / SAMPLE_RATE
    start = min(frames - 1, int(math.floor(start_seconds / hop)))
    end = min(frames, max(start + 1, int(math.ceil(end_seconds / hop))))
    return start, end


def make_lfo(frame_count: int, period: float, waveform: str, phase_degrees: float) -> np.ndarray:
    if frame_count < 1:
        raise ValueError("The LFO region must contain at least one latent frame.")
    if not math.isfinite(float(period)) or float(period) <= 0:
        raise ValueError("LFO period must be greater than zero latent frames.")
    phase = np.arange(frame_count, dtype=np.float32) / np.float32(period)
    phase += np.float32(float(phase_degrees) / 360.0)
    sine = np.sin(2 * np.pi * phase).astype(np.float32, copy=False)
    if waveform == "Sine":
        return sine
    if waveform == "Triangle":
        return (2 / np.pi * np.arcsin(sine)).astype(np.float32, copy=False)
    if waveform == "Square":
        return np.where(sine >= 0, 1.0, -1.0).astype(np.float32, copy=False)
    if waveform == "Sawtooth":
        return (2 * np.mod(phase, 1.0) - 1.0).astype(np.float32, copy=False)
    raise ValueError(f"Unknown LFO waveform: {waveform}")


def intervene(
    source: np.ndarray,
    channels: list[int],
    start: int,
    end: int,
    operation: str,
    amount: float = 0.0,
    seed: int = 0,
    lfo_period: float = 16.0,
    lfo_depth: float = 1.0,
    lfo_waveform: str = "Sine",
    lfo_phase: float = 0.0,
    frame_displacement: int = 1,
    lfo_invert: bool = False,
) -> np.ndarray:
    result = np.asarray(source, dtype=np.float32).copy()
    indices = np.asarray(channels, dtype=np.int64)
    region = slice(start, end)
    rng = np.random.default_rng(int(seed))
    if operation == OPERATIONS[0]:
        result[indices, region] = 0
    elif operation == OPERATIONS[1]:
        result[indices, region] *= float(amount)
    elif operation == OPERATIONS[2]:
        sigma = source[indices].std(axis=1, keepdims=True) + 1e-8
        result[indices, region] += float(amount) * sigma
    elif operation == OPERATIONS[3]:
        sigma = source[indices].std(axis=1, keepdims=True) + 1e-8
        noise = rng.standard_normal((len(indices), end - start), dtype=np.float32)
        result[indices, region] += noise * sigma * float(amount)
    elif operation == OPERATIONS[4]:
        result[indices, region] *= -1
    elif operation == OPERATIONS[5]:
        result[indices, region] = source[indices].mean(axis=1, keepdims=True)
    elif operation == OPERATIONS[6]:
        result[indices, region] = source[indices, start:start + 1]
    elif operation == OPERATIONS[7]:
        for channel in indices:
            rng.shuffle(result[channel, start:end])
    elif operation == OPERATIONS[8]:
        others = np.setdiff1d(np.arange(result.shape[0]), indices)
        result[others, region] = 0
    elif operation == OPERATIONS[9]:
        result[indices, region] = np.roll(source[indices, region], shift=int(frame_displacement), axis=1)
    elif operation.startswith("LFO - "):
        lfo = make_lfo(end - start, lfo_period, lfo_waveform, lfo_phase)[None, :]
        if lfo_invert:
            lfo = -lfo
        if operation == OPERATIONS[10]:
            result[indices, region] += float(lfo_depth) * lfo
        elif operation == OPERATIONS[11]:
            centre = source[indices].mean(axis=1, keepdims=True)
            result[indices, region] = centre + float(lfo_depth) * lfo
        elif operation == OPERATIONS[12]:
            result[indices, region] *= 1.0 + float(lfo_depth) * lfo
        else:
            raise ValueError(f"Unknown LFO operation: {operation}")
    else:
        raise ValueError(f"Unknown operation: {operation}")
    return result


def validate_bank(values, minimum: float, maximum: float, label: str) -> np.ndarray:
    bank = np.asarray(values, dtype=np.float32).reshape(-1)
    if bank.size != PROFILE_CHANNELS:
        raise ValueError(f"{label} must contain exactly {PROFILE_CHANNELS} values.")
    if not np.isfinite(bank).all():
        raise ValueError(f"{label} contains a non-finite value.")
    return np.clip(bank, minimum, maximum)


def mix_channels(source_a: np.ndarray, source_b: np.ndarray, amounts) -> np.ndarray:
    a = np.asarray(source_a, dtype=np.float32)
    b = np.asarray(source_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Latent shapes must match, got {a.shape} and {b.shape}.")
    mix = validate_bank(amounts, 0.0, 1.0, "Latent mix")
    return a * (1.0 - mix[:, None]) + b * mix[:, None]


def time_crossfade(
    source_a: np.ndarray,
    source_b: np.ndarray,
    start: int,
    end: int,
    direction: str = "A to B",
    curve: str = "Smoothstep",
) -> np.ndarray:
    a = np.asarray(source_a, dtype=np.float32)
    b = np.asarray(source_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Latent shapes must match, got {a.shape} and {b.shape}.")
    width = end - start
    ramp = np.array([0.5], dtype=np.float32) if width == 1 else np.linspace(0, 1, width, dtype=np.float32)
    if curve == "Smoothstep":
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    elif curve != "Linear":
        raise ValueError(f"Unknown crossfade curve: {curve}")
    mix = np.zeros(a.shape[1], dtype=np.float32)
    mix[start:end] = ramp
    mix[end:] = 1.0
    if direction == "B to A":
        mix = 1.0 - mix
    elif direction != "A to B":
        raise ValueError(f"Unknown crossfade direction: {direction}")
    return a * (1.0 - mix[None, :]) + b * mix[None, :]


def apply_profile(source: np.ndarray, values, start: int, end: int, mode: str) -> np.ndarray:
    profile = validate_bank(values, PROFILE_MIN, PROFILE_MAX, "Channel profile")
    result = np.asarray(source, dtype=np.float32).copy()
    if mode == "Offset":
        result[:, start:end] += profile[:, None]
    elif mode == "Multiply":
        result[:, start:end] *= 1.0 + profile[:, None]
    else:
        raise ValueError(f"Unknown profile mode: {mode}")
    return result


def channel_stats(latent: np.ndarray, top_k: int = 32) -> list[dict]:
    value = np.asarray(latent, dtype=np.float32)
    mean = value.mean(axis=1)
    std = value.std(axis=1)
    rms = np.sqrt(np.mean(value * value, axis=1))
    movement = np.mean(np.abs(np.diff(value, axis=1)), axis=1) if value.shape[1] > 1 else np.zeros(value.shape[0])
    order = np.argsort(std)[::-1][:top_k]
    return [
        {"channel": int(c), "mean": float(mean[c]), "std": float(std[c]), "rms": float(rms[c]), "movement": float(movement[c])}
        for c in order
    ]


def difference_audio(audio: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    difference = np.asarray(audio, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)
    peak = float(np.max(np.abs(difference))) if difference.size else 0.0
    return difference / peak * 0.95 if peak > 1e-8 else difference

