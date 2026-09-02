from __future__ import annotations

import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import mlx.core as mx

# This folder is intended to live at optimized/mlx/monitor/.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
_configured_root = Path(os.environ.get("SA3_MLX_ROOT") or os.environ.get("SA3_ROOT") or PROJECT.parent).expanduser().resolve()
REPO = _configured_root / "optimized" / "mlx" if (_configured_root / "optimized" / "mlx" / "scripts" / "sa3_mlx.py").exists() else _configured_root
SCRIPTS = REPO / "scripts"
if not (SCRIPTS / "sa3_mlx.py").exists():
    raise RuntimeError(
        f"SA3 MLX source not found under {REPO}. Set SA3_ROOT to the stable-audio-3 checkout "
        "or SA3_MLX_ROOT to its optimized/mlx directory."
    )
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sa3_mlx import (  # noqa: E402
    DIT_CHOICES,
    DECODER_CHOICES,
    SAMPLE_RATE,
    SAMPLES_PER_LATENT,
    T5GEMMA_NPZ_REL,
    load_dit,
    load_decoder,
    load_encoder,
    patch_audio,
    read_wav,
    save_wav,
)
from models.defs.sa3_pipeline import (  # noqa: E402
    apply_prompt_padding,
    build_pingpong_schedule,
    load_conditioner_from_npz,
    patched_decode,
)
from models.defs.t5gemma_mlx import T5Gemma  # noqa: E402
from weights import ensure_local  # noqa: E402
from backends.base import BackendInfo, CONTRACT_VERSION  # noqa: E402

EventFn = Callable[[dict], None]


def _reset_memory_peak() -> None:
    fn = getattr(mx, "reset_peak_memory", None)
    if fn is None:
        fn = getattr(getattr(mx, "metal", None), "reset_peak_memory", None)
    try:
        if fn:
            fn()
    except Exception:
        pass


def _memory_bytes() -> int:
    fn = getattr(mx, "get_peak_memory", None)
    if fn is None:
        fn = getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    try:
        return int(fn()) if fn else 0
    except Exception:
        return 0


def _stats_np(a: np.ndarray) -> dict:
    x = np.asarray(a, dtype=np.float32)
    if x.size == 0:
        return {"shape": list(x.shape), "mean": 0, "std": 0, "rms": 0, "min": 0, "max": 0}
    return {
        "shape": list(x.shape),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "rms": float(np.sqrt(np.mean(x * x))),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def tensor_stats(x) -> dict:
    if hasattr(x, "astype"):
        try:
            mx.eval(x)
            a = np.array(x.astype(mx.float32))
            return _stats_np(a)
        except Exception:
            pass
    return _stats_np(np.asarray(x))


def heatmap_preview(x, rows: int = 64, cols: int = 96) -> dict:
    """Return a compact uint8 2-D preview suitable for a browser canvas."""
    if hasattr(x, "astype"):
        mx.eval(x)
        a = np.array(x.astype(mx.float32))
    else:
        a = np.asarray(x, dtype=np.float32)

    while a.ndim > 2:
        a = a[0]
    if a.ndim == 1:
        a = a[None, :]

    r_idx = np.linspace(0, max(a.shape[0] - 1, 0), min(rows, a.shape[0]), dtype=int)
    c_idx = np.linspace(0, max(a.shape[1] - 1, 0), min(cols, a.shape[1]), dtype=int)
    view = a[np.ix_(r_idx, c_idx)]

    if view.size:
        lo, hi = np.percentile(view, [2.0, 98.0])
        if not np.isfinite(lo):
            lo = float(np.nanmin(view))
        if not np.isfinite(hi):
            hi = float(np.nanmax(view))
        if abs(hi - lo) < 1e-8:
            q = np.zeros_like(view, dtype=np.uint8)
        else:
            q = np.clip((view - lo) / (hi - lo), 0, 1)
            q = np.rint(q * 255.0).astype(np.uint8)
    else:
        lo, hi = 0.0, 1.0
        q = np.zeros((1, 1), dtype=np.uint8)

    return {
        "w": int(q.shape[1]),
        "h": int(q.shape[0]),
        "data": q.reshape(-1).tolist(),
        "lo": float(lo),
        "hi": float(hi),
    }


def waveform_preview(audio: np.ndarray, points: int = 1200) -> dict:
    """Min/max envelope for stereo audio. Browser draws this without shipping the full PCM."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[0] > 2 and a.shape[1] <= 2:
        a = a.T
    if a.shape[0] > 2:
        a = a[:2]

    n = a.shape[-1]
    if n == 0:
        return {"points": [], "channels": int(a.shape[0]), "samples": 0}
    bins = min(points, n)
    edges = np.linspace(0, n, bins + 1, dtype=int)
    packed = []
    for ch in range(a.shape[0]):
        out = []
        for i in range(bins):
            seg = a[ch, edges[i]:edges[i + 1]]
            if seg.size:
                out.append([float(seg.min()), float(seg.max())])
            else:
                out.append([0.0, 0.0])
        packed.append(out)
    return {"points": packed, "channels": int(a.shape[0]), "samples": int(n)}


class SA3MonitorEngine:
    """Long-lived SA3 MLX engine with model caches and inspectable sampling."""

    def __init__(self):
        self._t5: Optional[T5Gemma] = None
        self._conditioners: dict[str, tuple] = {}
        self._dits: dict[tuple[str, int], object] = {}
        self._dit_lru: list[tuple[str, int]] = []
        self._decoders: dict[str, tuple] = {}
        self._encoders: dict[str, tuple] = {}
        self._dit_cache_max = 2

    def _emit(self, emit: EventFn, kind: str, **data):
        emit({"type": kind, "time": time.time(), "contract_version": CONTRACT_VERSION, "memory_bytes": _memory_bytes(), **data})

    name = "mlx"
    dit_choices = tuple(DIT_CHOICES.keys())
    decoder_choices = tuple(DECODER_CHOICES.keys())
    max_seconds = 380.0
    sample_rate = SAMPLE_RATE

    def diagnostics(self) -> BackendInfo:
        try:
            framework_version = getattr(mx, "__version__", "unknown")
        except Exception:
            framework_version = "unknown"
        return BackendInfo(
            backend="mlx",
            device="gpu",
            device_name=f"Apple Silicon ({platform.machine()})",
            dtype="float16 (DiT) / float32 (SAME)",
            framework="MLX",
            framework_version=framework_version,
        )

    def _get_t5(self, emit: EventFn):
        if self._t5 is None:
            self._emit(emit, "stage", stage="t5", state="loading", label="Loading T5Gemma")
            t0 = time.perf_counter()
            self._t5 = T5Gemma.from_npz(str(ensure_local(T5GEMMA_NPZ_REL)))
            self._emit(emit, "model_load", model="T5Gemma", ms=(time.perf_counter() - t0) * 1000)
        return self._t5

    def _get_conditioner(self, dit_name: str):
        if dit_name not in self._conditioners:
            self._conditioners[dit_name] = load_conditioner_from_npz(
                str(ensure_local(DIT_CHOICES[dit_name]["ckpt"])), prefix="cond."
            )
        return self._conditioners[dit_name]

    def _get_dit(self, dit_name: str, t_lat: int, dtype, steps: int, emit: EventFn):
        key = (dit_name, t_lat)
        if key in self._dits:
            if key in self._dit_lru:
                self._dit_lru.remove(key)
            self._dit_lru.append(key)
            self._emit(emit, "model_load", model=dit_name, ms=0.0, cached=True)
            return self._dits[key]

        while len(self._dits) >= self._dit_cache_max:
            old = self._dit_lru.pop(0)
            self._dits.pop(old, None)
            self._emit(emit, "log", message=f"Evicted cached DiT {old[0]} at latent length {old[1]}")

        t0 = time.perf_counter()
        model, _ = load_dit(dit_name, T_lat=t_lat, dtype=dtype)
        self._dits[key] = model
        self._dit_lru.append(key)
        self._emit(emit, "model_load", model=dit_name, ms=(time.perf_counter() - t0) * 1000, cached=False)
        return model

    def _get_decoder(self, decoder_name: str, emit: EventFn):
        if decoder_name not in self._decoders:
            t0 = time.perf_counter()
            self._decoders[decoder_name] = load_decoder(decoder_name, mx.float32)
            self._emit(emit, "model_load", model=f"{decoder_name} decoder", ms=(time.perf_counter() - t0) * 1000)
        return self._decoders[decoder_name]

    def _get_encoder(self, decoder_name: str, emit: EventFn):
        if decoder_name not in self._encoders:
            t0 = time.perf_counter()
            self._encoders[decoder_name] = load_encoder(decoder_name, mx.float32)
            self._emit(emit, "model_load", model=f"{decoder_name} encoder", ms=(time.perf_counter() - t0) * 1000)
        return self._encoders[decoder_name]

    @property
    def same_codec_choices(self) -> tuple[str, ...]:
        return ("same-s", "same-l")

    def _read_same_audio(self, input_path: str) -> np.ndarray:
        try:
            return read_wav(str(input_path))
        except RuntimeError as original_error:
            converter = shutil.which("afconvert")
            if not converter:
                raise RuntimeError(
                    "MLX SAME input must be a readable WAV, or macOS afconvert must be available"
                ) from original_error
            with tempfile.TemporaryDirectory(prefix="sa3_same_") as directory:
                converted = Path(directory) / "converted.wav"
                result = subprocess.run(
                    [converter, "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "2", str(input_path), str(converted)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0 or not converted.exists():
                    detail = (result.stderr or result.stdout).strip()
                    raise RuntimeError(f"Could not convert input audio with afconvert: {detail}") from original_error
                return read_wav(str(converted))

    def same_encode(self, input_path: str, codec_name: str) -> dict:
        if codec_name not in self.same_codec_choices:
            raise ValueError(f"Unknown SAME codec: {codec_name}")
        mx.set_default_device(mx.gpu)
        audio = self._read_same_audio(input_path)
        target_samples = int(audio.shape[-1])
        if target_samples < 1:
            raise ValueError("Input audio is empty")
        frames = math.ceil(target_samples / SAMPLES_PER_LATENT)
        if codec_name == "same-s" and frames % 2:
            frames += 1
        padded_samples = frames * SAMPLES_PER_LATENT
        if padded_samples > target_samples:
            audio = np.pad(audio[:, :target_samples], ((0, 0), (0, padded_samples - target_samples)), mode="wrap")
        encoder, pad_mod = self._get_encoder(codec_name, lambda _event: None)
        patches = patch_audio(audio[None, ...], patch_size=256)
        if patches.shape[-1] % pad_mod:
            raise RuntimeError("Audio padding did not satisfy the selected SAME encoder")
        latent = encoder(mx.array(patches, dtype=mx.float32))
        mx.eval(latent)
        latent_np = np.array(latent, dtype=np.float32)[0]
        reconstruction = self.same_decode(latent_np, codec_name, target_samples)
        return {
            "latent": latent_np,
            "audio": reconstruction,
            "target_samples": target_samples,
            "sample_rate": SAMPLE_RATE,
        }

    def same_decode(self, latent: np.ndarray, codec_name: str, target_samples: int) -> np.ndarray:
        if codec_name not in self.same_codec_choices:
            raise ValueError(f"Unknown SAME codec: {codec_name}")
        decoder, chunk_fn, (chunk, overlap) = self._get_decoder(codec_name, lambda _event: None)
        values = mx.array(np.asarray(latent, dtype=np.float32)[None, ...], dtype=mx.float32)
        kernel = chunk + 2 * overlap
        if values.shape[-1] > kernel:
            patches = chunk_fn(decoder, values, chunk, overlap)
        elif values.shape[-1] % 2 == 0 or codec_name == "same-l":
            patches = decoder(values)
        else:
            even = mx.concatenate([values, values[..., -1:]], axis=-1)
            patches = decoder(even)[..., : values.shape[-1] * 16]
        mx.eval(patches)
        audio = patched_decode(patches, patch_size=256, channels=2)
        mx.eval(audio)
        result = np.array(audio.astype(mx.float32))[0, :, : int(target_samples)]
        if not np.isfinite(result).all():
            raise RuntimeError("SAME decoded non-finite audio; reduce the edit amount")
        return result

    def _sample(self, model_fn, x, sigmas, emit: EventFn, *, seed: int, paste_back=None, before_step=None):
        key = mx.random.key(seed)
        total = int(sigmas.shape[0] - 1)
        self._emit(
            emit,
            "sampler_init",
            total=total,
            sigma=float(sigmas[0]),
            stats=tensor_stats(x),
            heatmap=heatmap_preview(x),
        )
        t_prev = time.perf_counter()
        for i in range(total):
            t_curr = sigmas[i]
            t_next = sigmas[i + 1]
            t_tensor = t_curr * mx.ones((x.shape[0],), dtype=x.dtype)
            if before_step is not None:
                before_step(i)
            v = model_fn(x, t_tensor)
            denoised = x - t_curr.astype(x.dtype) * v
            if i < total - 1 and float(t_next) > 0.0:
                key, sub = mx.random.split(key)
                fresh_noise = mx.random.normal(x.shape, dtype=x.dtype, key=sub)
                x = (1.0 - t_next).astype(x.dtype) * denoised + t_next.astype(x.dtype) * fresh_noise
            else:
                x = denoised
            mx.eval(x)
            now = time.perf_counter()
            self._emit(
                emit,
                "sampler_step",
                step=i + 1,
                total=total,
                sigma=float(t_curr),
                sigma_next=float(t_next),
                ms=(now - t_prev) * 1000,
                stats=tensor_stats(x),
                heatmap=heatmap_preview(x),
            )
            t_prev = now

        if paste_back is not None:
            init_latents, mask = paste_back
            m = mask.astype(x.dtype)
            x = init_latents.astype(x.dtype) * m + x * (1.0 - m)
            mx.eval(x)
            self._emit(emit, "paste_back", stats=tensor_stats(x), heatmap=heatmap_preview(x))
        return x

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        seconds: float = 10.0,
        steps: int = 8,
        seed: Optional[int] = None,
        cfg: float = 1.0,
        apg: float = 1.0,
        sigma_max: float = 1.0,
        dit_name: str = "sm-music",
        decoder_name: Optional[str] = None,
        mode: str = "text",
        input_audio: Optional[str] = None,
        inpaint_start: float = 0.0,
        inpaint_end: float = 0.0,
        output_path: str,
        emit: EventFn,
    ) -> dict:
        if dit_name not in DIT_CHOICES:
            raise ValueError(f"Unknown DiT: {dit_name}")
        if decoder_name in (None, "auto"):
            decoder_name = DIT_CHOICES[dit_name]["default_decoder"]
        if decoder_name not in DECODER_CHOICES:
            raise ValueError(f"Unknown decoder: {decoder_name}")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if sigma_max < 0.01:
            raise ValueError("sigma_max must be >= 0.01")
        if seconds <= 0:
            raise ValueError("seconds must be > 0")
        if mode not in {"text", "audio", "inpaint"}:
            raise ValueError("mode must be text, audio or inpaint")
        if mode in {"audio", "inpaint"} and not input_audio:
            raise ValueError(f"{mode} mode requires an input audio file")
        if mode == "inpaint" and not (0 <= inpaint_start < inpaint_end <= seconds):
            raise ValueError("inpaint range must satisfy 0 <= start < end <= seconds")

        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)
        dtype = mx.float16
        mx.set_default_device(mx.gpu)
        _reset_memory_peak()

        t_lat = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
        timing = {}
        started = time.perf_counter()
        self._emit(
            emit,
            "run_start",
            prompt=prompt,
            negative_prompt=negative_prompt,
            seconds=seconds,
            steps=steps,
            seed=seed,
            cfg=cfg,
            apg=apg,
            sigma_max=sigma_max,
            dit=dit_name,
            decoder=decoder_name,
            mode=mode,
            t_lat=t_lat,
            sample_rate=SAMPLE_RATE,
            latent_channels=256,
        )

        # 1. Text encoder
        self._emit(emit, "stage", stage="t5", state="active", label="T5Gemma text encoding")
        t0 = time.perf_counter()
        text_encoder = self._get_t5(emit)
        embeds, mask = text_encoder.encode([prompt], max_len=256)
        mx.eval(embeds, mask)
        timing["t5_ms"] = (time.perf_counter() - t0) * 1000
        self._emit(
            emit,
            "embedding",
            role="prompt",
            ms=timing["t5_ms"],
            tokens=int(np.array(mask).sum()),
            stats=tensor_stats(embeds),
            heatmap=heatmap_preview(embeds, rows=48, cols=96),
        )
        self._emit(emit, "stage", stage="t5", state="done", ms=timing["t5_ms"])

        # 2. Conditioning
        self._emit(emit, "stage", stage="conditioning", state="active", label="Prompt + duration conditioning")
        t0 = time.perf_counter()
        padding_emb, secs_embedder = self._get_conditioner(dit_name)
        embeds16 = embeds.astype(dtype)
        padded = apply_prompt_padding(embeds16, mask, padding_emb.astype(dtype))
        seconds_embed = secs_embedder(seconds).astype(dtype)
        cross_attn = mx.concatenate([padded, seconds_embed], axis=1)
        global_cond = seconds_embed[:, 0, :]
        null_cross_attn = None
        if cfg != 1.0:
            if negative_prompt.strip():
                neg_embeds, neg_mask = text_encoder.encode([negative_prompt.strip()], max_len=256)
                mx.eval(neg_embeds, neg_mask)
                neg_padded = apply_prompt_padding(neg_embeds.astype(dtype), neg_mask, padding_emb.astype(dtype))
                null_cross_attn = mx.concatenate([neg_padded, seconds_embed], axis=1)
                self._emit(
                    emit,
                    "embedding",
                    role="negative",
                    tokens=int(np.array(neg_mask).sum()),
                    stats=tensor_stats(neg_embeds),
                    heatmap=heatmap_preview(neg_embeds, rows=48, cols=96),
                )
            else:
                null_cross_attn = mx.zeros_like(cross_attn)
            mx.eval(null_cross_attn)
        mx.eval(cross_attn, global_cond)
        timing["conditioning_ms"] = (time.perf_counter() - t0) * 1000
        self._emit(
            emit,
            "conditioning",
            ms=timing["conditioning_ms"],
            cross_attn=tensor_stats(cross_attn),
            global_cond=tensor_stats(global_cond),
        )
        self._emit(emit, "stage", stage="conditioning", state="done", ms=timing["conditioning_ms"])

        # 3. Optional SAME encoding
        init_latents = None
        ctx_latents = None
        if mode in {"audio", "inpaint"}:
            self._emit(emit, "stage", stage="encoder", state="active", label=f"{decoder_name} input encoding")
            t0 = time.perf_counter()
            encoder, pad_mod = self._get_encoder(decoder_name, emit)
            enc_t_lat = t_lat
            if (t_lat * 16) % pad_mod != 0:
                enc_t_lat = math.ceil((t_lat * 16) / pad_mod) * pad_mod // 16
            target_samples = enc_t_lat * SAMPLES_PER_LATENT
            audio_np = read_wav(str(input_audio))
            if audio_np.shape[-1] >= target_samples:
                audio_np = audio_np[:, :target_samples]
            else:
                audio_np = np.pad(audio_np, ((0, 0), (0, target_samples - audio_np.shape[-1])))
            self._emit(emit, "input_audio", waveform=waveform_preview(audio_np), stats=_stats_np(audio_np))
            patches_np = patch_audio(audio_np[None, ...], patch_size=256)
            lat = encoder(mx.array(patches_np))[..., :t_lat]
            mx.eval(lat)
            timing["encode_ms"] = (time.perf_counter() - t0) * 1000
            self._emit(
                emit,
                "audio_latent",
                ms=timing["encode_ms"],
                stats=tensor_stats(lat),
                heatmap=heatmap_preview(lat),
            )
            self._emit(emit, "stage", stage="encoder", state="done", ms=timing["encode_ms"])
            if mode == "audio":
                init_latents = lat.astype(dtype)
            else:
                ctx_latents = lat.astype(dtype)
        else:
            self._emit(emit, "stage", stage="encoder", state="skipped", label="No input audio")

        # 4. DiT + ping-pong sampler
        self._emit(emit, "stage", stage="dit", state="active", label="DiT latent sampling")
        t0_load = time.perf_counter()
        dit_model = self._get_dit(dit_name, t_lat, dtype, steps, emit)
        timing["dit_ready_ms"] = (time.perf_counter() - t0_load) * 1000
        sigmas = build_pingpong_schedule(steps, sigma_max=sigma_max, use_logsnr_shift=True)
        key = mx.random.key(seed)
        pure_noise = mx.random.normal((1, 256, t_lat), dtype=dtype, key=key)
        noise = pure_noise if init_latents is None else init_latents * (1.0 - sigma_max) + pure_noise * sigma_max
        mx.eval(noise)

        local_add_cond = None
        paste_back = None
        if ctx_latents is not None:
            s0 = max(0, int(round(inpaint_start * SAMPLE_RATE / SAMPLES_PER_LATENT)))
            s1 = min(t_lat, int(round(inpaint_end * SAMPLE_RATE / SAMPLES_PER_LATENT)))
            mask_np = np.ones((1, 1, t_lat), dtype=np.float32)
            mask_np[:, :, s0:s1] = 0.0
            keep = mx.array(mask_np)
            masked_input = ctx_latents.astype(mx.float32) * keep
            local_add_cond = mx.concatenate([keep, masked_input], axis=1).transpose(0, 2, 1).astype(dtype)
            paste_back = (ctx_latents, keep)
            self._emit(emit, "inpaint_mask", start_lat=s0, end_lat=s1, total_lat=t_lat)

        def model_fn(x, tt):
            if cfg == 1.0:
                return dit_model(x, tt, cross_attn, global_cond, local_add_cond=local_add_cond)
            x2 = mx.concatenate([x, x], axis=0)
            t2 = mx.concatenate([tt, tt], axis=0)
            cross2 = mx.concatenate([cross_attn, null_cross_attn], axis=0)
            global2 = mx.concatenate([global_cond, global_cond], axis=0)
            lac2 = None if local_add_cond is None else mx.concatenate([local_add_cond, local_add_cond], axis=0)
            v2 = dit_model(x2, t2, cross2, global2, local_add_cond=lac2)
            cond_v, uncond_v = mx.split(v2, 2, axis=0)
            sigma = tt.reshape(-1, 1, 1).astype(mx.float32)
            x32 = x.astype(mx.float32)
            cond_d = x32 - cond_v.astype(mx.float32) * sigma
            uncond_d = x32 - uncond_v.astype(mx.float32) * sigma
            diff = cond_d - uncond_d
            if apg <= 0.0 or cfg < 1.0:
                guided_diff = diff
            else:
                norm = mx.sqrt((cond_d * cond_d).sum(axis=(-2, -1), keepdims=True))
                unit = cond_d / mx.maximum(norm, 1e-8)
                parallel = (diff * unit).sum(axis=(-2, -1), keepdims=True) * unit
                orth = diff - parallel
                guided_diff = orth if apg >= 1.0 else apg * orth + (1.0 - apg) * diff
            guided_d = cond_d + (cfg - 1.0) * guided_diff
            return ((x32 - guided_d) / sigma).astype(x.dtype)

        t0 = time.perf_counter()
        lora_plan = getattr(dit_model, "_lora_plan", None)
        latents = self._sample(
            model_fn,
            noise,
            sigmas,
            emit,
            seed=seed + 1,
            paste_back=paste_back,
            before_step=lora_plan.sync if lora_plan else None,
        )
        mx.eval(latents)
        timing["sample_ms"] = (time.perf_counter() - t0) * 1000
        self._emit(
            emit,
            "latent_final",
            ms=timing["sample_ms"],
            stats=tensor_stats(latents),
            heatmap=heatmap_preview(latents),
        )
        self._emit(emit, "stage", stage="dit", state="done", ms=timing["sample_ms"])

        # 5. SAME decode
        self._emit(emit, "stage", stage="decoder", state="active", label=f"{decoder_name} decoding")
        t0 = time.perf_counter()
        decoder, chunk_fn, (chunk, overlap) = self._get_decoder(decoder_name, emit)
        lat32 = latents.astype(mx.float32)
        kernel = chunk + 2 * overlap
        if t_lat > kernel:
            patches = chunk_fn(decoder, lat32, chunk, overlap)
            decode_mode = f"chunked {chunk}+2x{overlap}"
        elif t_lat % 2 == 0:
            patches = decoder(lat32)
            decode_mode = "direct"
        elif t_lat > 6:
            patches = chunk_fn(decoder, lat32, 2, 2)
            decode_mode = "chunked 2+2x2"
        else:
            lat_even = mx.concatenate([lat32, lat32[..., -1:]], axis=-1)
            patches = decoder(lat_even)[..., : t_lat * 16]
            decode_mode = "odd-pad/trim"
        mx.eval(patches)
        timing["decode_ms"] = (time.perf_counter() - t0) * 1000
        self._emit(
            emit,
            "decoder_patches",
            ms=timing["decode_ms"],
            mode=decode_mode,
            stats=tensor_stats(patches),
            heatmap=heatmap_preview(patches, rows=64, cols=96),
        )
        self._emit(emit, "stage", stage="decoder", state="done", ms=timing["decode_ms"])

        # 6. Unpatch + output
        self._emit(emit, "stage", stage="output", state="active", label="Unpatching + writing WAV")
        t0 = time.perf_counter()
        audio = patched_decode(patches, patch_size=256, channels=2)
        mx.eval(audio)
        audio_np = np.array(audio.astype(mx.float32))[0]
        requested = int(round(seconds * SAMPLE_RATE))
        if audio_np.shape[-1] > requested:
            audio_np = audio_np[..., :requested]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_wav(str(out), audio_np, SAMPLE_RATE)
        timing["output_ms"] = (time.perf_counter() - t0) * 1000
        timing["total_ms"] = (time.perf_counter() - started) * 1000
        timing["realtime"] = seconds / max(timing["total_ms"] / 1000.0, 1e-9)
        self._emit(
            emit,
            "output_audio",
            ms=timing["output_ms"],
            stats=_stats_np(audio_np),
            waveform=waveform_preview(audio_np),
            sample_rate=SAMPLE_RATE,
        )
        self._emit(emit, "stage", stage="output", state="done", ms=timing["output_ms"])
        self._emit(
            emit,
            "run_complete",
            seed=seed,
            output_name=out.name,
            timing=timing,
            final_stats=_stats_np(audio_np),
        )
        return {"seed": seed, "output_path": str(out), "timing": timing}
