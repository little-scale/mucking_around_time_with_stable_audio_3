from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

from backends.base import BackendInfo, BackendUnavailable, CONTRACT_VERSION, array_stats, heatmap_preview, waveform_preview


MODEL_MAP = {
    "sm-music": ("small-music", "same-s"),
    "sm-sfx": ("small-sfx", "same-s"),
    "medium": ("medium", "same-l"),
}


class CUDAMonitorEngine:
    """Adapter for Stability AI's official PyTorch StableAudioModel API.

    The upstream sampler already exposes a callback containing x, denoised,
    sigma and step index.  This adapter translates it into the exact monitor
    events consumed by the shared browser.
    """

    name = "cuda"
    dit_choices = tuple(MODEL_MAP.keys())
    decoder_choices = ("same-s", "same-l")
    max_seconds = 380.0
    sample_rate = 44_100

    def __init__(self):
        try:
            import torch
            import torchaudio
            from stable_audio_3 import AutoencoderModel, StableAudioModel
        except ImportError as exc:
            raise BackendUnavailable(
                "CUDA backend dependencies are missing. Install the official stable-audio-3 "
                "package/environment plus monitor requirements; see README.md. "
                f"Original import error: {exc}"
            ) from exc
        if not torch.cuda.is_available():
            raise BackendUnavailable(
                "PyTorch imported, but torch.cuda.is_available() is false. Check the NVIDIA "
                "driver and install a CUDA PyTorch wheel (the tested upstream pin is torch 2.7.1)."
            )
        self.torch = torch
        self.torchaudio = torchaudio
        self.StableAudioModel = StableAudioModel
        self.AutoencoderModel = AutoencoderModel
        self.device = "cuda:0"
        self._models: dict[str, object] = {}
        self._autoencoders: dict[str, object] = {}

    def _memory(self) -> int:
        return int(self.torch.cuda.max_memory_allocated(0))

    def _emit(self, emit, kind: str, **data):
        emit({
            "type": kind,
            "time": time.time(),
            "contract_version": CONTRACT_VERSION,
            "memory_bytes": self._memory(),
            **data,
        })

    def diagnostics(self) -> BackendInfo:
        p = self.torch.cuda.get_device_properties(0)
        free, total = self.torch.cuda.mem_get_info(0)
        return BackendInfo(
            backend="cuda",
            device=self.device,
            device_name=p.name,
            dtype="float16",
            vram_total_bytes=int(total),
            vram_free_bytes=int(free),
            framework="PyTorch/CUDA",
            framework_version=f"torch {self.torch.__version__} / CUDA {self.torch.version.cuda}",
        )

    def _load_model(self, dit_name: str, emit):
        upstream_name, _ = MODEL_MAP[dit_name]
        if upstream_name in self._models:
            self._emit(emit, "model_load", model=upstream_name, ms=0.0, cached=True)
            return self._models[upstream_name]
        if self._autoencoders:
            self._autoencoders.clear()
            self.torch.cuda.empty_cache()
        self._emit(emit, "stage", stage="t5", state="active", label=f"Loading {upstream_name} + T5Gemma")
        started = time.perf_counter()
        try:
            model = self.StableAudioModel.from_pretrained(upstream_name, device=self.device, model_half=True)
        except Exception as exc:
            raise BackendUnavailable(
                f"Could not load official SA3 model '{upstream_name}'. Ensure the Hugging Face "
                "license has been accepted, `hf auth login` succeeds, and the model cache is writable. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        self._models[upstream_name] = model
        self._emit(emit, "model_load", model=upstream_name, ms=(time.perf_counter() - started) * 1000, cached=False)
        return model

    def _tensor_np(self, value):
        return value.detach().float().cpu().numpy()

    def _tensor_event(self, emit, kind: str, tensor, **payload):
        a = self._tensor_np(tensor)
        self._emit(emit, kind, stats=array_stats(a), heatmap=heatmap_preview(a), **payload)

    def _first_tensor(self, value):
        if self.torch.is_tensor(value):
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                found = self._first_tensor(item)
                if found is not None:
                    return found
        if isinstance(value, dict):
            tensors = []
            for item in value.values():
                found = self._first_tensor(item)
                if found is not None:
                    tensors.append(found)
            if tensors:
                return max(tensors, key=lambda x: x.numel())
        return None

    @property
    def same_codec_choices(self) -> tuple[str, ...]:
        return ("same-s", "same-l")

    def _load_autoencoder(self, codec_name: str):
        if codec_name not in self.same_codec_choices:
            raise ValueError(f"Unknown SAME codec: {codec_name}")
        if codec_name not in self._autoencoders:
            # Avoid retaining a full DiT pipeline and a duplicate standalone
            # SAME model on 16 GB classroom GPUs. The HF disk cache remains hot.
            self._models.clear()
            self._autoencoders.clear()
            self.torch.cuda.empty_cache()
            try:
                self._autoencoders[codec_name] = self.AutoencoderModel.from_pretrained(codec_name, device=self.device)
            except Exception as exc:
                raise BackendUnavailable(
                    f"Could not load official SAME codec '{codec_name}'. Accept its Hugging Face license "
                    f"and run hf auth login. Original error: {type(exc).__name__}: {exc}"
                ) from exc
        return self._autoencoders[codec_name]

    def same_encode(self, input_path: str, codec_name: str) -> dict:
        model = self._load_autoencoder(codec_name)
        waveform, sample_rate = self.torchaudio.load(str(input_path))
        target_samples = int(round(waveform.shape[-1] * self.sample_rate / int(sample_rate)))
        with self.torch.inference_mode():
            chunked = target_samples > 128 * 4096
            latent = model.encode(waveform, int(sample_rate), chunked=chunked)
            baseline = model.decode(latent, chunked=chunked).float().clamp(-1, 1)
        self.torch.cuda.synchronize()
        latent_np = self._tensor_np(latent[0])
        audio_np = self._tensor_np(baseline[0, :, :target_samples])
        return {
            "latent": latent_np.astype(np.float32, copy=False),
            "audio": audio_np.astype(np.float32, copy=False),
            "target_samples": target_samples,
            "sample_rate": self.sample_rate,
        }

    def same_decode(self, latent: np.ndarray, codec_name: str, target_samples: int) -> np.ndarray:
        model = self._load_autoencoder(codec_name)
        dtype = next(model.autoencoder.parameters()).dtype
        values = self.torch.from_numpy(np.asarray(latent, dtype=np.float32)[None, ...]).to(self.device, dtype=dtype)
        with self.torch.inference_mode():
            audio = model.decode(values, chunked=values.shape[-1] > 128).float().clamp(-1, 1)
        self.torch.cuda.synchronize()
        result = self._tensor_np(audio[0, :, : int(target_samples)])
        if not np.isfinite(result).all():
            raise RuntimeError("SAME decoded non-finite audio; reduce the edit amount")
        return result.astype(np.float32, copy=False)

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
        decoder_name: str = "auto",
        mode: str = "text",
        input_audio: Optional[str] = None,
        inpaint_start: float = 0.0,
        inpaint_end: float = 0.0,
        output_path: str,
        emit,
    ) -> dict:
        if dit_name not in MODEL_MAP:
            raise ValueError(f"Unknown DiT: {dit_name}")
        upstream_name, required_codec = MODEL_MAP[dit_name]
        if decoder_name not in (None, "auto", required_codec):
            raise ValueError(f"{dit_name} is paired with {required_codec}; decoder override {decoder_name!r} is incompatible")
        if mode not in {"text", "audio", "inpaint"}:
            raise ValueError("mode must be text, audio or inpaint")
        if mode != "text" and not input_audio:
            raise ValueError(f"{mode} mode requires input audio")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)
        started = time.perf_counter()
        timing: dict[str, float] = {}
        self.torch.cuda.reset_peak_memory_stats(0)
        info = self.diagnostics()
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
            decoder=required_codec,
            mode=mode,
            t_lat=None,
            sample_rate=self.sample_rate,
            latent_channels=256,
            backend=info.as_dict(),
        )

        model = self._load_model(dit_name, emit)

        # Build and inspect the real upstream conditioner output once, then pass
        # it into generate() so prompt encoding is not repeated.
        t0 = time.perf_counter()
        conditioning, negative_conditioning = model._build_conditioning_dicts(
            prompt, negative_prompt if negative_prompt.strip() else None, seconds, 1
        )
        with self.torch.inference_mode():
            conditioning_tensors = model.model.conditioner(conditioning, self.device)
            negative_tensors = model.model.conditioner(negative_conditioning, self.device) if negative_conditioning else None
        timing["t5_ms"] = (time.perf_counter() - t0) * 1000
        preview = self._first_tensor(conditioning_tensors)
        if preview is not None:
            self._tensor_event(emit, "embedding", preview, role="prompt", tokens=int(preview.shape[-2]) if preview.ndim >= 2 else 0, ms=timing["t5_ms"])
        self._emit(emit, "stage", stage="t5", state="done", ms=timing["t5_ms"])
        self._emit(emit, "stage", stage="conditioning", state="active", label="Official SA3 conditioning tensors")
        cond_stats = array_stats(self._tensor_np(preview)) if preview is not None else array_stats(np.zeros(1))
        self._emit(emit, "conditioning", ms=0.0, cross_attn=cond_stats, global_cond=cond_stats)
        self._emit(emit, "stage", stage="conditioning", state="done", ms=0.0)

        audio_tuple = None
        encoder_pending = False
        encoder_started = 0.0
        if mode != "text":
            self._emit(emit, "stage", stage="encoder", state="active", label=f"{required_codec} input encoding")
            encoder_started = time.perf_counter()
            encoder_pending = True
            wav, sr = self.torchaudio.load(str(input_audio))
            wav_np = self._tensor_np(wav)
            self._emit(emit, "input_audio", waveform=waveform_preview(wav_np), stats=array_stats(wav_np))
            audio_tuple = (int(sr), wav)
            # generate() performs the authoritative SAME encode. The first
            # sampler callback below closes this stage, avoiding a second encode.
        else:
            self._emit(emit, "stage", stage="encoder", state="skipped", label="No input audio")

        if not encoder_pending:
            self._emit(emit, "stage", stage="dit", state="active", label="PyTorch CUDA ping-pong sampler")
        last_step = time.perf_counter()

        def callback(data):
            nonlocal last_step, encoder_pending
            x = data.get("x")
            now = time.perf_counter()
            step = int(data.get("i", 0)) + 1
            if encoder_pending:
                self._emit(emit, "stage", stage="encoder", state="done", ms=(now - encoder_started) * 1000)
                self._emit(emit, "stage", stage="dit", state="active", label="PyTorch CUDA ping-pong sampler")
                encoder_pending = False
            sigma_value = data.get("sigma", data.get("t", 0.0))
            sigma = float(sigma_value.flatten()[0].detach().float().cpu()) if self.torch.is_tensor(sigma_value) else float(sigma_value)
            if step == 1 and x is not None:
                a = self._tensor_np(x)
                self._emit(emit, "sampler_init", total=steps, sigma=sigma, stats=array_stats(a), heatmap=heatmap_preview(a))
            if x is not None:
                a = self._tensor_np(data.get("denoised", x))
                self._emit(
                    emit,
                    "sampler_step",
                    step=step,
                    total=steps,
                    sigma=sigma,
                    ms=(now - last_step) * 1000,
                    stats=array_stats(a),
                    heatmap=heatmap_preview(a),
                )
            last_step = now

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            duration=float(seconds),
            steps=int(steps),
            cfg_scale=float(cfg),
            apg_scale=float(apg),
            seed=seed,
            conditioning=conditioning,
            conditioning_tensors=conditioning_tensors,
            negative_conditioning=negative_conditioning,
            negative_conditioning_tensors=negative_tensors,
            return_latents=True,
            callback=callback,
            disable_tqdm=True,
            sampler_type="pingpong",
        )
        if mode == "audio":
            kwargs.update(init_audio=audio_tuple, init_noise_level=float(sigma_max))
        elif mode == "inpaint":
            kwargs.update(
                inpaint_audio=audio_tuple,
                inpaint_mask_start_seconds=float(inpaint_start),
                inpaint_mask_end_seconds=float(inpaint_end),
            )
        t0 = time.perf_counter()
        latents = model.generate(**kwargs)
        self.torch.cuda.synchronize()
        timing["sample_ms"] = (time.perf_counter() - t0) * 1000
        self._tensor_event(emit, "latent_final", latents, ms=timing["sample_ms"])
        self._emit(emit, "stage", stage="dit", state="done", ms=timing["sample_ms"])

        self._emit(emit, "stage", stage="decoder", state="active", label=f"{required_codec} CUDA decode")
        t0 = time.perf_counter()
        latents = latents.to(next(model.same.parameters()).dtype)
        with self.torch.inference_mode():
            audio = model.same.decode(latents, chunked=None).float().clamp(-1, 1)
        audio = audio[..., : int(round(float(seconds) * self.sample_rate))]
        self.torch.cuda.synchronize()
        timing["decode_ms"] = (time.perf_counter() - t0) * 1000
        self._tensor_event(emit, "decoder_patches", audio, ms=timing["decode_ms"], mode="official pretransform.decode")
        self._emit(emit, "stage", stage="decoder", state="done", ms=timing["decode_ms"])

        self._emit(emit, "stage", stage="output", state="active", label="Writing WAV on CUDA host")
        t0 = time.perf_counter()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        audio_np = self._tensor_np(audio[0])
        import soundfile as sf
        sf.write(str(out), audio_np.T, self.sample_rate, subtype="PCM_16")
        timing["output_ms"] = (time.perf_counter() - t0) * 1000
        timing["total_ms"] = (time.perf_counter() - started) * 1000
        timing["realtime"] = float(seconds) / max(timing["total_ms"] / 1000.0, 1e-9)
        stats = array_stats(audio_np)
        self._emit(emit, "output_audio", ms=timing["output_ms"], stats=stats, waveform=waveform_preview(audio_np), sample_rate=self.sample_rate)
        self._emit(emit, "stage", stage="output", state="done", ms=timing["output_ms"])
        self._emit(emit, "run_complete", seed=seed, output_name=out.name, timing=timing, final_stats=stats)
        return {"seed": seed, "output_path": str(out), "timing": timing}
