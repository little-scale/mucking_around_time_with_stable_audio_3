"""Stable browser/backend event contract for SA3 Monitor 1.x.

Backends may add fields and diagnostic event types, but must not remove or
rename these event families without a contract-version bump.
"""

CONTRACT_VERSION = "1.0"
CORE_EVENTS = frozenset({
    "stage",
    "run_start",
    "embedding",
    "conditioning",
    "sampler_init",
    "sampler_step",
    "latent_final",
    "output_audio",
    "run_complete",
    "tensor_preview",
    "metrics",
    "audio_ready",
})
OPTIONAL_EVENTS = frozenset({
    "model_load",
    "input_audio",
    "audio_latent",
    "inpaint_mask",
    "paste_back",
    "decoder_patches",
    "log",
})
SERVER_EVENTS = frozenset({"status", "error"})


def validate_event(item: dict) -> None:
    if not isinstance(item, dict):
        raise TypeError("event must be a dictionary")
    if not isinstance(item.get("type"), str):
        raise ValueError("event.type must be a string")
    if "time" not in item:
        raise ValueError("event.time is required")
    if item["type"] == "stage":
        if item.get("stage") not in {"t5", "conditioning", "encoder", "dit", "decoder", "output"}:
            raise ValueError("invalid stage name")
        if item.get("state") not in {"active", "done", "skipped", "error"}:
            raise ValueError("invalid stage state")
    if item["type"] == "sampler_step":
        for field in ("step", "total", "sigma", "stats", "heatmap"):
            if field not in item:
                raise ValueError(f"sampler_step.{field} is required")
    if item["type"] == "output_audio":
        for field in ("stats", "waveform", "sample_rate"):
            if field not in item:
                raise ValueError(f"output_audio.{field} is required")
