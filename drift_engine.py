from __future__ import annotations

import asyncio
import json
import os
import random
import re
import shutil
import threading
import uuid
import wave
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


TRACK_IDS = ("melodic", "percussion", "texture")
TRACK_MODELS = {
    "melodic": "sm-music",
    "percussion": "sm-music",
    "texture": "sm-sfx",
}
MODEL_IDS = ("sm-music", "sm-sfx", "medium")
TRACK_LABELS = {
    "melodic": "Melodic",
    "percussion": "Percussion",
    "texture": "Texture",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _weighted_prompt(tags: list[dict[str, str]]) -> str:
    phrases: list[str] = []
    for tag in tags:
        text = str(tag.get("text", "")).strip()
        if not text:
            continue
        weight = tag.get("weight", "normal")
        if weight == "subtle":
            phrases.append(f"subtle hint of {text}")
        elif weight == "strong":
            phrases.append(f"prominent {text}, {text}")
        else:
            phrases.append(text)
    return ", ".join(phrases)


@dataclass(frozen=True, slots=True)
class TrackConfig:
    prompt_tags: list[dict[str, str]]
    negative_prompt: str | None
    role_prompt: str | None
    model: str = "sm-music"
    beats: int = 16
    mutation: float = 0.14
    cfg: float = 2.0
    steps: int = 8
    mutation_interval_loops: int = 1
    seed_mode: str = "random"
    seed: int | None = None
    frozen: bool = False
    revision: int = 0

    @classmethod
    def defaults(cls, track_id: str) -> TrackConfig:
        if track_id == "melodic":
            return cls(
                prompt_tags=[{"text": "warm evolving harmony", "weight": "normal"}],
                negative_prompt="drums, percussion, cymbals, hi-hats, kick, snare",
                role_prompt="seamless pitched chordal melodic instrumental loop",
                model="sm-music",
            )
        if track_id == "percussion":
            return cls(
                prompt_tags=[{"text": "structured percussion", "weight": "normal"}],
                negative_prompt="melody, chords, harmony, bassline, vocals, pitched instruments",
                role_prompt="seamless structured percussion loop",
                model="sm-music",
            )
        if track_id == "texture":
            return cls(
                prompt_tags=[
                    {"text": "ambient texture", "weight": "normal"},
                    {"text": "evolving drone", "weight": "normal"},
                ],
                negative_prompt=None,
                role_prompt="seamless ambient texture and evolving drone",
                model="sm-sfx",
                cfg=1.0,
            )
        raise ValueError(f"unknown track: {track_id}")

    @classmethod
    def from_file(
        cls,
        path: Path,
        track_id: str,
        *,
        legacy_path: Path | None = None,
        bpm: float = 120.0,
    ) -> TrackConfig:
        default = cls.defaults(track_id)
        source = path if path.exists() else legacy_path
        if source is None or not source.exists():
            return default
        data = json.loads(source.read_text())
        if source == legacy_path:
            tags = data.get("prompt_tags")
            if not tags:
                tags = [
                    {"text": text.strip(), "weight": "normal"}
                    for text in re.split(r"[,;\n]+", data.get("target_prompt", ""))
                    if text.strip()
                ]
            tags = [
                tag for tag in tags
                if not re.fullmatch(r"\s*\d+(?:\.\d+)?\s*bpm\s*", str(tag.get("text", "")), re.IGNORECASE)
            ]
            loop_seconds = float(data.get("loop_seconds", 8.0))
            beats = max(1, round(loop_seconds * bpm / 60.0))
            data = {
                "prompt_tags": tags or default.prompt_tags,
                "negative_prompt": data.get("negative_prompt", default.negative_prompt),
                "role_prompt": default.role_prompt,
                "beats": beats,
                "mutation": data.get("mutation", default.mutation),
                "cfg": data.get("cfg", default.cfg),
                "steps": data.get("steps", default.steps),
                "mutation_interval_loops": 1,
                "seed_mode": data.get("seed_mode", default.seed_mode),
                "seed": data.get("seed", default.seed),
                "frozen": False,
                "revision": 0,
            }
        migrated_cadence = "cadence_seconds" in data and "mutation_interval_loops" not in data
        if migrated_cadence:
            data = {**data, "mutation_interval_loops": 1}
        allowed = set(cls.__dataclass_fields__)
        loaded = {**asdict(default), **{key: value for key, value in data.items() if key in allowed}}
        unsafe_mutation = float(loaded.get("mutation", default.mutation)) > 1.0
        if unsafe_mutation:
            loaded["mutation"] = 1.0
        config = cls(**loaded)
        config.validate()
        if (unsafe_mutation or migrated_cadence) and source == path:
            _atomic_json(path, asdict(config))
        return config

    def validate(self) -> None:
        if not self.prompt_tags:
            raise ValueError("a track must contain at least one prompt tag")
        for tag in self.prompt_tags:
            if not isinstance(tag, dict) or not str(tag.get("text", "")).strip():
                raise ValueError("each prompt tag must have non-empty text")
            if tag.get("weight") not in {"subtle", "normal", "strong"}:
                raise ValueError("prompt tag weight must be subtle, normal, or strong")
        if self.model not in MODEL_IDS:
            raise ValueError(f"model must be one of: {', '.join(MODEL_IDS)}")
        if self.beats < 1 or self.beats > 512:
            raise ValueError("beats must be between 1 and 512")
        if self.mutation < 0.01 or self.mutation > 1:
            raise ValueError("mutation must be between 0.01 and 1")
        if self.steps < 1 or self.steps > 64:
            raise ValueError("steps must be between 1 and 64")
        if self.mutation_interval_loops not in {1, 2, 4, 8}:
            raise ValueError("mutation_interval_loops must be 1, 2, 4, or 8")
        if self.seed_mode not in {"random", "increment", "fixed"}:
            raise ValueError("seed_mode must be random, increment, or fixed")
        if self.seed_mode in {"increment", "fixed"} and self.seed is None:
            raise ValueError(f"seed is required when seed_mode is {self.seed_mode}")

    def prompt(self, track_id: str, bpm: float) -> str:
        parts = [_weighted_prompt(self.prompt_tags)]
        if track_id in {"melodic", "percussion"}:
            bpm_text = str(int(bpm)) if float(bpm).is_integer() else f"{bpm:g}"
            parts.append(f"{bpm_text} BPM")
        parts.append((self.role_prompt or "").strip())
        return ", ".join(part for part in parts if part)


@dataclass(frozen=True, slots=True)
class TrackMix:
    volume: float = 0.82
    muted: bool = False
    solo: bool = False
    highpass_hz: float = 20.0
    lowpass_hz: float = 20000.0
    reverb_send: float = 0.18
    delay_send: float = 0.0

    def validate(self) -> None:
        if not 0 <= self.volume <= 1.5:
            raise ValueError("volume must be between 0 and 1.5")
        if not 20 <= self.highpass_hz <= 20000:
            raise ValueError("highpass_hz must be between 20 and 20000")
        if not 20 <= self.lowpass_hz <= 20000:
            raise ValueError("lowpass_hz must be between 20 and 20000")
        if self.highpass_hz > self.lowpass_hz:
            raise ValueError("highpass_hz cannot exceed lowpass_hz")
        if not 0 <= self.reverb_send <= 1:
            raise ValueError("reverb_send must be between 0 and 1")
        if not 0 <= self.delay_send <= 1:
            raise ValueError("delay_send must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class SidechainConfig:
    enabled: bool = True
    duck_melodic: bool = True
    duck_texture: bool = False
    threshold_db: float = -24.0
    ratio: float = 4.0
    attack_ms: float = 12.0
    release_ms: float = 180.0
    makeup_db: float = 0.0
    depth: float = 1.0

    def validate(self) -> None:
        if not -80 <= self.threshold_db <= 0:
            raise ValueError("sidechain threshold_db must be between -80 and 0")
        if not 1 <= self.ratio <= 20:
            raise ValueError("sidechain ratio must be between 1 and 20")
        if not 0.1 <= self.attack_ms <= 1000 or not 1 <= self.release_ms <= 5000:
            raise ValueError("sidechain attack or release is outside its supported range")
        if not -24 <= self.makeup_db <= 24 or not 0 <= self.depth <= 1:
            raise ValueError("sidechain makeup or depth is outside its supported range")


@dataclass(frozen=True, slots=True)
class ReverbConfig:
    enabled: bool = True
    return_level: float = 0.24
    decay_seconds: float = 3.5
    predelay_ms: float = 24.0
    damping_hz: float = 7000.0

    def validate(self) -> None:
        if not 0 <= self.return_level <= 1:
            raise ValueError("reverb return_level must be between 0 and 1")
        if not 0.1 <= self.decay_seconds <= 20:
            raise ValueError("reverb decay_seconds must be between 0.1 and 20")
        if not 0 <= self.predelay_ms <= 500:
            raise ValueError("reverb predelay_ms must be between 0 and 500")
        if not 200 <= self.damping_hz <= 20000:
            raise ValueError("reverb damping_hz must be between 200 and 20000")


@dataclass(frozen=True, slots=True)
class DelayConfig:
    enabled: bool = False
    return_level: float = 0.24
    division: str = "1/4"
    feedback: float = 0.36
    lowpass_hz: float = 6500.0
    stereo_width: float = 0.7

    def validate(self) -> None:
        if not 0 <= self.return_level <= 1:
            raise ValueError("delay return_level must be between 0 and 1")
        if self.division not in {"1/1", "1/2", "1/4D", "1/4", "1/8D", "1/8", "1/8T", "1/16"}:
            raise ValueError("delay division is not supported")
        if not 0 <= self.feedback <= 0.92:
            raise ValueError("delay feedback must be between 0 and 0.92")
        if not 200 <= self.lowpass_hz <= 20000:
            raise ValueError("delay lowpass_hz must be between 200 and 20000")
        if not 0 <= self.stereo_width <= 1:
            raise ValueError("delay stereo_width must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class SessionConfig:
    bpm: float = 120.0
    master_volume: float = 0.82
    mixer: dict[str, dict[str, Any]] | None = None
    sidechain: dict[str, Any] | None = None
    reverb: dict[str, Any] | None = None
    delay: dict[str, Any] | None = None
    revision: int = 0

    @classmethod
    def from_file(cls, path: Path) -> SessionConfig:
        defaults = cls(
            mixer={track_id: asdict(TrackMix()) for track_id in TRACK_IDS},
            sidechain=asdict(SidechainConfig()),
            reverb=asdict(ReverbConfig()),
            delay=asdict(DelayConfig()),
        )
        if not path.exists():
            return defaults
        data = json.loads(path.read_text())
        mixer = {track_id: asdict(TrackMix()) for track_id in TRACK_IDS}
        for track_id, changes in (data.get("mixer") or {}).items():
            if track_id in mixer and isinstance(changes, dict):
                mixer[track_id].update(changes)
        sidechain = {**asdict(SidechainConfig()), **(data.get("sidechain") or {})}
        reverb = {**asdict(ReverbConfig()), **(data.get("reverb") or {})}
        delay = {**asdict(DelayConfig()), **(data.get("delay") or {})}
        config = cls(
            bpm=data.get("bpm", defaults.bpm),
            master_volume=data.get("master_volume", defaults.master_volume),
            mixer=mixer,
            sidechain=sidechain,
            reverb=reverb,
            delay=delay,
            revision=data.get("revision", 0),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 30 <= self.bpm <= 300:
            raise ValueError("bpm must be between 30 and 300")
        if not 0 <= self.master_volume <= 1.5:
            raise ValueError("master_volume must be between 0 and 1.5")
        for track_id in TRACK_IDS:
            TrackMix(**(self.mixer or {}).get(track_id, {})).validate()
        SidechainConfig(**(self.sidechain or {})).validate()
        ReverbConfig(**(self.reverb or {})).validate()
        DelayConfig(**(self.delay or {})).validate()


class GenerationError(RuntimeError):
    def __init__(self, detail: dict[str, Any]):
        super().__init__(detail["message"])
        self.detail = detail


class _TrackState:
    def __init__(self, engine: MultitrackEngine, track_id: str) -> None:
        self.id = track_id
        self.label = engine.track_labels[track_id]
        self.output_dir = engine.output_dir / "tracks" / track_id
        self.versions_dir = self.output_dir / "versions"
        self.work_dir = self.output_dir / ".work"
        self.current_path = self.output_dir / "current.wav"
        self.current_metadata_path = self.output_dir / "current.json"
        self.lineage_state_path = self.output_dir / "lineage_state.json"
        self.introduction_state_path = self.output_dir / "introduction.json"
        self.config_path = self.output_dir / "config.json"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        legacy_config = engine.output_dir / "config.json" if track_id == "melodic" else None
        self.config = TrackConfig.from_file(
            self.config_path,
            track_id,
            legacy_path=legacy_config,
            bpm=engine.session.bpm,
        )
        overrides = engine.track_config_overrides.get(track_id, {})
        if overrides:
            overridden = replace(self.config, **overrides)
            overridden.validate()
            if overridden != self.config:
                self.config = replace(overridden, revision=self.config.revision + 1)
                _atomic_json(self.config_path, asdict(self.config))
        if track_id == "melodic" and not self.current_path.exists():
            legacy_current = engine.output_dir / "current.wav"
            if legacy_current.exists():
                shutil.copyfile(legacy_current, self.current_path)
        if not self.config_path.exists():
            _atomic_json(self.config_path, asdict(self.config))
        self.versions = self._load_versions()
        persisted_generation = self._load_generation_counter()
        self.generation = max(persisted_generation, max((item["generation"] for item in self.versions), default=0))
        self.last_completed = self._load_current_metadata() or (self.versions[0] if self.versions else None)
        self.introduction = self._load_introduction()
        active_introduction = self.introduction.get("filename") if self.introduction else None
        for stale_introduction in self.output_dir.glob("introduction-*.wav"):
            if stale_introduction.name != active_introduction:
                stale_introduction.unlink(missing_ok=True)
        self.last_error = self._load_last_error()
        self.phase = "frozen" if self.config.frozen else "stopped"
        self.generation_started_at: str | None = None
        self.next_generation_timestamp: float | None = None
        self.active_job_id: str | None = None
        self.pending_config: TrackConfig | None = None
        self.dirty = False
        self.failed = False
        self.client_ip = "unknown"
        for stale_work in self.work_dir.glob("*.wav"):
            stale_work.unlink(missing_ok=True)

    @property
    def model(self) -> str:
        return self.config.model

    def _load_last_error(self) -> dict[str, Any] | None:
        path = self.output_dir / "last_error.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {"at": _utc_now(), "message": "last_error.json could not be read", "returncode": None}

    def _load_generation_counter(self) -> int:
        if not self.lineage_state_path.exists():
            return 0
        try:
            return max(0, int(json.loads(self.lineage_state_path.read_text()).get("generation", 0)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return 0

    def _load_current_metadata(self) -> dict[str, Any] | None:
        if not self.current_path.exists() or not self.current_metadata_path.exists():
            return None
        try:
            return json.loads(self.current_metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _load_introduction(self) -> dict[str, Any] | None:
        if not self.introduction_state_path.exists():
            return None
        try:
            value = json.loads(self.introduction_state_path.read_text())
            audio_path = self.output_dir / value["filename"]
            if not audio_path.exists() or int(value.get("generations_remaining", 0)) < 1:
                self.introduction_state_path.unlink(missing_ok=True)
                return None
            return value
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.introduction_state_path.unlink(missing_ok=True)
            return None

    def _load_versions(self) -> list[dict[str, Any]]:
        versions: list[dict[str, Any]] = []
        for audio_path in self.versions_dir.glob("generation_*.wav"):
            try:
                generation = int(audio_path.stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            metadata_path = audio_path.with_suffix(".json")
            try:
                item = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            except (OSError, json.JSONDecodeError):
                item = {}
            item.setdefault("generation", generation)
            item.setdefault("filename", audio_path.name)
            item.setdefault("track_id", self.id)
            item.setdefault("created_at", datetime.fromtimestamp(audio_path.stat().st_mtime, tz=timezone.utc).isoformat())
            versions.append(item)
        return sorted(versions, key=lambda item: item["generation"], reverse=True)


class MultitrackEngine:
    """Three feedback lineages sharing the monitor's selected SA3 backend."""

    def __init__(
        self,
        *,
        project_root: Path,
        output_dir: Path | None = None,
        backend: Any | None = None,
        accelerator_lock: threading.RLock | None = None,
        request_logger: Callable[[dict[str, Any]], None] | None = None,
        track_ids: tuple[str, ...] = TRACK_IDS,
        track_labels: dict[str, str] | None = None,
        track_config_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.backend = backend
        self.accelerator_lock = accelerator_lock or threading.RLock()
        self.request_logger = request_logger
        self.track_ids = tuple(track_ids)
        if not self.track_ids or any(track_id not in TRACK_IDS for track_id in self.track_ids):
            raise ValueError("track_ids must be a non-empty subset of the supported tracks")
        self.track_labels = {track_id: TRACK_LABELS[track_id] for track_id in self.track_ids}
        self.track_labels.update(track_labels or {})
        self.track_config_overrides = {
            track_id: dict(values)
            for track_id, values in (track_config_overrides or {}).items()
        }
        unknown_override_tracks = set(self.track_config_overrides) - set(self.track_ids)
        if unknown_override_tracks:
            raise ValueError(f"config overrides contain unknown tracks: {', '.join(sorted(unknown_override_tracks))}")
        for values in self.track_config_overrides.values():
            if "revision" in values:
                raise ValueError("track config overrides cannot set revision")
        self.output_dir = (output_dir or Path(__file__).parent / "output").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_path = self.output_dir / "multitrack_config.json"
        self.session = SessionConfig.from_file(self.session_path)
        if not self.session_path.exists():
            _atomic_json(self.session_path, asdict(self.session))
        self.tracks = {track_id: _TrackState(self, track_id) for track_id in self.track_ids}

        self._state_lock = asyncio.Lock()
        self._publish_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._closing = False

        self.running = False
        self.started_at: str | None = None
        self.active_track_id: str | None = None
        self.active_job_id: str | None = None
        self.last_error: dict[str, Any] | None = None

    def _track(self, track_id: str) -> _TrackState:
        if track_id not in self.tracks:
            raise ValueError(f"unknown track: {track_id}")
        return self.tracks[track_id]

    def note_client(self, track_id: str, client_ip: str) -> None:
        self._track(track_id).client_ip = client_ip or "unknown"

    def note_all_clients(self, client_ip: str) -> None:
        for track in self.tracks.values():
            track.client_ip = client_ip or "unknown"

    async def update_track(
        self,
        track_id: str,
        *,
        latch_for_next: bool = False,
        **changes: Any,
    ) -> TrackConfig:
        track = self._track(track_id)
        async with self._state_lock:
            cleaned = dict(changes)
            if "prompt_tags" in cleaned:
                if cleaned["prompt_tags"] is None:
                    raise ValueError("prompt_tags cannot be null")
                cleaned["prompt_tags"] = [
                    {"text": str(tag["text"]).strip(), "weight": tag.get("weight", "normal")}
                    for tag in cleaned["prompt_tags"]
                ]
            if "role_prompt" in cleaned:
                cleaned["role_prompt"] = str(cleaned["role_prompt"] or "").strip() or None
            if "negative_prompt" in cleaned:
                cleaned["negative_prompt"] = str(cleaned["negative_prompt"] or "").strip() or None
            base = track.pending_config or track.config
            candidate = replace(base, **cleaned, revision=base.revision + 1)
            candidate.validate()
            _atomic_json(track.config_path, asdict(candidate))
            track.failed = False
            track.last_error = None
            (track.output_dir / "last_error.json").unlink(missing_ok=True)
            if latch_for_next and track.phase == "generating":
                # Keep the configuration snapshot used by the active render
                # stable. The newest pending configuration is promoted after
                # that WAV has been published.
                track.pending_config = candidate
                return candidate

            track.pending_config = None
            track.config = candidate
            if latch_for_next:
                # The next normally scheduled generation will use this config.
                # Do not create an extra job or move its existing deadline.
                return candidate
            if self.running and not candidate.frozen:
                track.dirty = True
                if track.phase != "generating":
                    track.phase = "waiting"
                track.next_generation_timestamp = datetime.now(timezone.utc).timestamp()
                self._wake.set()
            elif candidate.frozen and track.phase != "generating":
                track.phase = "frozen"
                track.next_generation_timestamp = None
            return candidate

    async def update_session(self, *, latch_for_next: bool = False, **changes: Any) -> SessionConfig:
        async with self._state_lock:
            cleaned = dict(changes)
            mixer = {key: dict(value) for key, value in (self.session.mixer or {}).items()}
            if "mixer" in cleaned:
                for track_id, track_changes in cleaned.pop("mixer").items():
                    if track_id not in self.track_ids:
                        raise ValueError(f"unknown track in mixer: {track_id}")
                    mixer[track_id] = {**mixer[track_id], **track_changes}
            sidechain = dict(self.session.sidechain or {})
            if "sidechain" in cleaned:
                sidechain.update(cleaned.pop("sidechain"))
            reverb = dict(self.session.reverb or {})
            if "reverb" in cleaned:
                reverb.update(cleaned.pop("reverb"))
            delay = dict(self.session.delay or {})
            if "delay" in cleaned:
                delay.update(cleaned.pop("delay"))
            old_bpm = self.session.bpm
            candidate = replace(
                self.session,
                **cleaned,
                mixer=mixer,
                sidechain=sidechain,
                reverb=reverb,
                delay=delay,
                revision=self.session.revision + 1,
            )
            candidate.validate()
            if candidate.bpm != old_bpm and any(track.config.frozen for track in self.tracks.values()):
                raise ValueError("unfreeze every track before changing BPM")
            self.session = candidate
            _atomic_json(self.session_path, asdict(candidate))
            if candidate.bpm != old_bpm and self.running and not latch_for_next:
                now = datetime.now(timezone.utc).timestamp()
                for track in self.tracks.values():
                    if not track.config.frozen:
                        track.dirty = True
                        track.next_generation_timestamp = now
                self._wake.set()
            return candidate

    async def set_frozen(self, track_id: str, frozen: bool) -> TrackConfig:
        track = self._track(track_id)
        async with self._state_lock:
            base = track.pending_config or track.config
            if base.frozen == frozen and track.pending_config is None:
                return base
            candidate = replace(base, frozen=frozen, revision=base.revision + 1)
            candidate.validate()
            track.pending_config = None
            track.config = candidate
            _atomic_json(track.config_path, asdict(candidate))
            if frozen:
                track.dirty = False
                track.next_generation_timestamp = None
                if track.phase != "generating":
                    track.phase = "frozen"
            else:
                track.failed = False
                track.last_error = None
                track.dirty = self.running
                track.next_generation_timestamp = datetime.now(timezone.utc).timestamp() if self.running else None
                track.phase = "waiting" if self.running else "stopped"
                self._wake.set()
            return candidate

    async def freeze_all(self, frozen: bool) -> None:
        for track_id in self.track_ids:
            await self.set_frozen(track_id, frozen)

    async def start(self) -> bool:
        async with self._state_lock:
            if self.running:
                return False
            for track in self.tracks.values():
                track.config.validate()
            self.running = True
            self.started_at = _utc_now()
            self._closing = False
            now = datetime.now(timezone.utc).timestamp()
            for track in self.tracks.values():
                if track.config.frozen:
                    track.phase = "frozen"
                else:
                    track.phase = "waiting"
                    track.dirty = True
                    track.next_generation_timestamp = now
            if self._worker is None or self._worker.done():
                self._worker = asyncio.create_task(self._run(), name="sa3-multitrack-loop")
            self._wake.set()
            return True

    async def stop(self) -> bool:
        async with self._state_lock:
            was_running = self.running
            self.running = False
            for track in self.tracks.values():
                track.dirty = False
                track.next_generation_timestamp = None
                if track.phase not in {"generating", "frozen", "error"}:
                    track.phase = "stopped"
            self._wake.set()
            return was_running

    async def shutdown(self) -> None:
        self._closing = True
        await self.stop()
        worker = self._worker
        if worker is not None and not worker.done():
            try:
                await asyncio.wait_for(worker, timeout=6)
            except asyncio.TimeoutError:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

    def _global_phase(self) -> str:
        if self.active_track_id:
            return "generating"
        if self.running:
            return "active"
        return "stopped"

    @staticmethod
    def _promote_pending_unlocked(track: _TrackState) -> bool:
        if track.pending_config is None:
            return False
        track.config = track.pending_config
        track.pending_config = None
        return True

    async def status(self) -> dict[str, Any]:
        async with self._state_lock:
            tracks: dict[str, Any] = {}
            for track_id, track in self.tracks.items():
                loop_seconds = track.config.beats * 60.0 / self.session.bpm
                next_at = None
                if track.next_generation_timestamp is not None:
                    next_at = datetime.fromtimestamp(track.next_generation_timestamp, timezone.utc).isoformat()
                tracks[track_id] = {
                    "id": track_id,
                    "label": track.label,
                    "model": track.model,
                    "phase": track.phase,
                    "generation": track.generation,
                    "current_available": track.current_path.exists(),
                    "generation_started_at": track.generation_started_at,
                    "next_generation_at": next_at,
                    "active_job_id": track.active_job_id,
                    "config": {
                        **asdict(track.config),
                        "constructed_prompt": track.config.prompt(track_id, self.session.bpm),
                        "loop_seconds": loop_seconds,
                        "mutation_interval_seconds": self._mutation_period_seconds(track.config, self.session.bpm),
                    },
                    "pending_config": ({
                        **asdict(track.pending_config),
                        "constructed_prompt": track.pending_config.prompt(track_id, self.session.bpm),
                        "loop_seconds": track.pending_config.beats * 60.0 / self.session.bpm,
                        "mutation_interval_seconds": self._mutation_period_seconds(
                            track.pending_config, self.session.bpm
                        ),
                    } if track.pending_config else None),
                    "last_completed": dict(track.last_completed) if track.last_completed else None,
                    "introduction": self._public_introduction(track.introduction),
                    "last_error": dict(track.last_error) if track.last_error else None,
                }
            return {
                "running": self.running,
                "phase": self._global_phase(),
                "started_at": self.started_at,
                "active_track_id": self.active_track_id,
                "active_job_id": self.active_job_id,
                "session": asdict(self.session),
                "tracks": tracks,
                "last_error": dict(self.last_error) if self.last_error else None,
            }

    def list_versions(self, track_id: str, *, offset: int = 0, limit: int = 50) -> list[dict[str, Any]]:
        return self._track(track_id).versions[offset:offset + limit]

    def get_version(self, track_id: str, generation: int) -> dict[str, Any] | None:
        return next((item for item in self._track(track_id).versions if item["generation"] == generation), None)

    @staticmethod
    def _public_introduction(value: dict[str, Any] | None) -> dict[str, Any] | None:
        if not value:
            return None
        return {key: item for key, item in value.items() if key != "filename"}

    def version_path(self, track_id: str, generation: int) -> Path | None:
        track = self._track(track_id)
        item = self.get_version(track_id, generation)
        if item is None:
            return None
        path = track.versions_dir / Path(item["filename"]).name
        return path if path.exists() else None

    def current_path_for(self, track_id: str) -> Path:
        return self._track(track_id).current_path

    async def clear_versions(self, track_id: str) -> dict[str, Any]:
        track = self._track(track_id)
        async with self._publish_lock:
            return await self._clear_versions_unlocked(track)

    async def clear_all_versions(self) -> list[dict[str, Any]]:
        async with self._publish_lock:
            return [await self._clear_versions_unlocked(self.tracks[track_id]) for track_id in self.track_ids]

    async def clear_output(self, track_id: str) -> dict[str, Any]:
        """Remove one feedback parent and its history without resetting controls.

        Bumping the config revision invalidates a render that started before the
        clear, preventing it from republishing audio after the user removed it.
        """
        track = self._track(track_id)
        async with self._publish_lock:
            async with self._state_lock:
                base = track.pending_config or track.config
                candidate = replace(base, revision=base.revision + 1)
                track.pending_config = None
                track.config = candidate
                _atomic_json(track.config_path, asdict(candidate))
                generation = track.generation
                was_available = track.current_path.exists()
                track.last_completed = None
                track.last_error = None
                track.failed = False
                track.versions.clear()
                introduction = track.introduction
                track.introduction = None
                track.dirty = bool(self.running and not candidate.frozen)
                track.next_generation_timestamp = (
                    datetime.now(timezone.utc).timestamp() if track.dirty else None
                )
                track.phase = "waiting" if track.dirty else ("frozen" if candidate.frozen else "stopped")

            freed = 0
            removed = 0
            targets = [
                track.current_path,
                track.current_metadata_path,
                track.output_dir / "last_error.json",
                track.introduction_state_path,
            ]
            if introduction:
                targets.append(track.output_dir / introduction["filename"])
            targets.extend(path for path in track.versions_dir.iterdir() if path.is_file())
            targets.extend(track.output_dir.glob("introduction-*.wav"))
            for path in targets:
                try:
                    freed += path.stat().st_size
                    path.unlink()
                    removed += 1
                except FileNotFoundError:
                    continue
            _atomic_json(track.lineage_state_path, {
                "generation": generation,
                "output_cleared_at": _utc_now(),
            })
            if track.dirty:
                self._wake.set()
            return {
                "track_id": track.id,
                "files_removed": removed,
                "bytes_freed": freed,
                "current_removed": was_available,
                "generation": generation,
                "settings_preserved": True,
                "restart_queued": track.dirty,
            }

    async def clear_all_outputs(self) -> list[dict[str, Any]]:
        results = []
        for track_id in self.track_ids:
            results.append(await self.clear_output(track_id))
        return results

    async def _clear_versions_unlocked(self, track: _TrackState) -> dict[str, Any]:
        async with self._state_lock:
            current_record = dict(track.last_completed) if track.last_completed else None
            generation = track.generation
        freed = 0
        removed = 0
        for path in track.versions_dir.iterdir():
            if not path.is_file():
                continue
            try:
                freed += path.stat().st_size
                path.unlink()
                removed += 1
            except FileNotFoundError:
                continue
        if current_record and track.current_path.exists():
            _atomic_json(track.current_metadata_path, current_record)
        _atomic_json(track.lineage_state_path, {
            "generation": generation,
            "history_cleared_at": _utc_now(),
        })
        async with self._state_lock:
            track.versions.clear()
        return {
            "track_id": track.id,
            "files_removed": removed,
            "bytes_freed": freed,
            "current_preserved": track.current_path.exists(),
            "generation": generation,
        }

    async def import_reference(
        self,
        track_id: str,
        source_path: Path,
        original_name: str | None = None,
        *,
        mode: str = "auto",
        blend: float = 0.25,
        generations: int = 4,
    ) -> dict[str, Any]:
        track = self._track(track_id)
        if mode not in {"auto", "introduce", "replace"}:
            raise ValueError("reference mode must be auto, introduce, or replace")
        if not 0.01 <= blend <= 0.75:
            raise ValueError("introduction blend must be between 0.01 and 0.75")
        if generations not in {1, 2, 4, 8}:
            raise ValueError("introduction generations must be 1, 2, 4, or 8")
        async with self._state_lock:
            bpm = self.session.bpm
            config = track.config
            generation = track.generation + 1
            loop_seconds = config.beats * 60.0 / bpm
            job_id = f"import-{uuid.uuid4().hex[:8]}"
            has_current = track.current_path.exists()
        normalized = track.work_dir / f"{job_id}.wav"
        await asyncio.to_thread(self._conform_reference, source_path, normalized, loop_seconds)
        resolved_mode = "introduce" if mode == "introduce" or (mode == "auto" and has_current) else "replace"
        if resolved_mode == "introduce" and has_current:
            introduction_id = uuid.uuid4().hex[:12]
            introduction_path = track.output_dir / f"introduction-{introduction_id}.wav"
            old_path: Path | None = None
            old_may_be_rendering = False
            async with self._publish_lock:
                os.replace(normalized, introduction_path)
                async with self._state_lock:
                    if track.introduction:
                        old_path = track.output_dir / track.introduction["filename"]
                        old_may_be_rendering = track.phase == "generating"
                    track.introduction = {
                        "id": introduction_id,
                        "filename": introduction_path.name,
                        "original_filename": original_name,
                        "blend": float(blend),
                        "generations_total": int(generations),
                        "generations_remaining": int(generations),
                        "queued_at": _utc_now(),
                    }
                    _atomic_json(track.introduction_state_path, track.introduction)
                    track.last_error = None
                    track.failed = False
                    if self.running and not track.config.frozen:
                        track.dirty = True
                        if track.phase != "generating":
                            track.phase = "waiting"
                            track.next_generation_timestamp = datetime.now(timezone.utc).timestamp()
                        self._wake.set()
            if old_path and old_path != introduction_path and not old_may_be_rendering:
                old_path.unlink(missing_ok=True)
            return {
                "queued": True,
                "mode": "introduction",
                **self._public_introduction(track.introduction),
            }

        old_introduction: dict[str, Any] | None = None
        completed = datetime.now(timezone.utc)
        record = {
            "track_id": track.id,
            "generation": generation,
            "filename": f"generation_{generation:06d}.wav",
            "created_at": completed.isoformat(),
            "elapsed_seconds": 0.0,
            "job_id": job_id,
            "mode": "uploaded-reference",
            "model": config.model,
            "prompt": config.prompt(track.id, bpm),
            "prompt_tags": config.prompt_tags,
            "negative_prompt": config.negative_prompt,
            "bpm": bpm,
            "beats": config.beats,
            "loop_seconds": loop_seconds,
            "mutation": config.mutation,
            "effective_noise": 0.0,
            "cfg": config.cfg,
            "steps": config.steps,
            "seed_mode": config.seed_mode,
            "seed": None,
            "config_revision": config.revision,
            "original_filename": original_name,
        }
        async with self._publish_lock:
            async with self._state_lock:
                old_introduction = track.introduction
                track.introduction = None
                track.introduction_state_path.unlink(missing_ok=True)
                generation = track.generation + 1
                record["generation"] = generation
                record["filename"] = f"generation_{generation:06d}.wav"
            await asyncio.to_thread(self._publish, track, normalized, record)
            async with self._state_lock:
                track.generation = generation
                track.versions.insert(0, record)
                track.last_completed = record
                track.last_error = None
                track.failed = False
                if self.running and not track.config.frozen:
                    track.phase = "waiting"
                    track.dirty = False
                    now = datetime.now(timezone.utc).timestamp()
                    period = self._mutation_period_seconds(track.config, self.session.bpm)
                    track.next_generation_timestamp = now + period
                    self._wake.set()
        if old_introduction:
            (track.output_dir / old_introduction["filename"]).unlink(missing_ok=True)
        return record

    @staticmethod
    def _conform_reference(source_path: Path, destination: Path, seconds: float) -> None:
        try:
            with wave.open(str(source_path), "rb") as source:
                channels = source.getnchannels()
                sample_width = source.getsampwidth()
                sample_rate = source.getframerate()
                compression = source.getcomptype()
                if compression != "NONE":
                    raise ValueError("the uploaded WAV must contain uncompressed PCM audio")
                if channels < 1 or sample_width < 1 or sample_rate < 1:
                    raise ValueError("the uploaded WAV has invalid audio parameters")
                target_frames = max(1, round(seconds * sample_rate))
                with wave.open(str(destination), "wb") as output:
                    output.setnchannels(channels)
                    output.setsampwidth(sample_width)
                    output.setframerate(sample_rate)
                    remaining = target_frames
                    while remaining:
                        frames = source.readframes(min(remaining, 65536))
                        if not frames:
                            break
                        frame_count = len(frames) // (channels * sample_width)
                        output.writeframesraw(frames[:frame_count * channels * sample_width])
                        remaining -= frame_count
                    if remaining:
                        silence = b"\x00" * channels * sample_width
                        while remaining:
                            count = min(remaining, 65536)
                            output.writeframesraw(silence * count)
                            remaining -= count
                    output.writeframes(b"")
        except wave.Error as exc:
            destination.unlink(missing_ok=True)
            raise ValueError("the uploaded reference must be a valid WAV file") from exc

    @staticmethod
    def _read_pcm_float(path: Path) -> tuple[np.ndarray, int]:
        try:
            with wave.open(str(path), "rb") as audio:
                channels = audio.getnchannels()
                width = audio.getsampwidth()
                sample_rate = audio.getframerate()
                frames = audio.getnframes()
                payload = audio.readframes(frames)
        except wave.Error as exc:
            raise ValueError(f"could not decode PCM WAV {path.name}") from exc
        if width == 1:
            values = (np.frombuffer(payload, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 2:
            values = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 3:
            raw = np.frombuffer(payload, dtype=np.uint8)
            if raw.size % 3:
                raise ValueError(f"invalid 24-bit PCM payload in {path.name}")
            triples = raw.reshape(-1, 3).astype(np.int32)
            integers = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
            integers = np.where(integers & 0x800000, integers - 0x1000000, integers)
            values = integers.astype(np.float32) / 8388608.0
        elif width == 4:
            values = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported {width * 8}-bit PCM WAV: {path.name}")
        if channels < 1 or values.size % channels:
            raise ValueError(f"invalid channel data in {path.name}")
        return values.reshape(-1, channels), sample_rate

    @staticmethod
    def _match_audio_shape(audio: np.ndarray, frames: int, channels: int) -> np.ndarray:
        if audio.shape[1] != channels:
            if channels == 1:
                audio = audio.mean(axis=1, keepdims=True)
            elif audio.shape[1] == 1:
                audio = np.repeat(audio, channels, axis=1)
            elif audio.shape[1] > channels:
                audio = audio[:, :channels]
            else:
                audio = np.resize(audio, (audio.shape[0], channels))
        if audio.shape[0] == frames:
            return audio
        if audio.shape[0] < 2:
            return np.repeat(audio[:1], frames, axis=0) if audio.shape[0] else np.zeros((frames, channels), np.float32)
        source_positions = np.linspace(0.0, 1.0, audio.shape[0], endpoint=False)
        target_positions = np.linspace(0.0, 1.0, frames, endpoint=False)
        return np.stack([
            np.interp(target_positions, source_positions, audio[:, channel])
            for channel in range(channels)
        ], axis=1).astype(np.float32)

    @classmethod
    def _mix_introduction(cls, current_path: Path, new_path: Path, destination: Path, blend: float) -> None:
        current, sample_rate = cls._read_pcm_float(current_path)
        introduced, _ = cls._read_pcm_float(new_path)
        introduced = cls._match_audio_shape(introduced, current.shape[0], current.shape[1])
        current_rms = float(np.sqrt(np.mean(np.square(current), dtype=np.float64)))
        introduced_rms = float(np.sqrt(np.mean(np.square(introduced), dtype=np.float64)))
        if current_rms > 1e-6 and introduced_rms > 1e-6:
            introduced *= float(np.clip(current_rms / introduced_rms, 0.25, 4.0))
        mixed = np.sqrt(1.0 - blend) * current + np.sqrt(blend) * introduced
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > 0.98:
            mixed *= 0.98 / peak
        pcm = np.round(np.clip(mixed, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(current.shape[1])
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(pcm.tobytes())

    def _next_track(self) -> tuple[_TrackState | None, float | None]:
        now = datetime.now(timezone.utc).timestamp()
        candidates = [
            track for track in self.tracks.values()
            if not track.config.frozen and not track.failed and (
                track.dirty or (
                    track.next_generation_timestamp is not None
                    and track.next_generation_timestamp <= now
                )
            )
        ]
        if candidates:
            candidates.sort(key=lambda track: (
                0 if track.dirty else 1,
                track.next_generation_timestamp or 0,
                self.track_ids.index(track.id),
            ))
            return candidates[0], 0
        due = [
            track.next_generation_timestamp for track in self.tracks.values()
            if not track.config.frozen and not track.failed and track.next_generation_timestamp is not None
        ]
        return None, max(0.0, min(due) - now) if due else None

    @staticmethod
    def _mutation_period_seconds(config: TrackConfig, bpm: float) -> float:
        return config.beats * 60.0 / bpm * config.mutation_interval_loops

    @staticmethod
    def _advance_mutation_deadline(scheduled_at: float, period: float, now: float) -> float:
        """Return the next start-to-start deadline without accumulating backlog."""
        return max(scheduled_at + period, now)

    async def _run(self) -> None:
        try:
            while True:
                scheduled_at = None
                async with self._state_lock:
                    if not self.running:
                        self.active_track_id = None
                        self.active_job_id = None
                        return
                    track, delay = self._next_track()
                    self._wake.clear()
                    if track is not None:
                        scheduled_at = track.next_generation_timestamp
                        track.dirty = False
                        track.next_generation_timestamp = None
                if track is None:
                    try:
                        if delay is None:
                            await self._wake.wait()
                        else:
                            await asyncio.wait_for(self._wake.wait(), timeout=delay)
                    except asyncio.TimeoutError:
                        pass
                    continue
                try:
                    record = await self._generate_once(track)
                except GenerationError as exc:
                    async with self._state_lock:
                        self._promote_pending_unlocked(track)
                        track.phase = "error"
                        track.failed = True
                        track.last_error = exc.detail
                        track.active_job_id = None
                        track.generation_started_at = None
                        self.active_track_id = None
                        self.active_job_id = None
                        self.last_error = exc.detail
                    continue
                except Exception as exc:
                    detail = {
                        "at": _utc_now(),
                        "track_id": track.id,
                        "job_id": track.active_job_id,
                        "message": f"internal loop error: {type(exc).__name__}: {exc}",
                        "returncode": None,
                    }
                    _atomic_json(track.output_dir / "last_error.json", detail)
                    async with self._state_lock:
                        self._promote_pending_unlocked(track)
                        track.phase = "error"
                        track.failed = True
                        track.last_error = detail
                        track.active_job_id = None
                        track.generation_started_at = None
                        self.active_track_id = None
                        self.active_job_id = None
                        self.last_error = detail
                    continue

                async with self._state_lock:
                    self._promote_pending_unlocked(track)
                    if record is not None:
                        track.last_completed = record
                        track.last_error = None
                    track.active_job_id = None
                    track.generation_started_at = None
                    self.active_track_id = None
                    self.active_job_id = None
                    if track.config.frozen:
                        track.phase = "frozen"
                        track.next_generation_timestamp = None
                    elif self.running:
                        track.phase = "waiting"
                        if track.dirty:
                            # A setting changed during this render. Keep the
                            # immediate replacement already queued by update_track().
                            self._wake.set()
                        else:
                            now = datetime.now(timezone.utc).timestamp()
                            period = self._mutation_period_seconds(track.config, self.session.bpm)
                            track.next_generation_timestamp = self._advance_mutation_deadline(
                                scheduled_at if scheduled_at is not None else now,
                                period,
                                now,
                            )
                    else:
                        track.phase = "stopped"
                        track.next_generation_timestamp = None
        except asyncio.CancelledError:
            async with self._state_lock:
                self.running = False
                self.active_track_id = None
                self.active_job_id = None
            raise

    def _seed_for(self, config: TrackConfig, generation: int) -> int:
        if config.seed_mode == "fixed":
            assert config.seed is not None
            return config.seed
        if config.seed_mode == "increment":
            assert config.seed is not None
            return config.seed + generation - 1
        return random.SystemRandom().randint(0, 2**31 - 1)

    async def _generate_once(self, track: _TrackState) -> dict[str, Any] | None:
        async with self._state_lock:
            config = track.config
            bpm = self.session.bpm
            generation = track.generation + 1
            parent_generation = track.generation
            seed = self._seed_for(config, generation)
            job_id = uuid.uuid4().hex[:12]
            track.phase = "generating"
            track.active_job_id = job_id
            track.generation_started_at = _utc_now()
            self.active_track_id = track.id
            self.active_job_id = job_id
            introduction = dict(track.introduction) if track.introduction else None

        started = datetime.now(timezone.utc)
        input_path = track.current_path if track.current_path.exists() else None
        if self.backend is None:
            raise GenerationError({
                "at": _utc_now(), "track_id": track.id, "job_id": job_id,
                "message": "DRIFT has no configured SA3 backend", "returncode": None,
            })
        introduction_details: dict[str, Any] | None = None
        mixed_input_path: Path | None = None
        backend_input_path = input_path
        if input_path and introduction:
            total = max(1, int(introduction["generations_total"]))
            remaining = max(1, int(introduction["generations_remaining"]))
            step = min(total, total - remaining + 1)
            applied_blend = float(introduction["blend"]) * step / total
            mixed_input_path = track.work_dir / f"{job_id}-introduction.wav"
            try:
                await asyncio.to_thread(
                    self._mix_introduction,
                    input_path,
                    track.output_dir / introduction["filename"],
                    mixed_input_path,
                    applied_blend,
                )
            except (OSError, ValueError) as exc:
                mixed_input_path.unlink(missing_ok=True)
                raise GenerationError({
                    "at": _utc_now(), "track_id": track.id, "job_id": job_id,
                    "message": f"could not prepare introduced audio: {exc}", "returncode": None,
                }) from exc
            backend_input_path = mixed_input_path
            introduction_details = {
                "id": introduction["id"],
                "original_filename": introduction.get("original_filename"),
                "step": step,
                "generations_total": total,
                "blend": round(applied_blend, 6),
                "target_blend": float(introduction["blend"]),
            }
        mode = "audio-introduction" if introduction_details else ("audio-to-audio" if input_path else "text-to-audio-bootstrap")
        work_path = track.work_dir / f"{job_id}.wav"
        loop_seconds = config.beats * 60.0 / bpm
        prompt = config.prompt(track.id, bpm)
        request = {
            "ip": track.client_ip,
            "prompt": prompt,
            "event": "drift_generate",
            "track": track.id,
            "negative_prompt": config.negative_prompt,
            "model": config.model,
            "seconds": loop_seconds,
            "steps": config.steps,
            "seed": seed,
            "cfg": config.cfg,
            "sigma_max": config.mutation if input_path else 1.0,
            "mutation_interval_loops": config.mutation_interval_loops,
            "mode": "audio" if input_path else "text",
            "job_id": job_id,
            "introduction": introduction_details,
        }
        if self.request_logger:
            self.request_logger(request)

        def render() -> None:
            def emit(value: dict[str, Any]) -> None:
                # DRIFT's compact UI polls status. The selected backend still
                # emits the shared monitor event contract for diagnostics.
                if value.get("type") == "sampler_step":
                    request["sampler_step"] = value.get("step")
                    request["sampler_total"] = value.get("total")

            with self.accelerator_lock:
                self.backend.generate(
                    prompt=prompt,
                    negative_prompt=config.negative_prompt or "",
                    seconds=float(loop_seconds),
                    steps=int(config.steps),
                    seed=int(seed),
                    cfg=float(config.cfg),
                    apg=1.0,
                    sigma_max=float(config.mutation if input_path else 1.0),
                    dit_name=config.model,
                    decoder_name="same-l" if config.model == "medium" else "same-s",
                    mode="audio" if input_path else "text",
                    input_audio=str(backend_input_path) if backend_input_path else None,
                    output_path=str(work_path),
                    emit=emit,
                )
        try:
            await asyncio.to_thread(render)
        except (OSError, asyncio.CancelledError) as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            async with self._state_lock:
                stale = track.config.revision != config.revision or track.generation != parent_generation
            if stale:
                work_path.unlink(missing_ok=True)
                return None
            raise GenerationError({
                "at": _utc_now(),
                "track_id": track.id,
                "job_id": job_id,
                "message": f"could not start SA3: {exc}",
                "returncode": None,
            }) from exc
        except Exception as exc:
            async with self._state_lock:
                stale = track.config.revision != config.revision or track.generation != parent_generation
            if stale:
                work_path.unlink(missing_ok=True)
                return None
            work_path.unlink(missing_ok=True)
            detail = {
                "at": _utc_now(),
                "track_id": track.id,
                "job_id": job_id,
                "message": f"SA3 backend failed: {type(exc).__name__}: {exc}",
                "returncode": None,
            }
            _atomic_json(track.output_dir / "last_error.json", detail)
            raise GenerationError(detail) from exc
        finally:
            if mixed_input_path:
                mixed_input_path.unlink(missing_ok=True)

        try:
            self._validate_wav(work_path)
        except (OSError, wave.Error, ValueError) as exc:
            work_path.unlink(missing_ok=True)
            detail = {
                "at": _utc_now(),
                "track_id": track.id,
                "job_id": job_id,
                "message": f"SA3 did not produce a valid WAV: {exc}",
                "returncode": None,
            }
            _atomic_json(track.output_dir / "last_error.json", detail)
            raise GenerationError(detail) from exc

        completed = datetime.now(timezone.utc)
        record = {
            "track_id": track.id,
            "generation": generation,
            "filename": f"generation_{generation:06d}.wav",
            "created_at": completed.isoformat(),
            "elapsed_seconds": round((completed - started).total_seconds(), 3),
            "job_id": job_id,
            "mode": mode,
            "model": config.model,
            "prompt": prompt,
            "prompt_tags": config.prompt_tags,
            "negative_prompt": config.negative_prompt,
            "bpm": bpm,
            "beats": config.beats,
            "loop_seconds": loop_seconds,
            "mutation": config.mutation,
            "mutation_interval_loops": config.mutation_interval_loops,
            "effective_noise": config.mutation if input_path else 1.0,
            "cfg": config.cfg,
            "steps": config.steps,
            "seed_mode": config.seed_mode,
            "seed": seed,
            "config_revision": config.revision,
            "parent_generation": parent_generation,
            "introduction": introduction_details,
        }
        finished_introduction_path: Path | None = None
        async with self._publish_lock:
            async with self._state_lock:
                stale_before_publish = (
                    track.config.frozen
                    or track.generation != parent_generation
                    or track.config.revision != config.revision
                )
            if stale_before_publish:
                work_path.unlink(missing_ok=True)
                return None
            await asyncio.to_thread(self._publish, track, work_path, record)
            async with self._state_lock:
                track.generation = generation
                track.versions.insert(0, record)
                if introduction_details and track.introduction and track.introduction.get("id") == introduction_details["id"]:
                    remaining = int(track.introduction["generations_remaining"]) - 1
                    if remaining <= 0:
                        finished_introduction_path = track.output_dir / track.introduction["filename"]
                        track.introduction = None
                        track.introduction_state_path.unlink(missing_ok=True)
                    else:
                        track.introduction["generations_remaining"] = remaining
                        _atomic_json(track.introduction_state_path, track.introduction)
                elif introduction_details:
                    finished_introduction_path = track.output_dir / introduction["filename"]
                self._promote_pending_unlocked(track)
        if finished_introduction_path:
            finished_introduction_path.unlink(missing_ok=True)
        return record

    @staticmethod
    def _validate_wav(path: Path) -> None:
        if not path.exists() or path.stat().st_size <= 44:
            raise ValueError("output file is missing or empty")
        with wave.open(str(path), "rb") as audio:
            if audio.getnframes() < 1:
                raise ValueError("output has no audio frames")

    @staticmethod
    def _publish(track: _TrackState, work_path: Path, record: dict[str, Any]) -> None:
        version_path = track.versions_dir / record["filename"]
        metadata_path = version_path.with_suffix(".json")
        os.replace(work_path, version_path)
        current_tmp = track.output_dir / f".current.{uuid.uuid4().hex}.wav"
        shutil.copyfile(version_path, current_tmp)
        os.replace(current_tmp, track.current_path)
        _atomic_json(metadata_path, record)
        _atomic_json(track.current_metadata_path, record)
        _atomic_json(track.lineage_state_path, {"generation": record["generation"]})
        (track.output_dir / "last_error.json").unlink(missing_ok=True)


# Retained for imports used by the original project and third-party scripts.
LoopEngine = MultitrackEngine
