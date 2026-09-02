from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
import zipfile
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from backends.base import BackendUnavailable, CONTRACT_VERSION
from backends.base import heatmap_preview as compact_heatmap
from backends.factory import create_backend, diagnostics_json
from drift_api import create_app as create_drift_app
from drift_engine import MultitrackEngine
from same_lab import (
    LATENT_FRAMES_PER_SECOND,
    OPERATIONS,
    PROFILE_CHANNELS,
    SAMPLE_RATE as SAME_SAMPLE_RATE,
    WAVEFORMS,
    LatentSession,
    apply_profile,
    channel_stats,
    difference_audio,
    intervene,
    mix_channels,
    new_session_id,
    parse_channels,
    seconds_to_frames,
    time_crossfade,
    validate_bank,
)
from same_osc import same_osc

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
RUN_ROOT = HERE / "output" / "monitor"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SA3 Graphical Pipeline Monitor", version="1.6.0")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@dataclass
class RunState:
    run_id: str
    run_dir: Path
    config: dict
    events: list[dict] = field(default_factory=list)
    status: str = "queued"
    created: float = field(default_factory=time.time)
    finished: Optional[float] = None
    output_name: Optional[str] = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    client_ip: str = "unknown"
    condition: threading.Condition = field(default_factory=threading.Condition)

    def emit(self, event: dict):
        event = {"run_id": self.run_id, "contract_version": CONTRACT_VERSION, **event}
        derived = []
        if event.get("heatmap") is not None and event.get("stats") is not None and event.get("type") != "tensor_preview":
            derived.append({
                "run_id": self.run_id,
                "contract_version": CONTRACT_VERSION,
                "type": "tensor_preview",
                "time": event.get("time", time.time()),
                "source": event.get("type"),
                "stats": event["stats"],
                "heatmap": event["heatmap"],
                "memory_bytes": event.get("memory_bytes", 0),
            })
        if event.get("type") == "sampler_step":
            derived.append({
                "run_id": self.run_id,
                "contract_version": CONTRACT_VERSION,
                "type": "metrics",
                "time": event.get("time", time.time()),
                **{k: event.get(k) for k in ("step", "total", "sigma", "ms", "stats", "memory_bytes")},
            })
        if event.get("type") == "output_audio":
            derived.append({
                "run_id": self.run_id,
                "contract_version": CONTRACT_VERSION,
                "type": "audio_ready",
                "time": event.get("time", time.time()),
                "output_url": f"/api/output/{self.run_id}/generated.wav",
                "sample_rate": event.get("sample_rate"),
                "stats": event.get("stats"),
                "waveform": event.get("waveform"),
            })
        with self.condition:
            self.events.append(event)
            self.events.extend(derived)
            self.condition.notify_all()


runs: dict[str, RunState] = {}
runs_lock = threading.Lock()
work_q: queue.Queue[str] = queue.Queue()
engine = None
accelerator_lock = threading.RLock()
same_sessions: dict[str, LatentSession] = {}
same_sessions_lock = threading.Lock()


def _drift_terminal_log(payload: dict):
    print("DRIFT_REQUEST " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def _loop_mutator_terminal_log(payload: dict):
    print("LOOP_MUTATOR_REQUEST " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


drift_loop = MultitrackEngine(
    project_root=HERE,
    output_dir=HERE / "output" / "drift",
    backend=None,
    accelerator_lock=accelerator_lock,
    request_logger=_drift_terminal_log,
)
app.mount("/api/drift", create_drift_app(drift_loop))

loop_mutator = MultitrackEngine(
    project_root=HERE,
    output_dir=HERE / "output" / "loop_mutator",
    backend=None,
    accelerator_lock=accelerator_lock,
    request_logger=_loop_mutator_terminal_log,
    track_ids=("melodic",),
    track_labels={"melodic": "Loop"},
    # These fields are intentionally absent from the compact Loop Mutator UI.
    # Clear legacy/default values so no hidden text is added to generation.
    track_config_overrides={
        "melodic": {"role_prompt": None, "negative_prompt": None},
    },
)
app.mount(
    "/api/loop-mutator",
    create_drift_app(
        loop_mutator,
        api_prefix="/api/loop-mutator",
        title="Loop Mutator",
        description="Single-loop Stable Audio 3 mutation engine.",
        interface_url="/loop-mutator",
    ),
)


def _terminal_log(event_name: str, state: RunState, **extra):
    payload = {
        "event": event_name,
        "ip": state.client_ip,
        "prompt": state.config.get("prompt", ""),
        "negative_prompt": state.config.get("negative_prompt", ""),
        "run_id": state.run_id,
        "mode": state.config.get("mode"),
        "model": state.config.get("dit_name"),
        "decoder": state.config.get("decoder_name"),
        "seconds": state.config.get("seconds"),
        "steps": state.config.get("steps"),
        "seed": state.config.get("seed"),
        "cfg": state.config.get("cfg"),
        "apg": state.config.get("apg"),
        "sigma_max": state.config.get("sigma_max"),
        **extra,
    }
    print("SA3_REQUEST " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def configure_backend(selected_engine, output_dir: str | Path | None = None):
    """Install a backend. Kept explicit so tests can use a hardware-free stub."""
    global engine, RUN_ROOT
    engine = selected_engine
    if output_dir is not None:
        RUN_ROOT = Path(output_dir).expanduser().resolve()
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
    app.state.engine = selected_engine
    app.state.run_root = RUN_ROOT
    drift_loop.backend = selected_engine
    loop_mutator.backend = selected_engine


def worker_loop():
    while True:
        run_id = work_q.get()
        try:
            with runs_lock:
                state = runs.get(run_id)
            if state is None:
                continue
            state.status = "running"
            _terminal_log("started", state, waiting=work_q.qsize())
            state.emit({"type": "status", "status": "running", "time": time.time()})
            cfg = dict(state.config)
            output_name = "generated.wav"
            output_path = state.run_dir / output_name
            if engine is None:
                raise RuntimeError("No backend configured")
            with accelerator_lock:
                engine.generate(output_path=str(output_path), emit=state.emit, **cfg)
            state.output_name = output_name
            state.status = "complete"
            state.finished = time.time()
            state.emit({
                "type": "status",
                "status": "complete",
                "time": state.finished,
                "output_url": f"/api/output/{state.run_id}/{output_name}",
            })
            _terminal_log("complete", state, output=str(output_path), elapsed_seconds=state.finished - state.created)
        except Exception as exc:
            state.status = "error"
            state.finished = time.time()
            state.error = f"{type(exc).__name__}: {exc}"
            state.error_traceback = traceback.format_exc()

            # Always make backend failures visible in the terminal that launched
            # sa3-monitor.  v0.1 only sent these through SSE, which made a failed
            # run unnecessarily hard to diagnose after a browser refresh/restart.
            print("\n" + "=" * 80, flush=True)
            print(f"SA3 MONITOR ERROR — run {state.run_id}", flush=True)
            print(state.error_traceback, flush=True)
            print("=" * 80 + "\n", flush=True)

            # Persist the failure alongside the run files so it survives a server
            # restart and can be inspected without racing the event stream.
            try:
                (state.run_dir / "error.txt").write_text(
                    f"Run: {state.run_id}\nError: {state.error}\n\n{state.error_traceback}",
                    encoding="utf-8",
                )
            except Exception:
                traceback.print_exc()

            state.emit({
                "type": "error",
                "message": state.error,
                "traceback": state.error_traceback,
                "time": state.finished,
            })
            state.emit({"type": "status", "status": "error", "time": state.finished})
            _terminal_log("error", state, error=state.error)
        finally:
            work_q.task_done()


threading.Thread(target=worker_loop, name="sa3-monitor-worker", daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/monitor", response_class=HTMLResponse)
def monitor_page():
    """Single-run pipeline monitor, sweeps, playlist and downloads."""
    return (STATIC / "monitor.html").read_text(encoding="utf-8")


@app.get("/sfx-matrix", response_class=HTMLResponse)
def sfx_matrix():
    """Dedicated Small SFX prompt-variation interface."""
    return (STATIC / "sfx-matrix.html").read_text(encoding="utf-8")


@app.get("/same-lab", response_class=HTMLResponse)
def same_lab_page():
    """Backend-neutral SAME latent editor."""
    return (STATIC / "same-lab.html").read_text(encoding="utf-8")


@app.get("/drift", response_class=HTMLResponse)
def drift_page():
    """Continuous three-track SA3 feedback instrument."""
    return (STATIC / "drift" / "index.html").read_text(encoding="utf-8")


@app.get("/loop-mutator", response_class=HTMLResponse)
def loop_mutator_page():
    """Single-track SA3 feedback loop with waveform and spectrum views."""
    return (STATIC / "loop-mutator.html").read_text(encoding="utf-8")


@app.get("/api/info")
def info():
    if engine is None:
        raise HTTPException(503, "backend not configured")
    backend_info = engine.diagnostics().as_dict()
    return {
        "contract_version": CONTRACT_VERSION,
        "backend": backend_info,
        "sample_rate": engine.sample_rate,
        "max_seconds": engine.max_seconds,
        "dit_choices": list(engine.dit_choices),
        "decoder_choices": ["auto", *engine.decoder_choices],
        "defaults": {
            "dit": "sm-music",
            "decoder": "auto",
            "seconds": 10,
            "steps": 8,
            "cfg": 1.0,
            "apg": 1.0,
            "sigma_max": 1.0,
        },
    }


def _same_session(session_id: str) -> LatentSession:
    if not re.fullmatch(r"[0-9a-f]{16}", session_id or ""):
        raise HTTPException(400, "invalid SAME lab session id")
    with same_sessions_lock:
        session = same_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "SAME lab session not found; encode Source A again")
    return session


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAME_SAMPLE_RATE):
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("audio must have channels x samples shape")
    if value.shape[0] > 2 and value.shape[1] <= 2:
        value = value.T
    value = np.clip(value[:2], -1.0, 1.0)
    pcm = np.rint(value.T * 32767.0).astype("<i2", copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(int(value.shape[0]))
        output.setsampwidth(2)
        output.setframerate(int(sample_rate))
        output.writeframes(pcm.tobytes())


def _same_summary(session: LatentSession, status: str) -> dict:
    revision = int(session.updated * 1000)
    delta = session.current - session.original
    return {
        "session_id": session.session_id,
        "codec": session.codec,
        "backend": engine.name if engine else "unconfigured",
        "duration": session.duration,
        "target_samples": session.target_samples,
        "latent_shape": list(session.current.shape),
        "frames_per_second": LATENT_FRAMES_PER_SECOND,
        "status": status,
        "original_heatmap": compact_heatmap(session.original, rows=128, cols=320),
        "current_heatmap": compact_heatmap(session.current, rows=128, cols=320),
        "delta_heatmap": compact_heatmap(delta, rows=128, cols=320),
        "stats": channel_stats(session.current),
        "audio": {
            "baseline": f"/api/same/output/{session.session_id}/baseline.wav?v={revision}",
            "current": f"/api/same/output/{session.session_id}/current.wav?v={revision}",
            "difference": f"/api/same/output/{session.session_id}/difference.wav?v={revision}",
            "source_b": f"/api/same/output/{session.session_id}/source_b.wav?v={revision}" if session.source_b is not None else None,
        },
        "latent_url": f"/api/same/output/{session.session_id}/current.npy?v={revision}",
        "has_source_b": session.source_b is not None,
    }


def _same_commit(session: LatentSession, audio: np.ndarray, status: str) -> dict:
    difference = difference_audio(audio, session.baseline_audio)
    _write_wav(session.directory / "current.wav", audio)
    _write_wav(session.directory / "difference.wav", difference)
    np.save(session.directory / "current.npy", session.current.astype(np.float32, copy=False))
    session.updated = time.time()
    return _same_summary(session, status)


def _same_apply_payload(session: LatentSession, payload: dict) -> dict:
    """Apply one browser or OSC edit through the same backend-neutral path."""
    if engine is None:
        raise RuntimeError("backend not configured")
    action = str(payload.get("action", "")).strip().lower()
    with session.lock:
        if action == "reset":
            session.current = session.original.copy()
            audio = session.baseline_audio.copy()
            status = "Working latent reset to the original Source A encoding."
        elif action == "intervention":
            source = session.current if bool(payload.get("cumulative", False)) else session.original
            channels = parse_channels(str(payload.get("channels", "0")))
            start, end = seconds_to_frames(payload.get("start", 0), payload.get("end", session.duration), source.shape[1], session.duration)
            operation = str(payload.get("operation", OPERATIONS[0]))
            session.current = intervene(
                source, channels, start, end, operation,
                amount=float(payload.get("amount", 0)), seed=int(payload.get("seed", 0)),
                lfo_period=float(payload.get("lfo_period", 16)), lfo_depth=float(payload.get("lfo_depth", 1)),
                lfo_waveform=str(payload.get("lfo_waveform", "Sine")), lfo_phase=float(payload.get("lfo_phase", 0)),
                frame_displacement=int(payload.get("frame_displacement", 1)), lfo_invert=bool(payload.get("lfo_invert", False)),
            )
            audio = None
            status = f"Applied {operation} to {len(channels)} channel(s), frames {start}:{end}."
        elif action == "mix":
            if session.source_b is None:
                raise ValueError("Encode Source B first")
            session.current = mix_channels(session.original, session.source_b, payload.get("values", []))
            audio = None
            status = "Applied the 256-channel Source A/B latent mix."
        elif action == "crossfade":
            if session.source_b is None:
                raise ValueError("Encode Source B first")
            start, end = seconds_to_frames(payload.get("start", 0), payload.get("end", session.duration), session.original.shape[1], session.duration)
            session.current = time_crossfade(
                session.original, session.source_b, start, end,
                str(payload.get("direction", "A to B")), str(payload.get("curve", "Smoothstep")),
            )
            audio = None
            status = f"Applied {payload.get('direction', 'A to B')} latent crossfade over frames {start}:{end}."
        elif action == "profile":
            source = session.current if bool(payload.get("cumulative", False)) else session.original
            start, end = seconds_to_frames(payload.get("start", 0), payload.get("end", session.duration), source.shape[1], session.duration)
            session.current = apply_profile(source, payload.get("values", []), start, end, str(payload.get("mode", "Offset")))
            audio = None
            status = f"Applied the 256-channel {payload.get('mode', 'Offset').lower()} profile over frames {start}:{end}."
        else:
            raise ValueError("Unknown action; expected reset, intervention, mix, crossfade or profile")
        if audio is None:
            with accelerator_lock:
                audio = engine.same_decode(session.current, session.codec, session.target_samples)
        return _same_commit(session, audio, status)


@app.get("/api/same/info")
def same_info():
    if engine is None:
        raise HTTPException(503, "backend not configured")
    return {
        "contract_version": CONTRACT_VERSION,
        "backend": engine.diagnostics().as_dict(),
        "codecs": list(engine.same_codec_choices),
        "operations": list(OPERATIONS),
        "waveforms": list(WAVEFORMS),
        "profile_channels": PROFILE_CHANNELS,
        "sample_rate": SAME_SAMPLE_RATE,
        "osc": {"enabled": same_osc.enabled, "host": same_osc.host, "port": same_osc.port},
    }


@app.post("/api/same/osc/arm/{session_id}")
def same_osc_arm(session_id: str):
    session = _same_session(session_id)
    if not same_osc.enabled:
        raise HTTPException(503, "OSC is disabled; restart sa3-monitor with --osc")
    return same_osc.arm(session_id, session.duration)


@app.post("/api/same/osc/disarm/{session_id}")
def same_osc_disarm(session_id: str):
    _same_session(session_id)
    return same_osc.disarm(session_id)


@app.get("/api/same/osc/status/{session_id}")
def same_osc_status(session_id: str):
    _same_session(session_id)
    return same_osc.snapshot(session_id)


@app.post("/api/same/encode")
async def same_encode(
    request: Request,
    codec: str = Form("same-l"),
    source: str = Form("a"),
    session_id: str = Form(""),
    audio: UploadFile = File(...),
):
    if engine is None:
        raise HTTPException(503, "backend not configured")
    if codec not in engine.same_codec_choices:
        raise HTTPException(400, f"unknown SAME codec: {codec}")
    source = source.strip().lower()
    if source not in {"a", "b"}:
        raise HTTPException(400, "source must be a or b")
    if source == "a":
        sid = new_session_id()
        directory = RUN_ROOT / "same_lab" / sid
        directory.mkdir(parents=True, exist_ok=False)
        session = None
    else:
        session = _same_session(session_id)
        sid = session.session_id
        directory = session.directory
        if codec != session.codec:
            raise HTTPException(400, "Source B must use the same codec as Source A")
    suffix = Path(audio.filename or "source.wav").suffix.lower()
    if not suffix or len(suffix) > 8:
        suffix = ".wav"
    input_path = directory / f"source_{source}{suffix}"
    with input_path.open("wb") as output:
        shutil.copyfileobj(audio.file, output)
    client_ip = request.client.host if request.client else "unknown"
    print(
        "SAME_LAB " + json.dumps({"event": f"encode_{source}", "ip": client_ip, "session_id": sid, "codec": codec, "file": audio.filename or "audio"}, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )
    try:
        with accelerator_lock:
            encoded = engine.same_encode(str(input_path), codec)
    except Exception as exc:
        raise HTTPException(500, f"SAME encode failed: {type(exc).__name__}: {exc}") from exc
    latent = np.asarray(encoded["latent"], dtype=np.float32)
    reconstructed = np.asarray(encoded["audio"], dtype=np.float32)
    target_samples = int(encoded["target_samples"])
    if latent.ndim != 2 or latent.shape[0] != PROFILE_CHANNELS:
        raise HTTPException(500, f"backend returned invalid SAME latent shape {latent.shape}")
    if source == "a":
        session = LatentSession(
            session_id=sid,
            directory=directory,
            client_ip=client_ip,
            codec=codec,
            original=latent,
            current=latent.copy(),
            baseline_audio=reconstructed,
            target_samples=target_samples,
            source_a_name=audio.filename or "Source A",
        )
        with same_sessions_lock:
            same_sessions[sid] = session
        _write_wav(directory / "baseline.wav", reconstructed)
        status = f"Encoded Source A with {codec.upper()}: 256 x {latent.shape[1]} latent values, {session.duration:.3f} seconds."
        return _same_commit(session, reconstructed, status)
    assert session is not None
    with session.lock:
        if target_samples != session.target_samples or latent.shape != session.original.shape:
            raise HTTPException(
                400,
                f"Source B must exactly match Source A. A={session.target_samples} samples/{session.original.shape}; B={target_samples} samples/{latent.shape}",
            )
        session.source_b = latent
        session.source_b_audio = reconstructed
        session.source_b_name = audio.filename or "Source B"
        _write_wav(directory / "source_b.wav", reconstructed)
        session.updated = time.time()
        return _same_summary(session, f"Encoded Source B with {codec.upper()}; it is ready for channel mixing or time crossfade.")


@app.post("/api/same/edit/{session_id}")
async def same_edit(session_id: str, request: Request):
    if engine is None:
        raise HTTPException(503, "backend not configured")
    session = _same_session(session_id)
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(400, "request body must be JSON") from exc
    action = str(payload.get("action", "")).strip().lower()
    print(
        "SAME_LAB " + json.dumps({"event": action, "ip": request.client.host if request.client else "unknown", "session_id": session_id, "codec": session.codec}, separators=(",", ":")),
        flush=True,
    )
    try:
        return _same_apply_payload(session, payload)
    except HTTPException:
        raise
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"SAME decode/edit failed: {type(exc).__name__}: {exc}") from exc


@app.get("/api/same/output/{session_id}/{name}")
def same_output(session_id: str, name: str):
    session = _same_session(session_id)
    allowed = {"baseline.wav", "current.wav", "difference.wav", "source_b.wav", "current.npy"}
    if name not in allowed:
        raise HTTPException(404, "file not found")
    path = session.directory / name
    if not path.exists():
        raise HTTPException(404, "file not ready")
    if name.endswith(".wav"):
        return FileResponse(path, media_type="audio/wav", filename=f"same_{session_id}_{name}")
    return FileResponse(path, media_type="application/octet-stream", filename=f"same_{session_id}_latent.npy")


@app.post("/api/generate")
async def generate(
    request: Request,
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    mode: str = Form("text"),
    dit: str = Form("sm-music"),
    decoder: str = Form("auto"),
    seconds: float = Form(10.0),
    steps: int = Form(8),
    seed: str = Form(""),
    cfg: float = Form(1.0),
    apg: float = Form(1.0),
    sigma_max: float = Form(1.0),
    inpaint_start: float = Form(0.0),
    inpaint_end: float = Form(0.0),
    audio: Optional[UploadFile] = File(None),
):
    mode = mode.strip().lower()
    if mode not in {"text", "audio", "inpaint"}:
        raise HTTPException(400, "mode must be text, audio or inpaint")
    if engine is None:
        raise HTTPException(503, "backend not configured")
    if dit not in engine.dit_choices:
        raise HTTPException(400, f"unknown DiT: {dit}")
    if decoder != "auto" and decoder not in engine.decoder_choices:
        raise HTTPException(400, f"unknown decoder: {decoder}")
    if seconds <= 0 or seconds > engine.max_seconds:
        raise HTTPException(400, f"seconds must be between 0 and {engine.max_seconds:g}")
    if steps < 1 or steps > 64:
        raise HTTPException(400, "steps must be between 1 and 64")
    if sigma_max < 0.01:
        raise HTTPException(400, "sigma max must be >= 0.01")
    if mode in {"audio", "inpaint"} and audio is None:
        raise HTTPException(400, "this mode needs an input audio file")
    if mode == "inpaint" and not (0 <= inpaint_start < inpaint_end <= seconds):
        raise HTTPException(400, "inpaint start/end must fall inside the requested duration")

    try:
        seed_value = None if seed.strip() == "" else int(seed.strip())
    except ValueError:
        raise HTTPException(400, "seed must be an integer or blank")

    run_id = uuid.uuid4().hex[:12]
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    input_path = None
    if audio is not None:
        suffix = Path(audio.filename or "input.wav").suffix.lower()
        if not suffix or len(suffix) > 8:
            suffix = ".wav"
        input_path = run_dir / f"input{suffix}"
        with input_path.open("wb") as f:
            shutil.copyfileobj(audio.file, f)

    config = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seconds": float(seconds),
        "steps": int(steps),
        "seed": seed_value,
        "cfg": float(cfg),
        "apg": float(apg),
        "sigma_max": float(sigma_max),
        "dit_name": dit,
        "decoder_name": decoder,
        "mode": mode,
        "input_audio": str(input_path) if input_path else None,
        "inpaint_start": float(inpaint_start),
        "inpaint_end": float(inpaint_end),
    }
    state = RunState(
        run_id=run_id,
        run_dir=run_dir,
        config=config,
        client_ip=request.client.host if request.client else "unknown",
    )
    with runs_lock:
        runs[run_id] = state
    position = work_q.qsize() + 1
    state.emit({"type": "status", "status": "queued", "time": time.time(), "queue_depth": position})
    work_q.put(run_id)
    _terminal_log("queued", state, queue_position=position, waiting=work_q.qsize())
    return {"run_id": run_id, "events_url": f"/api/events/{run_id}"}


@app.get("/api/events/{run_id}")
def events(run_id: str):
    with runs_lock:
        state = runs.get(run_id)
    if state is None:
        raise HTTPException(404, "run not found")

    def stream():
        cursor = 0
        last_keepalive = time.time()
        while True:
            batch = []
            with state.condition:
                if cursor >= len(state.events) and state.status not in {"complete", "error"}:
                    state.condition.wait(timeout=1.0)
                if cursor < len(state.events):
                    batch = state.events[cursor:]
                    cursor = len(state.events)
            for ev in batch:
                yield "data: " + json.dumps(ev, separators=(",", ":")) + "\n\n"
            now = time.time()
            if now - last_keepalive > 10:
                yield ": keepalive\n\n"
                last_keepalive = now
            if state.status in {"complete", "error"} and cursor >= len(state.events):
                break

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/run/{run_id}")
def run_status(run_id: str):
    with runs_lock:
        state = runs.get(run_id)
    if state is None:
        raise HTTPException(404, "run not found")
    return {
        "run_id": state.run_id,
        "status": state.status,
        "created": state.created,
        "finished": state.finished,
        "output_name": state.output_name,
        "error": state.error,
        "traceback": state.error_traceback,
        "events": len(state.events),
    }


@app.get("/api/debug/{run_id}")
def debug_run(run_id: str):
    """Return diagnostics for an in-memory run or a persisted failed run."""
    with runs_lock:
        state = runs.get(run_id)
    if state is not None:
        return {
            "run_id": state.run_id,
            "status": state.status,
            "error": state.error,
            "traceback": state.error_traceback,
            "run_dir": str(state.run_dir),
            "client_ip": state.client_ip,
            "events": state.events,
        }

    run_dir = RUN_ROOT / run_id
    error_path = run_dir / "error.txt"
    if error_path.exists():
        return {
            "run_id": run_id,
            "status": "error",
            "error": "persisted failure — see traceback",
            "traceback": error_path.read_text(encoding="utf-8"),
            "run_dir": str(run_dir),
            "events": [],
        }
    raise HTTPException(404, "run not found")


def _run_dir_for(run_id: str) -> Path:
    with runs_lock:
        state = runs.get(run_id)
    return state.run_dir if state is not None else RUN_ROOT / run_id


def _safe_num(value) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _download_name(run_id: str, index: int | None = None) -> str:
    with runs_lock:
        state = runs.get(run_id)
    prefix = f"{index:03d}_" if index is not None else ""
    if state is None:
        return f"{prefix}sa3_{run_id}.wav"
    c = state.config
    return (
        f"{prefix}sa3_sigma-{_safe_num(c.get('sigma_max', 1))}"
        f"_cfg-{_safe_num(c.get('cfg', 1))}"
        f"_apg-{_safe_num(c.get('apg', 1))}"
        f"_seed-{c.get('seed') if c.get('seed') is not None else 'random'}_{run_id}.wav"
    )


@app.get("/api/output/{run_id}/{name}")
def output(run_id: str, name: str):
    if name != "generated.wav":
        raise HTTPException(404, "file not found")
    path = _run_dir_for(run_id) / name
    if not path.exists():
        raise HTTPException(404, "file not ready")
    return FileResponse(path, media_type="audio/wav", filename=_download_name(run_id))


@app.get("/api/archive")
def archive(run_ids: str):
    ids = [x.strip() for x in run_ids.split(",") if x.strip()]
    if not ids:
        raise HTTPException(400, "run_ids is required")
    if len(ids) > 256:
        raise HTTPException(400, "too many runs")
    for rid in ids:
        if not re.fullmatch(r"[0-9a-f]{12}", rid):
            raise HTTPException(400, f"invalid run id: {rid}")

    export_dir = RUN_ROOT / "batch_exports"
    export_dir.mkdir(exist_ok=True)
    zip_path = export_dir / f"sa3_batch_{uuid.uuid4().hex[:10]}.zip"
    included = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, rid in enumerate(ids, 1):
            wav = _run_dir_for(rid) / "generated.wav"
            if wav.exists():
                zf.write(wav, arcname=_download_name(rid, i))
                included += 1
    if not included:
        zip_path.unlink(missing_ok=True)
        raise HTTPException(404, "none of the requested outputs are ready")
    return FileResponse(zip_path, media_type="application/zip", filename="sa3_batch_results.zip")


def main():
    ap = argparse.ArgumentParser(description="SA3 graphical browser monitor")
    ap.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto", help="Inference backend (default: conservative auto-detection)")
    ap.add_argument("--host", default=None, help="Bind address. Defaults to 0.0.0.0 for CUDA and 127.0.0.1 for MLX")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--sa3-root", default=os.environ.get("SA3_ROOT"), help="stable-audio-3 checkout (required for MLX unless auto-located)")
    ap.add_argument("--output-dir", default=None, help="Host directory for generated runs (default: project/output/monitor)")
    ap.add_argument("--osc", action="store_true", help="Enable session-routed SAME Lab OSC control")
    ap.add_argument("--osc-host", default="127.0.0.1", help="OSC UDP bind address (use 0.0.0.0 only on a trusted LAN)")
    ap.add_argument("--osc-port", type=int, default=9000, help="SAME Lab OSC UDP port (default: 9000)")
    ap.add_argument("--diagnose", action="store_true", help="Print backend dependency probes and exit")
    args = ap.parse_args()
    if args.diagnose:
        print(json.dumps({"platform": os.sys.platform, "python": os.sys.version, "backends": diagnostics_json(args.sa3_root)}, indent=2))
        return
    try:
        selected = create_backend(args.backend, args.sa3_root)
        configure_backend(selected, args.output_dir)
    except BackendUnavailable as exc:
        ap.error(str(exc))
    host = args.host or ("0.0.0.0" if selected.name == "cuda" else "127.0.0.1")
    if args.osc:
        try:
            same_osc.start(args.osc_host, args.osc_port, _same_session, _same_apply_payload)
        except Exception as exc:
            ap.error(str(exc))
    info = selected.diagnostics()
    print("\nSA3 Graphical Pipeline Monitor")
    print(f"  backend : {info.backend}")
    print(f"  device  : {info.device_name} ({info.device})")
    print(f"  dtype   : {info.dtype}")
    if info.vram_total_bytes:
        print(f"  VRAM    : {info.vram_total_bytes / 2**30:.2f} GiB total")
    print(f"  outputs : {RUN_ROOT}")
    print(f"  URL     : http://{'<this-linux-host-ip>' if host == '0.0.0.0' else host}:{args.port}")
    print(f"  OSC     : {'udp://' + args.osc_host + ':' + str(args.osc_port) if args.osc else 'disabled (enable with --osc)'}")
    if host == "0.0.0.0":
        print("  SECURITY: listening on all interfaces; allow port 7861 only from your trusted LAN.")
        print("            Do not port-forward this unauthenticated service to the internet.\n")
    if args.osc and args.osc_host == "0.0.0.0":
        print("  OSC SECURITY: accepting unauthenticated UDP on all interfaces.")
        print(f"                Allow UDP {args.osc_port} only from your trusted LAN; never expose it publicly.\n")
    uvicorn.run(app, host=host, port=args.port, log_level="info", proxy_headers=False)


if __name__ == "__main__":
    main()
