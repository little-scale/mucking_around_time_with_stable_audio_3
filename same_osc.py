"""Optional, explicitly armed OSC control for the integrated SAME Latent Lab."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from same_lab import LATENT_FRAMES_PER_SECOND, OPERATIONS, PROFILE_CHANNELS, WAVEFORMS


CONTROL_DEFAULTS: dict[str, Any] = {
    "channels": "0",
    "start": 0.0,
    "end": 1.0,
    "operation": OPERATIONS[0],
    "amount": 0.0,
    "frame_displacement": 1,
    "seed": 0,
    "cumulative": False,
    "bpm": 120.0,
    "cycle_beats": 1.0,
    "period_seconds": 0.5,
    "lfo_period": round(0.5 * LATENT_FRAMES_PER_SECOND, 6),
    "lfo_depth": 1.0,
    "lfo_waveform": "Sine",
    "lfo_phase": 0.0,
    "lfo_invert": False,
    "crossfade_direction": "A to B",
    "crossfade_curve": "Smoothstep",
    "mix_random_seed": 0,
    "profile_mode": "Offset",
    "profile_set_value": 0.0,
    "profile_scale_factor": 1.0,
    "profile_random_depth": 0.25,
    "profile_random_seed": 0,
}

ALIASES = {
    "channel_spec": "channels", "model": "codec", "displacement": "frame_displacement",
    "waveform": "lfo_waveform", "phase": "lfo_phase", "invert_lfo": "lfo_invert",
    "direction": "crossfade_direction", "curve": "crossfade_curve",
    "latent_mix_random_seed": "mix_random_seed",
}
OPERATION_ALIASES = {
    "zero": OPERATIONS[0], "scale": OPERATIONS[1], "offset": OPERATIONS[2],
    "offset_sigma": OPERATIONS[2], "noise": OPERATIONS[3], "invert": OPERATIONS[4],
    "mean": OPERATIONS[5], "freeze": OPERATIONS[6], "shuffle": OPERATIONS[7],
    "keep": OPERATIONS[8], "displace": OPERATIONS[9], "lfo_add": OPERATIONS[10],
    "lfo_replace": OPERATIONS[11], "lfo_multiply": OPERATIONS[12],
}
FLOAT_FIELDS = {
    "start", "end", "amount", "bpm", "cycle_beats", "period_seconds", "lfo_period",
    "lfo_depth", "lfo_phase", "profile_set_value", "profile_scale_factor", "profile_random_depth",
}
INT_FIELDS = {"frame_displacement", "seed", "mix_random_seed", "profile_random_seed"}
BOOL_FIELDS = {"cumulative", "lfo_invert"}


@dataclass
class OSCSessionState:
    controls: dict[str, Any] = field(default_factory=lambda: dict(CONTROL_DEFAULTS))
    mix: np.ndarray = field(default_factory=lambda: np.zeros(PROFILE_CHANNELS, dtype=np.float32))
    profile: np.ndarray = field(default_factory=lambda: np.zeros(PROFILE_CHANNELS, dtype=np.float32))
    revision: int = 0
    busy: bool = False
    error: str = ""
    last_result: dict | None = None
    last_message: str = ""


class SameOSCController:
    def __init__(self):
        self.enabled = False
        self.host = "127.0.0.1"
        self.port = 9000
        self.active_session_id: str | None = None
        self._states: dict[str, OSCSessionState] = {}
        self._lock = threading.RLock()
        self._actions: queue.Queue[tuple[str, str]] = queue.Queue()
        self._server = None
        self._get_session: Callable[[str], Any] | None = None
        self._apply: Callable[[Any, dict], dict] | None = None

    def start(self, host: str, port: int, get_session: Callable[[str], Any], apply_payload: Callable[[Any, dict], dict]):
        try:
            from pythonosc import dispatcher, osc_server
        except ImportError as exc:
            raise RuntimeError(
                "OSC requested but python-osc is not installed. Install requirements-monitor.txt or run without --osc."
            ) from exc
        self.host, self.port = host, int(port)
        self._get_session, self._apply = get_session, apply_payload
        route = dispatcher.Dispatcher()
        route.set_default_handler(self._handle)
        self._server = osc_server.ThreadingOSCUDPServer((host, int(port)), route)
        self.enabled = True
        threading.Thread(target=self._server.serve_forever, name="same-osc-server", daemon=True).start()
        threading.Thread(target=self._worker, name="same-osc-actions", daemon=True).start()

    def arm(self, session_id: str, duration: float) -> dict:
        with self._lock:
            state = self._states.setdefault(session_id, OSCSessionState())
            state.controls["end"] = float(duration)
            state.revision += 1
            state.last_message = "Session armed for OSC"
            self.active_session_id = session_id
        return self.snapshot(session_id)

    def disarm(self, session_id: str) -> dict:
        with self._lock:
            if self.active_session_id == session_id:
                self.active_session_id = None
            state = self._states.setdefault(session_id, OSCSessionState())
            state.revision += 1
            state.last_message = "Session disarmed"
        return self.snapshot(session_id)

    def snapshot(self, session_id: str) -> dict:
        with self._lock:
            state = self._states.setdefault(session_id, OSCSessionState())
            return {
                "enabled": self.enabled,
                "host": self.host,
                "port": self.port,
                "armed": self.active_session_id == session_id,
                "active_session_id": self.active_session_id,
                "revision": state.revision,
                "busy": state.busy,
                "error": state.error,
                "message": state.last_message,
                "controls": dict(state.controls),
                "mix": state.mix.tolist(),
                "profile": state.profile.tolist(),
                "result": state.last_result,
            }

    def _bump(self, state: OSCSessionState, message: str = ""):
        state.revision += 1
        if message:
            state.last_message = message

    @staticmethod
    def _bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "off", "no"}
        return bool(value)

    def _coerce(self, name: str, value: Any) -> tuple[str, Any]:
        name = ALIASES.get(name, name)
        if name not in CONTROL_DEFAULTS:
            raise ValueError(f"Unknown OSC control: {name}")
        if name in FLOAT_FIELDS:
            value = float(value)
        elif name in INT_FIELDS:
            value = int(value)
        elif name in BOOL_FIELDS:
            value = self._bool(value)
        elif name == "operation":
            if isinstance(value, (int, float)):
                value = OPERATIONS[int(value)]
            else:
                value = OPERATION_ALIASES.get(str(value).strip().lower(), str(value))
            if value not in OPERATIONS:
                raise ValueError(f"Unknown operation: {value}")
        elif name == "lfo_waveform":
            value = WAVEFORMS[int(value)] if isinstance(value, (int, float)) else str(value).title()
            if value not in WAVEFORMS:
                raise ValueError(f"Unknown waveform: {value}")
        elif name == "crossfade_direction":
            value = ("A to B", "B to A")[int(value)] if isinstance(value, (int, float)) else str(value).replace("→", "to")
        elif name == "crossfade_curve":
            value = ("Smoothstep", "Linear")[int(value)] if isinstance(value, (int, float)) else str(value).title()
        elif name == "profile_mode":
            value = ("Offset", "Multiply")[int(value)] if isinstance(value, (int, float)) else str(value).title()
        else:
            value = str(value)
        return name, value

    def _active(self) -> tuple[str, OSCSessionState]:
        if not self.active_session_id:
            raise ValueError("No SAME browser session is armed for OSC")
        return self.active_session_id, self._states.setdefault(self.active_session_id, OSCSessionState())

    def _handle(self, address: str, *args):
        try:
            with self._lock:
                session_id, state = self._active()
                state.error = ""
                if address.startswith("/same/set/"):
                    if not args:
                        raise ValueError("OSC control message requires a value")
                    name, value = self._coerce(address.removeprefix("/same/set/"), args[0])
                    state.controls[name] = value
                    self._bump(state, f"{name} = {value}")
                    return
                if address.startswith("/same/mix/"):
                    self._handle_bank(session_id, state, "mix", address.removeprefix("/same/mix/"), args)
                    return
                if address == "/same/mix" and len(args) >= 2:
                    self._handle_bank(session_id, state, "mix", str(int(args[0])), (args[1],))
                    return
                if address.startswith("/same/profile/"):
                    self._handle_bank(session_id, state, "profile", address.removeprefix("/same/profile/"), args)
                    return
                if address == "/same/profile" and len(args) >= 2:
                    self._handle_bank(session_id, state, "profile", str(int(args[0])), (args[1],))
                    return
                if address == "/same/action/period_from_bpm":
                    seconds = 60.0 * float(state.controls["cycle_beats"]) / float(state.controls["bpm"])
                    state.controls["period_seconds"] = seconds
                    state.controls["lfo_period"] = seconds * LATENT_FRAMES_PER_SECOND
                    self._bump(state, "Updated LFO period from BPM")
                    return
                if address == "/same/action/period_from_seconds":
                    seconds = float(state.controls["period_seconds"])
                    state.controls["cycle_beats"] = seconds * float(state.controls["bpm"]) / 60.0
                    state.controls["lfo_period"] = seconds * LATENT_FRAMES_PER_SECOND
                    self._bump(state, "Updated LFO period from seconds")
                    return
                if address == "/same/action/period_from_frames":
                    seconds = float(state.controls["lfo_period"]) / LATENT_FRAMES_PER_SECOND
                    state.controls["period_seconds"] = seconds
                    state.controls["cycle_beats"] = seconds * float(state.controls["bpm"]) / 60.0
                    self._bump(state, "Updated LFO period from frames")
                    return
                actions = {
                    "/same/apply": "mix", "/same/reset": "reset",
                    "/same/action/apply_intervention": "intervention",
                    "/same/action/apply_profile": "profile",
                    "/same/action/apply_time_crossfade": "crossfade",
                }
                if address in actions:
                    self._queue_action(session_id, state, actions[address])
                    return
                raise ValueError(f"Unknown SAME OSC address: {address}")
        except Exception as exc:
            with self._lock:
                if self.active_session_id:
                    state = self._states.setdefault(self.active_session_id, OSCSessionState())
                    state.error = str(exc)
                    self._bump(state, "OSC message rejected")

    def _handle_bank(self, session_id: str, state: OSCSessionState, bank_name: str, command: str, args: tuple):
        bank = state.mix if bank_name == "mix" else state.profile
        lo, hi = ((0.0, 1.0) if bank_name == "mix" else (-3.0, 3.0))
        if command.isdigit():
            if not args:
                raise ValueError("Bank channel message requires a value")
            index = int(command)
            if not 0 <= index < PROFILE_CHANNELS:
                raise ValueError("Latent channel must be 0-255")
            bank[index] = np.clip(float(args[0]), lo, hi)
        elif command == "all":
            values = np.asarray(args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args, dtype=np.float32).reshape(-1)
            if values.size == 1:
                bank.fill(np.clip(values[0], lo, hi))
            elif values.size == PROFILE_CHANNELS:
                bank[:] = np.clip(values, lo, hi)
            else:
                raise ValueError("Bank /all needs one value or exactly 256 values")
        elif command in {"a", "reset"}:
            bank.fill(0.0)
        elif bank_name == "mix" and command == "half":
            bank.fill(0.5)
        elif bank_name == "mix" and command == "b":
            bank.fill(1.0)
        elif command in {"flip", "invert"}:
            bank[:] = 1.0 - bank if bank_name == "mix" else -bank
        elif command == "set" and bank_name == "profile":
            value = args[0] if args else state.controls["profile_set_value"]
            bank.fill(np.clip(float(value), lo, hi))
        elif command == "scale" and bank_name == "profile":
            factor = args[0] if args else state.controls["profile_scale_factor"]
            bank[:] = np.clip(bank * float(factor), lo, hi)
        elif command == "random":
            if bank_name == "mix":
                seed = int(args[0]) if args else int(state.controls["mix_random_seed"])
                bank[:] = np.random.default_rng(seed).uniform(0, 1, PROFILE_CHANNELS)
            else:
                depth = float(args[0]) if args else float(state.controls["profile_random_depth"])
                seed = int(args[1]) if len(args) > 1 else int(state.controls["profile_random_seed"])
                bank[:] = np.clip(np.random.default_rng(seed).uniform(-depth, depth, PROFILE_CHANNELS), lo, hi)
        elif command == "apply":
            self._queue_action(session_id, state, bank_name)
            return
        else:
            raise ValueError(f"Unknown /same/{bank_name}/{command} command")
        self._bump(state, f"Updated {bank_name} bank")

    def _queue_action(self, session_id: str, state: OSCSessionState, action: str):
        self._actions.put((session_id, action))
        state.busy = True
        self._bump(state, f"Queued {action}")

    def _payload(self, state: OSCSessionState, action: str) -> dict:
        c = state.controls
        if action == "intervention":
            return {
                "action": action, "channels": c["channels"], "start": c["start"], "end": c["end"],
                "operation": c["operation"], "amount": c["amount"], "seed": c["seed"],
                "cumulative": c["cumulative"], "frame_displacement": c["frame_displacement"],
                "lfo_period": c["lfo_period"], "lfo_depth": c["lfo_depth"],
                "lfo_waveform": c["lfo_waveform"], "lfo_phase": c["lfo_phase"], "lfo_invert": c["lfo_invert"],
            }
        if action == "mix":
            return {"action": action, "values": state.mix.tolist()}
        if action == "profile":
            return {"action": action, "values": state.profile.tolist(), "mode": c["profile_mode"], "start": c["start"], "end": c["end"], "cumulative": c["cumulative"]}
        if action == "crossfade":
            return {"action": action, "start": c["start"], "end": c["end"], "direction": c["crossfade_direction"], "curve": c["crossfade_curve"]}
        return {"action": "reset"}

    def _worker(self):
        while True:
            session_id, action = self._actions.get()
            try:
                with self._lock:
                    state = self._states.setdefault(session_id, OSCSessionState())
                    payload = self._payload(state, action)
                if self._get_session is None or self._apply is None:
                    raise RuntimeError("OSC controller is not configured")
                session = self._get_session(session_id)
                result = self._apply(session, payload)
                with self._lock:
                    state.last_result = result
                    state.error = ""
                    state.busy = False
                    self._bump(state, result.get("status", f"Completed {action}"))
            except Exception as exc:
                with self._lock:
                    state = self._states.setdefault(session_id, OSCSessionState())
                    state.error = f"{type(exc).__name__}: {exc}"
                    state.busy = False
                    self._bump(state, f"{action} failed")
            finally:
                self._actions.task_done()


same_osc = SameOSCController()
