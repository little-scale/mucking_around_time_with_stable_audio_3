import asyncio
import json
import shutil
import struct
import threading
import wave
from dataclasses import asdict, replace
from pathlib import Path

from fastapi.testclient import TestClient

from drift_api import create_app
from drift_engine import MultitrackEngine, TrackConfig
import server


class DriftStubBackend:
    name = "cuda"
    sample_rate = 44_100

    def generate(self, *, output_path, emit, **config):
        emit({"type": "stage", "stage": "dit", "state": "active"})
        with wave.open(output_path, "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(self.sample_rate)
            output.writeframes(struct.pack("<" + "h" * 882, *([0] * 882)))
        emit({"type": "audio_ready", "output_path": output_path})


class IntroductionCaptureBackend(DriftStubBackend):
    def __init__(self):
        self.inputs = []

    def generate(self, *, input_audio, output_path, emit, **config):
        assert input_audio is not None
        with wave.open(input_audio, "rb") as audio:
            self.inputs.append(audio.readframes(audio.getnframes()))
        shutil.copyfile(input_audio, output_path)
        emit({"type": "audio_ready", "output_path": output_path})


def write_test_wav(path, value, frames=441):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(44_100)
        output.writeframes(struct.pack("<" + "h" * frames, *([value] * frames)))


def make_loop(tmp_path, logs=None):
    return MultitrackEngine(
        project_root=tmp_path,
        output_dir=tmp_path / "drift",
        backend=DriftStubBackend(),
        accelerator_lock=threading.RLock(),
        request_logger=(logs.append if logs is not None else None),
    )


def test_drift_uses_selected_backend_and_preserves_shared_conditions(tmp_path):
    logs = []
    loop = make_loop(tmp_path, logs)
    track = loop.tracks["melodic"]
    record = asyncio.run(loop._generate_once(track))
    assert record is not None
    assert track.current_path.exists()
    assert logs[0]["track"] == "melodic"
    assert logs[0]["model"] == "sm-music"
    assert logs[0]["mode"] == "text"
    assert logs[0]["sigma_max"] == 1.0
    assert logs[0]["prompt"] == "warm evolving harmony, 120 BPM, seamless pitched chordal melodic instrumental loop"


def test_clear_one_output_removes_audio_but_preserves_settings(tmp_path):
    loop = make_loop(tmp_path)
    track = loop.tracks["texture"]
    original = track.config
    asyncio.run(loop._generate_once(track))
    assert track.current_path.exists() and track.versions

    result = asyncio.run(loop.clear_output("texture"))
    assert result["current_removed"] is True
    assert result["settings_preserved"] is True
    assert not track.current_path.exists()
    assert track.versions == []
    assert track.config.prompt_tags == original.prompt_tags
    assert track.config.model == original.model
    assert track.config.revision == original.revision + 1


def test_drift_api_and_frontend_expose_per_track_and_clear_all_controls(tmp_path):
    loop = make_loop(tmp_path)
    client = TestClient(create_app(loop))
    assert client.get("/status").status_code == 200
    assert client.delete("/tracks/melodic/output").status_code == 200
    cleared = client.delete("/outputs")
    assert cleared.status_code == 200
    assert len(cleared.json()["tracks"]) == 3

    source = (Path(__file__).parents[1] / "static" / "drift" / "index.html").read_text()
    script = (Path(__file__).parents[1] / "static" / "drift" / "app.js").read_text()
    assert "CLEAR MELODIC OUTPUT" in source
    assert "CLEAR ALL OUTPUTS" in source
    assert 'const API_BASE = "/api/drift"' in script
    assert '/tracks/${track.id}/output' in script


def test_mutation_interval_is_loop_based_and_uses_start_to_start_deadlines(tmp_path):
    loop = make_loop(tmp_path)
    config = replace(TrackConfig.defaults("percussion"), beats=4, mutation_interval_loops=2)
    assert loop._mutation_period_seconds(config, 120) == 4.0
    assert loop._advance_mutation_deadline(10.0, 4.0, 11.0) == 14.0
    # A late render is made due immediately instead of accumulating old jobs.
    assert loop._advance_mutation_deadline(10.0, 4.0, 15.5) == 15.5


def test_latched_track_settings_preserve_active_render_and_existing_deadline(tmp_path):
    loop = MultitrackEngine(
        project_root=tmp_path,
        output_dir=tmp_path / "loop-mutator",
        backend=DriftStubBackend(),
        accelerator_lock=threading.RLock(),
        track_ids=("melodic",),
    )
    track = loop.tracks["melodic"]
    loop.running = True
    track.phase = "waiting"
    track.next_generation_timestamp = 1234.5

    waiting_config = asyncio.run(loop.update_track(
        "melodic", latch_for_next=True, mutation=0.4, cfg=3.0,
    ))
    assert waiting_config.mutation == 0.4
    assert track.config == waiting_config
    assert track.pending_config is None
    assert track.next_generation_timestamp == 1234.5
    assert track.dirty is False

    track.phase = "generating"
    pending_config = asyncio.run(loop.update_track(
        "melodic", latch_for_next=True, mutation=0.7, steps=12,
    ))
    assert track.config == waiting_config
    assert track.pending_config == pending_config
    status = asyncio.run(loop.status())["tracks"]["melodic"]
    assert status["config"]["mutation"] == 0.4
    assert status["pending_config"]["mutation"] == 0.7

    record = asyncio.run(loop._generate_once(track))
    assert record is not None
    assert record["mutation"] == 0.4
    assert record["steps"] == waiting_config.steps
    assert track.current_path.exists()
    assert track.config == pending_config
    assert track.pending_config is None


def test_latched_bpm_does_not_queue_an_extra_generation(tmp_path):
    loop = make_loop(tmp_path)
    loop.running = True
    deadlines = {}
    for index, track in enumerate(loop.tracks.values(), start=1):
        track.phase = "waiting"
        track.dirty = False
        track.next_generation_timestamp = 1000.0 + index
        deadlines[track.id] = track.next_generation_timestamp

    session = asyncio.run(loop.update_session(latch_for_next=True, bpm=90))
    assert session.bpm == 90
    for track in loop.tracks.values():
        assert track.dirty is False
        assert track.next_generation_timestamp == deadlines[track.id]


def test_blank_optional_prompt_clauses_are_omitted(tmp_path):
    loop = make_loop(tmp_path)
    client = TestClient(create_app(loop))
    updated = client.put(
        "/tracks/melodic",
        json={"role_prompt": "   ", "negative_prompt": "  "},
    )
    assert updated.status_code == 200
    config = updated.json()["config"]
    assert config["role_prompt"] is None
    assert config["negative_prompt"] is None
    assert config["constructed_prompt"] == "warm evolving harmony, 120 BPM"


def test_track_config_overrides_clear_hidden_prompts_and_migrate_saved_config(tmp_path):
    output_dir = tmp_path / "loop-mutator"
    config_path = output_dir / "tracks" / "melodic" / "config.json"
    config_path.parent.mkdir(parents=True)
    original = TrackConfig.defaults("melodic")
    config_path.write_text(json.dumps(asdict(original)))

    logs = []
    loop = MultitrackEngine(
        project_root=tmp_path,
        output_dir=output_dir,
        backend=DriftStubBackend(),
        accelerator_lock=threading.RLock(),
        request_logger=logs.append,
        track_ids=("melodic",),
        track_config_overrides={
            "melodic": {"role_prompt": None, "negative_prompt": None},
        },
    )
    config = loop.tracks["melodic"].config
    saved = json.loads(config_path.read_text())
    assert config.role_prompt is None
    assert config.negative_prompt is None
    assert config.prompt("melodic", 120) == "warm evolving harmony, 120 BPM"
    assert config.revision == original.revision + 1
    assert saved["role_prompt"] is None
    assert saved["negative_prompt"] is None
    status = asyncio.run(loop.status())["tracks"]["melodic"]
    assert status["config"]["constructed_prompt"] == "warm evolving harmony, 120 BPM"
    record = asyncio.run(loop._generate_once(loop.tracks["melodic"]))
    assert record["prompt"] == "warm evolving harmony, 120 BPM"
    assert logs[0]["prompt"] == "warm evolving harmony, 120 BPM"
    assert logs[0]["negative_prompt"] is None


def test_existing_loop_upload_is_introduced_without_raw_replacement(tmp_path):
    backend = IntroductionCaptureBackend()
    loop = MultitrackEngine(
        project_root=tmp_path,
        output_dir=tmp_path / "loop-mutator",
        backend=backend,
        accelerator_lock=threading.RLock(),
        track_ids=("melodic",),
    )
    loop.tracks["melodic"].config = replace(loop.tracks["melodic"].config, beats=1)
    original = tmp_path / "original.wav"
    introduced = tmp_path / "introduced.wav"
    write_test_wav(original, 5000)
    write_test_wav(introduced, -5000)

    seed = asyncio.run(loop.import_reference("melodic", original, "original.wav", mode="auto"))
    track = loop.tracks["melodic"]
    generation_before = track.generation
    current_before = track.current_path.read_bytes()
    queued = asyncio.run(loop.import_reference(
        "melodic", introduced, "introduced.wav", mode="auto", blend=0.25, generations=4,
    ))

    assert seed["mode"] == "uploaded-reference"
    assert queued["queued"] is True
    assert queued["generations_remaining"] == 4
    assert track.generation == generation_before
    assert track.current_path.read_bytes() == current_before
    assert track.last_completed == seed

    blends = []
    for remaining in (3, 2, 1, 0):
        record = asyncio.run(loop._generate_once(track))
        assert record["mode"] == "audio-introduction"
        blends.append(record["introduction"]["blend"])
        if remaining:
            assert track.introduction["generations_remaining"] == remaining
        else:
            assert track.introduction is None
            assert not track.introduction_state_path.exists()
    assert blends == [0.0625, 0.125, 0.1875, 0.25]
    assert len(backend.inputs) == 4


def test_reference_api_auto_introduces_but_replace_remains_explicit(tmp_path):
    loop = MultitrackEngine(
        project_root=tmp_path,
        output_dir=tmp_path / "loop-mutator",
        backend=DriftStubBackend(),
        accelerator_lock=threading.RLock(),
        track_ids=("melodic",),
    )
    loop.tracks["melodic"].config = replace(loop.tracks["melodic"].config, beats=1)
    client = TestClient(create_app(loop))
    source = tmp_path / "upload.wav"
    write_test_wav(source, 4000)
    payload = source.read_bytes()

    seed = client.put("/tracks/melodic/reference?mode=auto", content=payload)
    assert seed.status_code == 201 and seed.json()["imported"] is True
    generation = loop.tracks["melodic"].generation
    introduced = client.put(
        "/tracks/melodic/reference?mode=auto&blend=0.4&generations=2", content=payload,
    )
    assert introduced.status_code == 201 and introduced.json()["introduced"] is True
    assert loop.tracks["melodic"].generation == generation
    assert client.get("/status").json()["tracks"]["melodic"]["introduction"]["blend"] == 0.4

    replaced = client.put("/tracks/melodic/reference?mode=replace", content=payload)
    assert replaced.status_code == 201 and replaced.json()["imported"] is True
    assert loop.tracks["melodic"].generation == generation + 1
    assert loop.tracks["melodic"].introduction is None


def test_legacy_seconds_cadence_migrates_to_every_loop(tmp_path):
    path = tmp_path / "config.json"
    payload = asdict(TrackConfig.defaults("percussion"))
    payload.pop("mutation_interval_loops")
    payload["cadence_seconds"] = 8.0
    path.write_text(json.dumps(payload))
    config = TrackConfig.from_file(path, "percussion")
    saved = json.loads(path.read_text())
    assert config.mutation_interval_loops == 1
    assert saved["mutation_interval_loops"] == 1
    assert "cadence_seconds" not in saved


def test_drift_frontend_exposes_mutation_interval_and_beat_grid_swaps(tmp_path):
    loop = make_loop(tmp_path)
    client = TestClient(create_app(loop))
    updated = client.put("/tracks/percussion", json={"beats": 4, "mutation_interval_loops": 2})
    assert updated.status_code == 200
    assert updated.json()["config"]["mutation_interval_loops"] == 2
    assert updated.json()["config"]["mutation_interval_seconds"] == 4.0

    source = (Path(__file__).parents[1] / "static" / "drift" / "index.html").read_text()
    script = (Path(__file__).parents[1] / "static" / "drift" / "app.js").read_text()
    assert "MUTATION INTERVAL" in source
    assert "Every loop" in source
    assert "CADENCE" not in source
    assert "mutation_interval_loops" in script
    assert "nextBoundaryBeat" in script


def test_drift_is_mounted_in_shared_monitor_server(tmp_path):
    server.configure_backend(DriftStubBackend(), tmp_path)
    client = TestClient(server.app)
    page = client.get("/drift")
    assert page.status_code == 200
    assert "DRIFT LOOPER" in page.text
    status = client.get("/api/drift/status")
    assert status.status_code == 200
    assert set(status.json()["tracks"]) == {"melodic", "percussion", "texture"}


def test_loop_mutator_is_a_separate_single_track_engine_and_page(tmp_path):
    server.configure_backend(DriftStubBackend(), tmp_path)
    client = TestClient(server.app)
    page = client.get("/loop-mutator")
    assert page.status_code == 200
    assert "Loop Mutator" in page.text
    assert 'const API="/api/loop-mutator"' in page.text
    assert 'id="wave"' in page.text
    assert 'id="spectrum"' in page.text
    assert 'class="controls"' in page.text
    assert 'class="visual-grid"' in page.text
    assert "grid-template-columns:minmax(660px,1.05fr) minmax(620px,.95fr)" in page.text
    assert ".analysis{grid-column:1;grid-row:1}" in page.text
    assert ".controls{grid-column:2;grid-row:1}" in page.text
    assert 'id="promptCloud"' in page.text
    assert '<select id="beats"><option>1</option><option>2</option><option>4</option>' in page.text
    assert "function changeTagWeight(" in page.text
    assert "sourceStartedAt" in page.text
    assert "active.buffer.duration" in page.text
    assert 'id="role"' not in page.text
    assert 'id="negative"' not in page.text
    assert "role_prompt:ui.role" not in page.text
    assert "negative_prompt:ui.negative" not in page.text
    assert 'id="seedMode"' not in page.text
    assert 'id="seed"' not in page.text
    assert "ui.seedMode" not in page.text
    assert "ui.seed" not in page.text
    assert "Apply settings" not in page.text
    assert 'id="save"' not in page.text
    assert "scheduleSettingsSave" in page.text
    assert '`/tracks/${TRACK}?latch=true`' in page.text
    assert '"/session?latch=true"' in page.text
    assert '<span class="label">Next</span>' in page.text
    assert '<span class="label">Render</span>' in page.text
    assert 'id="renderTime"' in page.text
    assert "function renderTiming(" in page.text
    assert '× realtime' in page.text
    assert "function updateCountdown()" in page.text
    assert 'setInterval(updateCountdown,250)' in page.text
    assert 'ui.nextRender.textContent="Rendering…"' in page.text
    assert 'ui.nextRender.textContent="Queued…"' in page.text
    assert '<details class="panel history-panel">' in page.text
    assert 'id="historyCount"' in page.text
    assert 'id="importMode"' in page.text
    assert 'id="introBlend"' in page.text
    assert 'id="introGenerations"' in page.text
    assert 'id="introductionState"' in page.text
    assert "new URLSearchParams({mode,blend:ui.introBlend.value,generations:ui.introGenerations.value})" in page.text
    assert "Use +/− to change influence" not in page.text
    assert "Live FFT magnitude" not in page.text
    assert "completed audio takes over" not in page.text
    assert "Audio and model execution remain" not in page.text
    assert "MUTATION INTERVAL" in page.text.upper()
    assert "bpmDirty" in page.text
    assert "document.activeElement!==ui.bpm" in page.text
    assert "function nextPollDelay()" in page.text
    assert "setInterval(refresh,400)" not in page.text

    status = client.get("/api/loop-mutator/status")
    assert status.status_code == 200
    assert set(status.json()["tracks"]) == {"melodic"}
    assert status.json()["tracks"]["melodic"]["label"] == "Loop"
    assert status.json()["tracks"]["melodic"]["config"]["role_prompt"] is None
    assert status.json()["tracks"]["melodic"]["config"]["negative_prompt"] is None
    assert "BPM" in status.json()["tracks"]["melodic"]["config"]["constructed_prompt"]
    server.loop_mutator.tracks["melodic"].phase = "generating"
    latched = client.put(
        "/api/loop-mutator/tracks/melodic?latch=true",
        json={"mutation": 0.61},
    )
    assert latched.status_code == 200
    assert latched.json()["latched"] is True
    pending = client.get("/api/loop-mutator/status").json()["tracks"]["melodic"]
    assert pending["pending_config"]["mutation"] == 0.61
    assert client.get("/api/loop-mutator/tracks/percussion/versions").status_code == 404
    assert server.loop_mutator.backend is server.engine
