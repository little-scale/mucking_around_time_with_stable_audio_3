import json
import struct
import time
import wave

import numpy as np
from fastapi.testclient import TestClient

import server
from backends.base import BackendInfo, CONTRACT_VERSION, array_stats, heatmap_preview, waveform_preview
from protocol import CORE_EVENTS, validate_event


class StubBackend:
    name = "cuda"
    dit_choices = ("sm-music", "sm-sfx", "medium")
    decoder_choices = ("same-s", "same-l")
    max_seconds = 380.0
    sample_rate = 44_100

    def diagnostics(self):
        return BackendInfo("cuda", "cuda:0", "Mock RTX 4080", "float16", 16 * 2**30, 12 * 2**30, "PyTorch/CUDA", "mock")

    def generate(self, *, output_path, emit, steps, seed, **config):
        x = np.linspace(-1, 1, 256 * 8, dtype=np.float32).reshape(1, 256, 8)
        stats = array_stats(x)
        hm = heatmap_preview(x)
        now = time.time()
        common = {"time": now, "memory_bytes": 1234}
        emit({"type": "run_start", **common, "mode": config["mode"], "dit": config["dit_name"], "decoder": "same-s", "seconds": config["seconds"], "steps": steps, "seed": seed or 7, "latent_channels": 256, "t_lat": 8})
        for stage in ("t5", "conditioning"):
            emit({"type": "stage", **common, "stage": stage, "state": "active"})
            if stage == "t5": emit({"type": "embedding", **common, "role": "prompt", "tokens": 4, "stats": stats, "heatmap": hm})
            else: emit({"type": "conditioning", **common, "cross_attn": stats, "global_cond": stats})
            emit({"type": "stage", **common, "stage": stage, "state": "done"})
        emit({"type": "stage", **common, "stage": "encoder", "state": "skipped"})
        emit({"type": "stage", **common, "stage": "dit", "state": "active"})
        emit({"type": "sampler_init", **common, "total": steps, "sigma": 1.0, "stats": stats, "heatmap": hm})
        for i in range(steps):
            emit({"type": "sampler_step", **common, "step": i + 1, "total": steps, "sigma": 1 - i / steps, "ms": 1.0, "stats": stats, "heatmap": hm})
        emit({"type": "latent_final", **common, "stats": stats, "heatmap": hm})
        emit({"type": "stage", **common, "stage": "dit", "state": "done"})
        emit({"type": "stage", **common, "stage": "decoder", "state": "active"})
        emit({"type": "stage", **common, "stage": "decoder", "state": "done"})
        audio = np.zeros((2, 441), dtype=np.float32)
        with wave.open(output_path, "wb") as f:
            f.setnchannels(2); f.setsampwidth(2); f.setframerate(self.sample_rate)
            f.writeframes(struct.pack("<" + "h" * (audio.size), *([0] * audio.size)))
        emit({"type": "stage", **common, "stage": "output", "state": "active"})
        emit({"type": "output_audio", **common, "stats": array_stats(audio), "waveform": waveform_preview(audio), "sample_rate": self.sample_rate})
        emit({"type": "stage", **common, "stage": "output", "state": "done"})
        emit({"type": "run_complete", **common, "seed": seed or 7, "timing": {"total_ms": 1, "realtime": 1}})

    @property
    def same_codec_choices(self):
        return ("same-s", "same-l")

    def same_encode(self, input_path, codec_name):
        latent = np.linspace(-1, 1, 256 * 4, dtype=np.float32).reshape(256, 4)
        audio = np.zeros((2, 4096 * 4), dtype=np.float32)
        return {"latent": latent, "audio": audio, "target_samples": audio.shape[1], "sample_rate": self.sample_rate}

    def same_decode(self, latent, codec_name, target_samples):
        level = float(np.asarray(latent).mean())
        return np.full((2, target_samples), level, dtype=np.float32)


def test_shared_http_sse_contract_and_output(tmp_path):
    server.configure_backend(StubBackend(), tmp_path)
    client = TestClient(server.app)
    info = client.get("/api/info").json()
    assert info["contract_version"] == CONTRACT_VERSION
    assert info["backend"]["device_name"] == "Mock RTX 4080"
    response = client.post("/api/generate", data={"prompt": "test", "steps": "2", "seed": "9"})
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    for _ in range(100):
        status = client.get(f"/api/run/{run_id}").json()
        if status["status"] in {"complete", "error"}: break
        time.sleep(0.01)
    assert status["status"] == "complete"
    body = client.get(f"/api/events/{run_id}").text
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    kinds = {item["type"] for item in events}
    assert CORE_EVENTS <= kinds
    assert {"status", "stage", "sampler_step", "output_audio"} <= kinds
    assert all(item["contract_version"] == CONTRACT_VERSION for item in events)
    for item in events:
        validate_event(item)
    output = client.get(f"/api/output/{run_id}/generated.wav")
    assert output.status_code == 200 and output.headers["content-type"].startswith("audio/wav")


def test_frontend_has_runtime_backend_diagnostics():
    html = (server.STATIC / "monitor.html").read_text(encoding="utf-8")
    for marker in ("backendName", "deviceName", "vramTotal", "activeDtype", "/api/info"):
        assert marker in html
    assert "eventSource?.close();eventSource=null" in html
    assert "p.dataset.outputUrl===url" in html
    assert "Minimal monochrome interface chrome" in html


def test_root_is_landing_page_and_monitor_keeps_pipeline_interface():
    client = TestClient(server.app)
    landing = (server.STATIC / "index.html").read_text(encoding="utf-8")
    assert client.get("/").text == landing
    for href in ('href="/monitor"', 'href="/same-lab"', 'href="/sfx-matrix"', 'href="/drift"', 'href="/loop-mutator"'):
        assert href in landing
    assert "Graphical Pipeline Monitor" in landing
    assert "SAME Lab" in landing
    assert "SFX Matrix" in landing
    assert "Drift Looper" in landing
    assert "Loop Mutator" in landing
    assert "/api/info" in landing

    monitor = (server.STATIC / "monitor.html").read_text(encoding="utf-8")
    response = client.get("/monitor")
    assert response.status_code == 200
    assert response.text == monitor
    assert "backendName" in monitor


def test_every_interface_links_directly_to_the_other_sections():
    pages = {
        "index.html": ("/monitor", "/same-lab", "/sfx-matrix", "/drift", "/loop-mutator"),
        "monitor.html": ("/", "/same-lab", "/sfx-matrix", "/drift", "/loop-mutator"),
        "same-lab.html": ("/", "/monitor", "/sfx-matrix", "/drift", "/loop-mutator"),
        "sfx-matrix.html": ("/", "/monitor", "/same-lab", "/drift", "/loop-mutator"),
        "drift/index.html": ("/", "/monitor", "/same-lab", "/sfx-matrix", "/loop-mutator"),
        "loop-mutator.html": ("/", "/monitor", "/same-lab", "/sfx-matrix", "/drift"),
    }
    for filename, destinations in pages.items():
        html = (server.STATIC / filename).read_text(encoding="utf-8")
        for destination in destinations:
            assert f'href="{destination}"' in html, f"{filename} lacks {destination}"


def test_drift_uses_shared_monochrome_visual_system():
    html = (server.STATIC / "drift" / "index.html").read_text(encoding="utf-8")
    css = (server.STATIC / "drift" / "app.css").read_text(encoding="utf-8")
    assert "/static/drift/app.css?v=9" in html
    assert "/* Monochrome system shared with the other SA3 tools. */" in css
    assert ":root{--base:#000" in css
    assert ".track,.history-columns>div,.history-columns h3{--track:#fff!important}" in css
    assert "body::before,.ambient{display:none!important}" in css


def test_sfx_matrix_is_separate_and_uses_shared_protocol():
    html = (server.STATIC / "sfx-matrix.html").read_text(encoding="utf-8")
    assert "SFX Variation Matrix" in html
    assert "dit:'sm-sfx'" in html
    assert "fetch('/api/generate'" in html
    assert "new EventSource(data.events_url)" in html
    assert "/api/archive?run_ids=" in html
    assert "play-result" in html
    assert "Audition</div>" not in html
    assert "id=\"previous\"" not in html
    assert '<textarea id="basePrompt"></textarea>' in html
    assert '<textarea id="rowPrompts"></textarea>' in html
    assert '<textarea id="columnPrompts"></textarea>' in html
    assert '<input id="negativePrompt">' in html
    assert "/* Minimal monochrome interface */" in html
    assert "The prompt matrix will appear here." in html
    assert "mlx" not in html.lower()
    assert "cuda" not in html.lower()
    response = TestClient(server.app).get("/sfx-matrix")
    assert response.status_code == 200
    assert response.text == html


def test_same_lab_page_and_backend_neutral_session_api(tmp_path):
    server.configure_backend(StubBackend(), tmp_path)
    client = TestClient(server.app)
    html = (server.STATIC / "same-lab.html").read_text(encoding="utf-8")
    response = client.get("/same-lab")
    assert response.status_code == 200 and response.text == html
    assert "/api/same/info" in html
    assert "/api/same/encode" in html
    assert "/api/same/edit/" in html
    assert "mlx" not in html.lower()
    assert "cuda" not in html.lower()

    info = client.get("/api/same/info").json()
    assert info["codecs"] == ["same-s", "same-l"]
    assert info["osc"]["enabled"] is False
    encoded = client.post(
        "/api/same/encode",
        data={"codec": "same-s", "source": "a"},
        files={"audio": ("source.wav", b"stub", "audio/wav")},
    )
    assert encoded.status_code == 200, encoded.text
    result = encoded.json()
    session_id = result["session_id"]
    assert result["latent_shape"] == [256, 4]
    assert result["backend"] == "cuda"
    assert client.get(result["audio"]["baseline"]).status_code == 200
    assert client.get(result["latent_url"]).status_code == 200
    assert client.post(f"/api/same/osc/arm/{session_id}").status_code == 503

    edited = client.post(
        f"/api/same/edit/{session_id}",
        json={
            "action": "intervention",
            "channels": "0-3",
            "operation": "Zero / ablate selected",
            "start": 0,
            "end": result["duration"],
            "cumulative": False,
        },
    )
    assert edited.status_code == 200, edited.text
    assert "Applied Zero / ablate selected" in edited.json()["status"]


def test_same_lab_two_source_mix_and_profile(tmp_path):
    server.configure_backend(StubBackend(), tmp_path)
    client = TestClient(server.app)
    a = client.post(
        "/api/same/encode",
        data={"codec": "same-l", "source": "a"},
        files={"audio": ("a.wav", b"a", "audio/wav")},
    ).json()
    sid = a["session_id"]
    b = client.post(
        "/api/same/encode",
        data={"codec": "same-l", "source": "b", "session_id": sid},
        files={"audio": ("b.wav", b"b", "audio/wav")},
    )
    assert b.status_code == 200 and b.json()["has_source_b"] is True
    mix = client.post(f"/api/same/edit/{sid}", json={"action": "mix", "values": [0.5] * 256})
    assert mix.status_code == 200
    profile = client.post(
        f"/api/same/edit/{sid}",
        json={"action": "profile", "values": [0.0] * 256, "mode": "Offset", "start": 0, "end": a["duration"]},
    )
    assert profile.status_code == 200


def test_classroom_request_log_contains_ip_prompt_and_conditions(tmp_path, capsys):
    state = server.RunState(
        run_id="abc123def456",
        run_dir=tmp_path,
        client_ip="192.168.0.145",
        config={
            "prompt": "bright classroom percussion",
            "negative_prompt": "vocals",
            "mode": "text",
            "dit_name": "medium",
            "decoder_name": "auto",
            "seconds": 20.0,
            "steps": 8,
            "seed": 42,
            "cfg": 1.0,
            "apg": 1.0,
            "sigma_max": 1.0,
        },
    )
    server._terminal_log("started", state, waiting=2)
    line = capsys.readouterr().out.strip()
    assert line.startswith("SA3_REQUEST ")
    payload = json.loads(line.removeprefix("SA3_REQUEST "))
    assert payload["event"] == "started"
    assert payload["ip"] == "192.168.0.145"
    assert payload["prompt"] == "bright classroom percussion"
    assert payload["model"] == "medium"
    assert payload["waiting"] == 2


def test_backend_failure_is_streamed_and_persisted(tmp_path):
    class FailingBackend(StubBackend):
        def generate(self, **kwargs):
            raise RuntimeError("mock accelerator failure")

    server.configure_backend(FailingBackend(), tmp_path)
    client = TestClient(server.app)
    response = client.post("/api/generate", data={"prompt": "fail"})
    run_id = response.json()["run_id"]
    for _ in range(100):
        status = client.get(f"/api/run/{run_id}").json()
        if status["status"] == "error": break
        time.sleep(0.01)
    assert status["status"] == "error"
    events = [json.loads(line[6:]) for line in client.get(f"/api/events/{run_id}").text.splitlines() if line.startswith("data: ")]
    failure = next(item for item in events if item["type"] == "error")
    assert "mock accelerator failure" in failure["message"]
    assert (tmp_path / run_id / "error.txt").exists()
