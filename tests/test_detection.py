import sys
from types import SimpleNamespace

import pytest

from backends.base import BackendUnavailable, auto_backend_name


def test_auto_detects_usable_cuda(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "torch" else None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    assert auto_backend_name() == "cuda"


def test_auto_detects_mlx_on_apple_silicon(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "mlx" else None)
    assert auto_backend_name() == "mlx"


def test_auto_detection_fails_actionably(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(BackendUnavailable, match="--backend cuda"):
        auto_backend_name()

