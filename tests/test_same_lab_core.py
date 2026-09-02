import numpy as np
import pytest

from same_lab import (
    OPERATIONS,
    apply_profile,
    intervene,
    make_lfo,
    mix_channels,
    parse_channels,
    seconds_to_frames,
    time_crossfade,
)


def latent():
    return np.linspace(-1, 1, 256 * 20, dtype=np.float32).reshape(256, 20)


def test_channel_parser_and_interventions_are_backend_neutral():
    source = latent()
    assert parse_channels("0, 3-5, 255") == [0, 3, 4, 5, 255]
    edited = intervene(source, [0, 1], 2, 10, OPERATIONS[0])
    assert np.all(edited[:2, 2:10] == 0)
    assert np.array_equal(edited[2:], source[2:])
    with pytest.raises(ValueError):
        parse_channels("256")


@pytest.mark.parametrize("waveform", ["Sine", "Triangle", "Square", "Sawtooth"])
def test_lfo_waveforms(waveform):
    result = make_lfo(64, 8.0, waveform, 45.0)
    assert result.shape == (64,)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) <= 1.00001


def test_two_source_mix_crossfade_and_profile():
    a = latent()
    b = a[::-1].copy()
    mixed = mix_channels(a, b, [0.5] * 256)
    assert np.allclose(mixed, (a + b) / 2)
    crossed = time_crossfade(a, b, 2, 18, "A to B", "Smoothstep")
    assert crossed.shape == a.shape
    assert np.array_equal(crossed[:, :2], a[:, :2])
    assert np.array_equal(crossed[:, 18:], b[:, 18:])
    profile = [0.0] * 256
    profile[12] = 0.5
    profiled = apply_profile(a, profile, 4, 9, "Offset")
    assert np.allclose(profiled[12, 4:9], a[12, 4:9] + 0.5)
    assert np.array_equal(profiled[11], a[11])


def test_time_range_conversion():
    start, end = seconds_to_frames(0.0, 1.0, 20, 1.5)
    assert start == 0
    assert 1 <= end <= 20
    with pytest.raises(ValueError):
        seconds_to_frames(1.0, 0.5, 20, 1.5)
