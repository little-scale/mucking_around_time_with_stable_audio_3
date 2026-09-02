import numpy as np

from same_lab import OPERATIONS
from same_osc import SameOSCController


def test_osc_requires_explicit_session_and_mirrors_controls():
    osc = SameOSCController()
    osc.enabled = True
    armed = osc.arm("0123456789abcdef", 3.25)
    assert armed["armed"] is True
    assert armed["controls"]["end"] == 3.25

    osc._handle("/same/set/channels", "0-31,64")
    osc._handle("/same/set/operation", "lfo_add")
    osc._handle("/same/set/lfo_depth", 0.75)
    snapshot = osc.snapshot("0123456789abcdef")
    assert snapshot["controls"]["channels"] == "0-31,64"
    assert snapshot["controls"]["operation"] == OPERATIONS[10]
    assert snapshot["controls"]["lfo_depth"] == 0.75


def test_osc_mix_and_profile_banks_support_channel_and_bulk_messages():
    osc = SameOSCController()
    osc.enabled = True
    osc.arm("0123456789abcdef", 2.0)
    osc._handle("/same/mix/12", 0.75)
    osc._handle("/same/profile", 7, -0.4)
    snapshot = osc.snapshot("0123456789abcdef")
    assert snapshot["mix"][12] == 0.75
    assert np.isclose(snapshot["profile"][7], -0.4)

    osc._handle("/same/mix/all", 0.5)
    osc._handle("/same/profile/random", 0.25, 42)
    snapshot = osc.snapshot("0123456789abcdef")
    assert snapshot["mix"] == [0.5] * 256
    assert len(snapshot["profile"]) == 256
    assert max(abs(x) for x in snapshot["profile"]) <= 0.25001


def test_osc_actions_build_the_same_http_payloads():
    osc = SameOSCController()
    osc.enabled = True
    osc.arm("0123456789abcdef", 4.0)
    _, state = osc._active()
    state.controls["channels"] = "4-8"
    state.controls["operation"] = OPERATIONS[9]
    payload = osc._payload(state, "intervention")
    assert payload["action"] == "intervention"
    assert payload["channels"] == "4-8"
    assert payload["operation"] == OPERATIONS[9]
    assert len(osc._payload(state, "mix")["values"]) == 256
    assert len(osc._payload(state, "profile")["values"]) == 256


def test_disarm_and_unarmed_messages_are_rejected_in_status():
    osc = SameOSCController()
    osc.enabled = True
    osc.arm("0123456789abcdef", 1.0)
    result = osc.disarm("0123456789abcdef")
    assert result["armed"] is False
    osc._handle("/same/set/amount", 1.0)
    assert osc.snapshot("0123456789abcdef")["armed"] is False
