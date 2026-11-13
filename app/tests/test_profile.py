import numpy as np
import pytest

from cadenza_app import profile

FLAT = [30] * 8
SLOPING = [25, 30, 40, 50, 60, 70, 80, 80]


def test_parse_levels_ok():
    assert profile.parse_levels(SLOPING).shape == (8,)


@pytest.mark.parametrize("bad", [[1] * 7, [1] * 9, ["a"] * 8, [200] * 8, [-50] * 8, [float("nan")] * 8])
def test_parse_levels_rejects(bad):
    with pytest.raises(ValueError):
        profile.parse_levels(bad)


def test_compressor_params_shape_and_monotone():
    params = profile.compressor_params(profile.parse_levels(SLOPING), profile.parse_levels(FLAT))
    left, right = params["left"], params["right"]
    assert len(left["makeup_gain"]) == 6 and len(right["makeup_gain"]) == 6
    assert left["makeup_gain"] == sorted(left["makeup_gain"])  # loss grows with frequency
    assert all(g >= 0 for g in left["makeup_gain"])
    assert left["ratio"] == right["ratio"] == profile.COMPRESSION_RATIO


def test_no_loss_means_no_makeup_gain():
    params = profile.compressor_params(profile.parse_levels([10] * 8), profile.parse_levels([10] * 8))
    assert params["left"]["makeup_gain"] == [0.0] * 6


def test_default_gains_flat_loss_is_zero():
    g = profile.default_gains(["Flute", "Oboe", "Bassoon"], np.array(FLAT), np.array(FLAT))
    assert g == [0.0, 0.0, 0.0]


def test_default_gains_boost_the_worse_heard_instrument():
    lv = np.array(SLOPING)
    g = dict(zip(["Flute", "Bassoon"], profile.default_gains(["Flute", "Bassoon"], lv, lv)))
    assert g["Flute"] > 0 > g["Bassoon"]       # high-frequency loss favours the flute
    assert abs(sum(g.values())) < 0.11          # zero-mean (up to rounding)
    assert max(abs(x) for x in g.values()) <= profile.MAX_AUTO_GAIN_DB


def test_default_gains_unknown_instrument():
    with pytest.raises(ValueError):
        profile.default_gains(["Tuba"], np.array(FLAT), np.array(FLAT))
