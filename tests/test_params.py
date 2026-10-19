import pytest


def test_bandwidth(p):
    assert p.bandwidth == pytest.approx(1.2e9)


def test_range_resolution(p):
    assert p.range_resolution == pytest.approx(0.125, rel=1e-3)  # c / 2 * 1.2 GHz


def test_max_range(p):
    assert p.max_range == pytest.approx(50.0, rel=1e-3)  # 10 Mhz * c / (2 * 30 Mhz/us)


def test_velocity_resolution(p):
    assert p.velocity_resolution == pytest.approx(0.380, rel=1e-2)  # lambda / (2 * 128 * 40 us)


def test_max_velocity(p):
    assert p.max_velocity == pytest.approx(24.3, rel=1e-2)  # lambda / (4 * 40 us)
