import pytest
from scipy import constants


def test_bandwidth(p):
    assert p.bandwidth == pytest.approx(1.2e9)


def test_range_resolution(p):
    assert p.range_resolution == pytest.approx(constants.c / (2 * 1.2e9))  # c / 2 * 1.2 GHz


def test_max_range(p):
    assert p.max_range == pytest.approx(
        p.fs * constants.c / (2 * p.slope)
    )  # 10 Mhz * c / (2 * 30 Mhz/us)


def test_velocity_resolution(p):
    assert p.velocity_resolution == pytest.approx(
        (constants.c / p.fc) / (2 * p.n_chirps * p.Tc)
    )  # lambda / (2 * 128 * 40 us)


def test_max_velocity(p):
    assert p.max_velocity == pytest.approx(
        (constants.c / p.fc) / (4 * p.Tc)
    )  # lambda / (4 * 40 us)
