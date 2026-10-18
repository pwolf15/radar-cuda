import pytest

from radar.params import RadarParams


@pytest.fixture
def p():
    return RadarParams(
        fc=77e9, slope=30e12, Tc=40e-6, fs=10e6,
        n_samples=400, n_chirps=128, n_channels=8
    )
