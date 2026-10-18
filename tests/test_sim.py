import pytest
from radar.params import RadarParams
from radar.sim import beat_signal
import numpy as np

@pytest.fixture
def p():
    return RadarParams(
        fc=77e9, slope=30e12, Tc=40e-6, fs=10e6,
        n_samples=400, n_chirps=128, n_channels=8
    )

def test_single_target_range_bin(p):
    R = 20.0
    s = beat_signal(R, p)
    spectrum = np.abs(np.fft.fft(s))
    assert np.argmax(spectrum) == round(R / p.range_resolution) # 160
