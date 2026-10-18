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

def test_single_target_range_doppler_bins(p):
    R = 20.0    # target distance
    v = 5       # target velocity
    sig = beat_signal(R, p, v) # (n_chirps, n_samples)
    range_spec = np.fft.fft(sig, axis=1)
    rd = np.fft.fftshift(np.fft.fft(range_spec, axis=0), axes=0)
    (doppler_bin, range_bin) = np.unravel_index(np.argmax(np.abs(rd)), rd.shape)
    assert doppler_bin - p.n_chirps // 2 == round(v / p.velocity_resolution)
    assert range_bin == round(R / p.range_resolution)