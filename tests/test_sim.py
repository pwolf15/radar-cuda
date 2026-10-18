import pytest
from radar.sim import beat_signal
import numpy as np

@pytest.mark.parametrize("R, v, theta_deg", [
    (20.0, 5.0, 20.0),
    (20.0, -5.0, -20.0),
    (10.0, 0.0, 0.0),
    (45.0, 12.0, 45.0)
])
def test_single_target_angle_range_doppler_bins(p, R, v, theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    sig = beat_signal(p, R, v, theta_rad)
    N_fft = 64
    rd = np.fft.fft(np.fft.fft(sig, axis=2), axis=1)
    rd = np.fft.fftshift(rd, axes=1)
    _, d_bin, r_bin = np.unravel_index(np.argmax(np.abs(rd)), rd.shape)
    ang = np.fft.fftshift(np.fft.fft(rd[:, d_bin, r_bin], n=N_fft))
    a_bin = np.argmax(np.abs(ang)) - N_fft // 2
    theta_hat = np.arcsin(2 * a_bin / N_fft)
    assert abs(theta_hat - theta_rad) < np.arcsin(2 / N_fft)
    assert r_bin == round(R / p.range_resolution)
    assert d_bin - p.n_chirps // 2 == round(v / p.velocity_resolution)
