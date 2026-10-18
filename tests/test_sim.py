import pytest
from radar.sim import beat_signal, adc_cube
import numpy as np
from scipy.ndimage import maximum_filter

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

def test_adc_cube(p):
    """Sum of point-target beat signals plus complex white Gaussian noise.
    
    Two targets 26 dB apart in noise are both recovered after range-Doppler
    processing: proves noise is complex Gaussian and processing gain is real."""
    targets = [
        (20, 5.0, np.deg2rad(20.0), 1.0),
        (35, -5.0, np.deg2rad(-20.0), 0.05)
    ]
    noise_stddev = 1.0
    sig = adc_cube(p, targets, noise_stddev=noise_stddev, seed=42)
    # Hann on both FFT axes: rectangular-window sidelobes of the strong
    # target (-13 dB) would bury the weak one (-26 dB).
    w_fast = np.hanning(p.n_samples)[None, None, :]
    w_slow = np.hanning(p.n_chirps)[None, :, None]
    rd = np.fft.fft(sig * w_fast * w_slow, axis=2)
    rd = np.fft.fftshift(np.fft.fft(rd, axis=1), axes=1)
    power = np.sum(np.abs(rd) ** 2, axis=0)
    expected = {(p.n_chirps // 2 + round(v / p.velocity_resolution),
                round(R / p.range_resolution)) for R, v, _, _ in targets}
    # One cell per target: a peak occupies a 3x3 blob, so top-k cells
    # would return the strong target twice.
    peaks = (power == maximum_filter(power, size=3))
    cand = np.flatnonzero(peaks)
    top2 = cand[np.argsort(power.flat[cand])[-2:]]
    found = {tuple(int(x) for x in np.unravel_index(i, power.shape)) for i in top2 }
    assert found == expected