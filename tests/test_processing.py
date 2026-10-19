import pytest
import numpy as np
from scipy.ndimage import maximum_filter
from radar.processing import doppler_fft, power_map, range_fft
from radar.sim import adc_cube

def test_range_doppler_chain(p):
    targets = [
        (20, 5.0, np.deg2rad(20.0), 1.0),
        (35, -5.0, np.deg2rad(-20.0), 0.05)
    ]
    noise_stddev = 1.0
    cube = adc_cube(p, targets, noise_stddev=noise_stddev, seed=42)
    power = power_map(doppler_fft(range_fft(cube)))
    expected = {(p.n_chirps // 2 + round(v / p.velocity_resolution),
                 round(R / p.range_resolution)) for R, v, _, _ in targets}
    peaks = (power == maximum_filter(power, size=3))
    cand = np.flatnonzero(peaks)
    top2 = cand[np.argsort(power.flat[cand])[-2:]]
    found = {tuple(int(x) for x in np.unravel_index(i, power.shape)) for i in top2}
    assert found == expected