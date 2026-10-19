import numpy as np
import pytest

from radar.processing import doppler_fft, range_fft
from radar.sim import adc_cube

radar_cuda = pytest.importorskip("radar_cuda")


def test_cuda_range_doppler_matches_numpy(p, expected_cells):
    targets = [(20, 5.0, np.deg2rad(20.0), 1.0), (35, -5.0, np.deg2rad(-20.0), 0.05)]
    noise_stddev = 1.0
    cube = adc_cube(p, targets, noise_stddev=noise_stddev, seed=42)
    numpy = doppler_fft(range_fft(cube))
    cuda = radar_cuda.range_doppler(cube)
    err = np.abs(cuda - numpy).max() / np.abs(numpy).max()
    assert err < 1e-3, f"max rel err {err:.2e}"
