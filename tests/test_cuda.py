import numpy as np
import pytest

from radar.processing import (
    range_fft,
)
from radar.sim import adc_cube

radar_cuda = pytest.importorskip("radar_cuda")


def test_cuda_range_fft_matches_numpy(p, expected_cells):
    targets = [(20, 5.0, np.deg2rad(20.0), 1.0), (35, -5.0, np.deg2rad(-20.0), 0.05)]
    noise_stddev = 1.0
    cube = adc_cube(p, targets, noise_stddev=noise_stddev, seed=42)
    numpy = range_fft(cube)
    cuda = radar_cuda.range_doppler(cube)
    assert np.allclose(cuda, numpy, rtol=1e-3, atol=1e-2 * np.abs(numpy).max())
