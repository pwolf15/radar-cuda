import numpy as np
import pytest

from radar.processing import ca_cfar_2d, doppler_fft, peaks, power_map, range_fft
from radar.sim import adc_cube

radar_cuda = pytest.importorskip("radar_cuda")


def test_cuda_range_doppler_matches_numpy(cube):
    numpy = doppler_fft(range_fft(cube))
    cuda = radar_cuda.range_doppler(cube)
    err = np.abs(cuda - numpy).max() / np.abs(numpy).max()
    assert err < 1e-3, f"max rel err {err:.2e}"


def test_cuda_power_map_matches_numpy(cube):
    numpy_rd = doppler_fft(range_fft(cube))
    numpy = power_map(numpy_rd)
    cuda = radar_cuda.power_map(numpy_rd)
    err = np.abs(cuda - numpy).max() / np.abs(numpy).max()
    assert err < 1e-3, f"max rel err {err:.2e}"


def test_cuda_chain_power_matches_numpy(cube):
    cuda = radar_cuda.power_map(radar_cuda.range_doppler(cube))
    ref = power_map(doppler_fft(range_fft(cube)))
    err = np.abs(cuda - ref).max() / ref.max()
    assert err < 1e-3, f"max rel err {err:.2e}"


def test_cuda_cfar_false_alarm_rate(p):
    cube = adc_cube(p, targets=[], noise_stddev=1.0, seed=0)
    rd = doppler_fft(range_fft(cube))
    power_ch0 = np.abs(rd[0]) ** 2
    pfa = 1e-3

    mask_cuda = radar_cuda.ca_cfar_2d(power_ch0, pfa=pfa)
    n_cells = mask_cuda.size
    n_hits_cuda = mask_cuda.sum()

    mask_numpy = ca_cfar_2d(power_ch0, pfa=pfa)
    n_hits_numpy = mask_numpy.sum()
    assert n_hits_cuda == n_hits_numpy

    expected_cuda = n_cells * pfa
    assert abs(n_hits_cuda - expected_cuda) < 3 * np.sqrt(expected_cuda)


def test_cuda_cfar_mask_matches_numpy(cube):
    power = radar_cuda.power_map(radar_cuda.range_doppler(cube))
    mask_cuda = radar_cuda.ca_cfar_2d(power)
    mask_ref = ca_cfar_2d(power)
    assert np.array_equal(mask_cuda, mask_ref)


def test_cuda_cfar_detects_targets(p, expected_cells):
    targets = [
        (20, 5.0, np.deg2rad(20), 1.0),
        (35, -5.0, np.deg2rad(-20), 0.05),
        (10, 0.0, 0.0, 0.2),
    ]
    cube = adc_cube(p, targets, noise_stddev=1.0, seed=42)
    power = radar_cuda.power_map(radar_cuda.range_doppler(cube))
    mask = radar_cuda.ca_cfar_2d(power, pfa=1e-4)

    expected = set(expected_cells(targets))
    for cell in expected:
        assert mask[cell]

    found = {tuple(c) for c in peaks(power, mask, 5)}
    assert found == expected


def test_cuda_pipeline_matches_stages(cube):
    pipe = radar_cuda.RadarPipeline(*cube.shape)
    r = pipe.process(cube)
    power = radar_cuda.power_map(radar_cuda.range_doppler(cube))
    assert np.allclose(r["power"], power, rtol=1e-5)
    assert np.array_equal(r["mask"], radar_cuda.ca_cfar_2d(power))
    r2 = pipe.process(cube)
    assert np.array_equal(r2["mask"], r["mask"])


def test_cuda_pipeline_in_place_matches_stages(cube):
    pipe = radar_cuda.RadarPipeline(*cube.shape)
    pipe.input[...] = cube
    r = pipe.process()
    power = radar_cuda.power_map(radar_cuda.range_doppler(cube))
    assert np.allclose(r["power"], power, rtol=1e-5)
    assert np.array_equal(r["mask"], radar_cuda.ca_cfar_2d(power))
    r2 = pipe.process(cube)
    assert np.array_equal(r2["mask"], r["mask"])
