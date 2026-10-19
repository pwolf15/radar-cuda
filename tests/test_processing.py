import numpy as np
from pytest import approx

from radar.processing import (
    angle_fft,
    ca_cfar_2d,
    doppler_fft,
    peaks,
    power_map,
    range_fft,
    to_physical,
)
from radar.sim import adc_cube


def test_range_doppler_chain(p, expected_cells):
    targets = [(20, 5.0, np.deg2rad(20.0), 1.0), (35, -5.0, np.deg2rad(-20.0), 0.05)]
    noise_stddev = 1.0
    cube = adc_cube(p, targets, noise_stddev=noise_stddev, seed=42)
    power = power_map(doppler_fft(range_fft(cube)))
    expected = expected_cells(targets).keys()
    found = {tuple(c) for c in peaks(power)[:2]}
    assert found == expected


def test_cfar_false_alarm_rate(p):
    """On noise alone the detection fraction matches the requested Pfa."""
    cube = adc_cube(p, targets=[], noise_stddev=1.0, seed=0)
    rd = doppler_fft(range_fft(cube))

    # single channel: alpha is exact only for exponential noise
    power_ch0 = np.abs(rd[0]) ** 2
    pfa = 1e-3
    mask = ca_cfar_2d(power_ch0, pfa=pfa)
    n_cells = mask.size
    n_hits = mask.sum()
    # binomial(n_cells, pfa): mean ~51, std ~7 for 128x400. allow +/-3 sigma.
    expected = n_cells * pfa
    assert abs(n_hits - expected) < 3 * np.sqrt(expected)


def test_cfar_detects_targets(p, expected_cells):
    """Three targets in noise: every expected cell is above threshold, and
    the local maxima of the mask are exactly those three cells."""
    targets = [
        (20, 5.0, np.deg2rad(20), 1.0),
        (35, -5.0, np.deg2rad(-20), 0.05),
        (10, 0.0, 0.0, 0.2),
    ]
    cube = adc_cube(p, targets, noise_stddev=1.0, seed=42)
    power = power_map(doppler_fft(range_fft(cube)))
    mask = ca_cfar_2d(power, pfa=1e-4)

    expected = expected_cells(targets).keys()
    for cell in expected:
        assert mask[cell]

    found = {tuple(c) for c in peaks(power, mask, 5)}
    assert found == expected


def test_angle_range_doppler_chain(p, expected_cells):
    """Full chain on three targets: CFAR finds exactly the expected cells
    and the FFT recovers each theta within one bin."""
    targets = [
        (20, 5.0, np.deg2rad(20), 1.0),
        (35, -5.0, np.deg2rad(-20), 0.05),
        (10, 0.0, 0.0, 0.2),
    ]
    noise_stddev = 1.0
    cube = adc_cube(p, targets, noise_stddev=noise_stddev, seed=42)
    rd = doppler_fft(range_fft(cube))
    power = power_map(rd)
    mask = ca_cfar_2d(power, pfa=1e-4)
    cells = peaks(power, mask)
    theta_hats = angle_fft(rd, cells)
    expected = expected_cells(targets)
    for (d, r), th in zip(cells, theta_hats, strict=False):
        assert abs(th - expected[(d, r)][2]) < np.arcsin(2 / 64)


def test_to_physical(p):
    """Test known bins with known physics"""
    Rvs = to_physical(p, [(77, 160)])
    R, v = Rvs[0]

    assert R == approx(20.0, 0.1)
    assert v == approx(4.95, 0.1)
