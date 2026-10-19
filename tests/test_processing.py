import numpy as np

from radar.processing import ca_cfar_2d, doppler_fft, peaks, power_map, range_fft
from radar.sim import adc_cube


def test_range_doppler_chain(p, expected_cells):
    targets = [(20, 5.0, np.deg2rad(20.0), 1.0), (35, -5.0, np.deg2rad(-20.0), 0.05)]
    noise_stddev = 1.0
    cube = adc_cube(p, targets, noise_stddev=noise_stddev, seed=42)
    power = power_map(doppler_fft(range_fft(cube)))
    expected = expected_cells(targets)
    found = {tuple(c) for c in peaks(power)[:2]}
    assert found == expected


def test_cfar_false_alarm_rate(p):
    """On noise alone the detection fraction matches the requested Pfa."""
    cube = adc_cube(p, targets=[], noise_stddev=1.0, seed=0)
    rd = doppler_fft(range_fft(cube))

    # single channel: alpha is exact only for exponential noise
    power_0 = np.abs(rd[0]) ** 2
    pfa = 1e-3
    mask = ca_cfar_2d(power_0, pfa=pfa)
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

    expected = expected_cells(targets)
    for cell in expected:
        assert mask[cell]

    found = {tuple(c) for c in peaks(power, mask, 5)}
    assert found == expected
