"""NumPy reference implementation of the FMCW range-Doppler-angle chain
range_fft -> doppler_fft -> power_map -> ca_cfar_2d -> peaks -> angle_fft -> to_physical

axis convention: (C, M, N)
"""

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter


def _window(n, kind):
    if kind is None:
        return np.ones(n)
    if kind == "hann":
        return np.hanning(n)
    raise ValueError(f"unknown window {kind!r}")


def range_fft(cube, window="hann"):
    """Windowed FFT along fast time (axis 2): (C, M, N) ADC cube -> range spectrum."""
    n_samples = cube.shape[2]
    x_w = cube * _window(n_samples, window)[None, None, :]
    return np.fft.fft(x_w, axis=2)


def doppler_fft(x, window="hann"):
    """Windowed FFT along slow time (axis 1), fftshifted so index M//2 is zero Doppler."""
    n_chirps = x.shape[1]
    x_w = x * _window(n_chirps, window)[None, :, None]
    return np.fft.fftshift(np.fft.fft(x_w, axis=1), axes=1)


def power_map(rd):
    """|.|^2 summed over channels (non-coherent integration): (C, M, N) -> (M, N)."""
    return np.sum(np.abs(rd) ** 2, axis=0)


def ca_cfar_2d(power, guard=(2, 2), train=(4, 8), pfa=1e-4):
    """2-D cell-averaging CFAR on a (M,N) range-doppler power map.

    For each cell under test (CUT), noise level is mean of training
    cells: a (2(g_d+t_d)+1) x (2(g_r+t_r)+1) box minus the (2g_d+1) x (2g_r+1)
    guard box centered on the CUT. guard/train are (doppler,range) half-widths;
    the window is asymmetric because range leakage is wider than Doppler leakage after Hann
    windowing. Detection: power > alpha * noise with alpha = n_train * (pfa**(-1/n_train) - 1),
    the exact threshold for exponentially distributed (square-law, single-channel) noise.
    On a channel-summed map the noise is Gamma-distributed and this alpha is conservative
    (fewer false alarms than pfa).

    Box sums are two separable uniform filters (outer - inner). Doppler is
    padded with wrap (slow-time FFT is periodic); range with edge (no
    physical neighbor beyond 0 or max range). Returns a bool mask (M,N).
    """
    assert power.ndim == 2, power.shape

    # bboxes for train and guard
    g_d, g_r = guard
    t_d, t_r = train
    inner = (2 * g_d + 1, 2 * g_r + 1)
    outer = (2 * (g_d + t_d) + 1, 2 * (g_r + t_r) + 1)
    n_inner = inner[0] * inner[1]
    n_outer = outer[0] * outer[1]
    n_train = n_outer - n_inner

    pd, pr = g_d + t_d, g_r + t_r
    # pad with 'wrap'; doppler is circular (slow-time FFT is periodic)
    padded = np.pad(power, pad_width=((pd, pd), (0, 0)), mode="wrap")
    # assume noise beyond edge looks like noise at edge
    padded = np.pad(padded, ((0, 0), (pr, pr)), mode="edge")

    # zero padding never reaches a kept cell: crop below
    mean_outer = uniform_filter(padded, size=outer, mode="constant")
    sum_outer = mean_outer * n_outer
    mean_inner = uniform_filter(padded, size=inner, mode="constant")
    sum_inner = mean_inner * n_inner

    sum_train = sum_outer - sum_inner
    noise = sum_train / n_train

    alpha = n_train * (pfa ** (-1 / n_train) - 1)
    mask = padded > alpha * noise

    return mask[pd:-pd, pr:-pr]


def peaks(power, mask=None, size=3):
    """(d_bin, r_bin) of local maxima of power, optionally restricted to mask,
    sorted by descending power. Returns an (K, 2) int array."""
    is_max = power == maximum_filter(power, size=size)
    if mask is not None:
        is_max &= mask
    cells = np.argwhere(is_max)
    order = np.argsort(power[cells[:, 0], cells[:, 1]])[::-1]
    return cells[order]


def angle_fft(rd, cells, n_fft=64):
    """Angle of arrival for each (d_bin,r_bin) cell: zero-padded
    FFT across channels, peak bin -> theta = arcsin(2k/n_fft).
    Returns radians, shape (K,).

    theta > 0 right of boresight, matching sim.beat_signal
    resolution = 2/n_fft in sin(theta)
    """
    theta_hats = []
    for d, r in cells:
        ang = np.fft.fftshift(np.fft.fft(rd[:, d, r], n=n_fft))
        a_bin = np.argmax(np.abs(ang)) - n_fft // 2
        theta_hats.append(np.arcsin(2 * a_bin / n_fft))
    return np.array(theta_hats)


def to_physical(params, cells):
    """(d_bin, r_bin) -> (R [m], v [m/s]) using bin centers.

    Returns (R,v) arrays of shape (K,)
    """
    cells = np.asarray(cells)
    R = cells[:, 1] * params.range_resolution
    v = (cells[:, 0] - params.n_chirps // 2) * params.velocity_resolution
    return R, v
