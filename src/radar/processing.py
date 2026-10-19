import numpy as np

def _window(n, kind):
    if kind is None: return np.ones(n)
    if kind == "hann": return np.hanning(n)
    raise ValueError(f"unknown window {kind!r}")

def range_fft(cube, window="hann"):
    """Windowed FFT along fast time (axis 2): (C, M, N) ADC cube -> range spectrum."""
    n_samples = cube.shape[2]
    sig = cube * _window(n_samples, window)[None, None, :]
    return np.fft.fft(sig, axis=2)

def doppler_fft(x, window="hann"):
    """Windowed FFT along slow time (axis 1), fftshifted so index M//2 is zero Doppler."""
    n_chirps = x.shape[1]
    rd = x * _window(n_chirps, window)[None, :, None]
    return np.fft.fftshift(np.fft.fft(rd, axis=1), axes=1)

def power_map(rd):
    """|.|^2 summed over channels (non-coherent integration): (C, M, N) -> (M, N)."""
    return np.sum(np.abs(rd)**2, axis=0)