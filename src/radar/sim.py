import numpy as np
from scipy import constants

from radar.params import RadarParams


def beat_signal(params: RadarParams, R, v=0.0, theta=0.0):
    """Ideal beat signal of a single point target after dechirp mixing.

    Returns a complex (I/Q) array of shape (n_channels, n_chirps, n_samples):
    axis 0 = receive channel, axis 1 = slow time (chirp), axis 2 = fast time
    (ADC sample). The signal is a unit-amplitude complex exponential whose
    phase ramps linearly along each axis:

        fast time : 2*pi * f_b * t,     f_b = 2*slope*R/c    -> range
        slow time : 2*pi * f_d * m*Tc,  f_d = 2*v/wavelength -> Doppler
        channel   : 2*pi * (d/wavelength) * n * sin(theta)   -> angle

    Conventions:
        v > 0       target approaching (positive Doppler frequency)
        theta > 0   right of boresight (positive spatial frequency), radians
        d = wavelength/2, uniform linear array

    Not modeled: amplitude / range attenuation, noise, range migration
    within a frame, intra-chirp Doppler, multiple targets.
    """
    # fast-time vector
    t = np.arange(params.n_samples) / params.fs

    # beat frequency
    f_b = 2.0 * params.slope * R / constants.c

    # slow-time vector
    m = np.arange(params.n_chirps)

    # Doppler frequency
    f_d = 2.0 * v / params.wavelength

    # angle vector
    n = np.arange(params.n_channels)
    d_over_lambda = 0.5  # half-wavelength spacing

    phase = (
        2
        * np.pi
        * (
            d_over_lambda * n[:, None, None] * np.sin(theta)
            + f_b * t[None, None, :]
            + f_d * m[None, :, None] * params.Tc
        )
    )
    return np.exp(1j * phase)


def adc_cube(
    params: RadarParams, targets: list[tuple[float, float, float, float]], noise_stddev=0.0, seed=0
):
    """Sum of point-target beat signals plus complex white Gaussian noise.

    targets: iterable of (R [m], v [m/s], theta [rad], amplitude);
    noise_stddev: std of complex Gaussian: sigma/sqrt(2) per component so
    E|noise|^2 = noise_stddev^2 SNR per ADC sample is roughly
    amplitude^2/noise_stddev^2. The three FFTs add
    10*log10(n_samples*n_chirps*n_channels) dB of processing gain,
    so targets well below the per-sample noise floor are still detectable
    """
    signal = np.zeros((params.n_channels, params.n_chirps, params.n_samples), dtype=np.complex128)
    for R, v, theta, amplitude in targets:
        signal += amplitude * beat_signal(params, R, v, theta)
    if noise_stddev > 0:
        rng = np.random.default_rng(seed)
        sigma = noise_stddev / np.sqrt(2)
        noise = rng.normal(0, sigma, signal.shape) + 1j * rng.normal(0, sigma, signal.shape)
        signal += noise
    return signal
