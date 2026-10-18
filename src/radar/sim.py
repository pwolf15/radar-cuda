from radar.params import RadarParams
from scipy import constants
import numpy as np

def beat_signal(params: RadarParams, R, v=0.0, theta=0.0):
    """Ideal beat signal of a single point target after dechipr mixing.

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
    d_over_lambda = 0.5 # half-wavelength spacing

    phase = 2 * np.pi * (d_over_lambda * n[:, None, None] * np.sin(theta) + f_b * t[None, None, :] + f_d * m[None,:,None] * params.Tc)
    return np.exp(1j * phase)