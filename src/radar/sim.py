from radar.params import RadarParams
from scipy import constants
import numpy as np

def beat_signal(R, params: RadarParams, v=0.0):
    """One target at range R, one chirp, one channel, no noise"""
    """positive v = approaching"""
    # fast-time vector
    t = np.arange(params.n_samples) / params.fs
    # beat frequency
    f_b = 2.0 * params.slope * R / constants.c
    # slow-time vector
    m = np.arange(params.n_chirps)
    f_d = 2.0 * v / params.wavelength
    
    phase = 2 * np.pi * (f_b * t[None, :] + f_d * m[:,None] * params.Tc)
    return np.exp(1j * phase)