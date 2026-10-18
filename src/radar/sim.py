from radar.params import RadarParams
from scipy import constants
import numpy as np

def beat_signal(R, params: RadarParams):
    """One target at range R, one chirp, one channel, no noise"""
    # fast-time vector
    t = np.arange(params.n_samples) / params.fs
    # beat frequency
    f_b = 2 * params.slope * R / constants.c
    return np.exp(1j * 2 * np.pi * f_b * t)