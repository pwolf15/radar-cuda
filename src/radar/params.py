from dataclasses import dataclass

from scipy import constants


@dataclass
class RadarParams:
    fc: float  # carrier, Hz
    slope: float  # Hz/s
    Tc: float  # chirp duration
    fs: float  # ADC sample rate
    n_samples: int
    n_chirps: int
    n_channels: int

    @property
    def wavelength(self):
        return constants.c / self.fc

    @property
    def bandwidth(self):  # slope * Tc
        return self.slope * self.Tc

    @property
    def range_resolution(self):  # c / (2 B)
        return constants.c / (2 * self.bandwidth)

    @property
    def max_range(self):  # fs * c / (2 slope)
        return self.fs * constants.c / (2 * self.slope)

    @property
    def velocity_resolution(self):
        # bin spacing
        return self.wavelength / (2 * self.n_chirps * self.Tc)

    @property
    def max_velocity(self):
        # nyquist edge
        return self.wavelength / (4 * self.Tc)
