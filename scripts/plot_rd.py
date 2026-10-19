"""Range-Doppler map with CFAR detections for three-target synthetic frame."""

import matplotlib.pyplot as plt
import numpy as np

from radar.params import RadarParams
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

p = RadarParams(fc=77e9, slope=30e12, Tc=40e-6, fs=10e6, n_samples=400, n_chirps=128, n_channels=8)
targets = [(20, 5.0, np.deg2rad(20), 1.0), (35, -5.0, np.deg2rad(-20), 0.05), (10, 0.0, 0.0, 0.2)]
cube = adc_cube(p, targets, noise_stddev=1.0, seed=42)
rd = doppler_fft(range_fft(cube))
power = power_map(rd)
mask = ca_cfar_2d(power, pfa=1e-4)
cells = peaks(power, mask)
R, v = to_physical(p, cells)
theta = np.rad2deg(angle_fft(rd, cells))

fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(
    10 * np.log10(power),
    origin="lower",
    aspect="auto",
    cmap="viridis",
    extent=[0, p.max_range, -p.max_velocity, p.max_velocity],
)
ax.scatter(R, v, s=120, facecolors="none", edgecolors="w", linewidths=1.5)
for r, vv, th in zip(R, v, theta, strict=False):
    ax.annotate(
        f"{r:.1f} m, {vv:+.1f} m/s, {th:+.0f}°",
        (r, vv),
        xytext=(8, 8),
        textcoords="offset points",
        color="w",
        fontsize=8,
    )
ax.set_xlabel("Range [m]")
ax.set_ylabel("Radial velocity [m/s]")
ax.set_title("Range-Doppler map, CA-CFAR detections (Pfa = 1e-4)")
fig.colorbar(im, label="Power [dB]")
fig.tight_layout()
fig.savefig("docs/range_doppler.png", dpi=150)
