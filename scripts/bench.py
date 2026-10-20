"""Per-call wall-clock timings from Python, NumPy vs CUDA.

Includes host-to-device/device-to-host transfers and per-call allocation.
See README for kernel-only timings.
"""

import time

import numpy as np
import radar_cuda

from radar.params import RadarParams
from radar.processing import ca_cfar_2d, doppler_fft, power_map, range_fft
from radar.sim import adc_cube

p = RadarParams(fc=77e9, slope=30e12, Tc=40e-6, fs=10e6, n_samples=400, n_chirps=128, n_channels=8)
cube = adc_cube(p, [(20, 5.0, np.deg2rad(20), 1.0)], noise_stddev=1.0, seed=0).astype(np.complex64)


def bench(fn, *args, n=50):
    fn(*args)  # warmup (plan creation, JIT, caches)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args)
    return (time.perf_counter() - t0) / n * 1e3


rd_np = doppler_fft(range_fft(cube))
pw_np = power_map(rd_np)
rd_cu = radar_cuda.range_doppler(cube)
pw_cu = radar_cuda.power_map(rd_cu)


def np_chain(c):
    return ca_cfar_2d(power_map(doppler_fft(range_fft(c))))


def cu_chain(c):
    return radar_cuda.ca_cfar_2d(radar_cuda.power_map(radar_cuda.range_doppler(c)))


rows = [
    (
        "range+doppler FFT",
        bench(lambda c: doppler_fft(range_fft(c)), cube),
        bench(radar_cuda.range_doppler, cube),
    ),
    ("power_map", bench(power_map, rd_np), bench(radar_cuda.power_map, rd_cu)),
    ("ca_cfar_2d", bench(ca_cfar_2d, pw_np), bench(radar_cuda.ca_cfar_2d, pw_cu)),
    ("full chain", bench(np_chain, cube), bench(cu_chain, cube)),
]
print(f"{'stage':<20}{'NumPy ms':>10}{'CUDA ms':>10}{'speedup':>9}")
for name, a, b in rows:
    print(f"{name:<20}{a:10.2f}{b:10.2f}{a / b:>9.1f}x")
