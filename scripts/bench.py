"""Wall-clock per call for NumPy vs CUDA.

Includes per-stage GPU times (cudaEvent) for the pipeline as median and p99.
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
pipe = radar_cuda.RadarPipeline(*cube.shape)


def np_chain(c):
    return ca_cfar_2d(power_map(doppler_fft(range_fft(c))))


def cu_chain(c):
    return radar_cuda.ca_cfar_2d(radar_cuda.power_map(radar_cuda.range_doppler(c)))


pipe.input[...] = cube  # producer writes into pinned memory, outside timing

np_full = bench(np_chain, cube)
rows = [
    (
        "range+doppler FFT",
        bench(lambda c: doppler_fft(range_fft(c)), cube),
        bench(radar_cuda.range_doppler, cube),
    ),
    ("power_map", bench(power_map, rd_np), bench(radar_cuda.power_map, rd_cu)),
    ("ca_cfar_2d", bench(ca_cfar_2d, pw_np), bench(radar_cuda.ca_cfar_2d, pw_cu)),
    ("full chain", np_full, bench(cu_chain, cube)),
    ("pipeline, pageable input", np_full, bench(pipe.process, cube)),
    ("pipeline, pinned input", np_full, bench(pipe.process)),
]
print(f"{'stage':<26}{'NumPy ms':>10}{'CUDA ms':>10}{'speedup':>9}")
for name, a, b in rows:
    print(f"{name:<26}{a:10.2f}{b:10.2f}{a / b:>9.1f}x")

N_WARMUP = 3
N_ITER = 500


def stage_times(call, n_warmup=N_WARMUP, n_iter=N_ITER):
    for _ in range(n_warmup):
        call()
    samples: dict[str, list[float]] = {}
    for _ in range(n_iter):
        for stage, ms in call()["t_ms"].items():
            samples.setdefault(stage, []).append(ms)
    return samples


pageable = stage_times(lambda: pipe.process(cube))
pinned = stage_times(lambda: pipe.process())

print(f"\nkernel breakdown, ms median over {N_ITER} calls")
print(f"{'stage':<8}{'pageable':>10}{'pinned':>10}")
for stage in pageable:
    print(f"{stage:<8}{np.median(pageable[stage]):10.3f}{np.median(pinned[stage]):10.3f}")
