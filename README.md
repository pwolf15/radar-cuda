# radar-cuda

FMCW radar signal-processing chain - synthetic ADC cube -> range/Doppler FFTs -> CA-CFAR detection -> angle estimation. Tested NumPy reference, with CUDA port in progress.

![Range-Doppler Map with CA-CFAR detections](docs/range_doppler.png)

Three simulated targets in complex Gaussian noise (amplitudes 1.0, 0.2, 0.05),
77 GHz, 1.2 GHz bandwidth, 128 chirps x 400 samples x 8 channels. Detections are reported at bin centers (0.125 m, 0.38 m/s, ~1.8° at boresight); no peak interpolation yet.

## Layout

| | |
|---|---|
| `src/radar/params.py` | `RadarParams`: chirp/frame parameters and derived resolutions |
| `src/radar/sim.py` | point-target beat-signal simulator, multi-target cube + noise |
| `src/radar/processing.py` | NumPy reference chain: `range_fft -> doppler_fft -> power_map -> ca_cfar_2d -> peaks -> angle_fft -> to_physical` |
| `tests/` | every stage tested against physics (bin positions, Pfa on pure noise, sign conventions) |
| `scripts/plot_rd.py` | produces the figure above |
| `cuda/radar_cuda.cu` | cuFFT range-Doppler (Hann windows, fftshift) as a pybind11 module |

## Running (NumPy Reference)

``` bash
uv sync
uv run pytest                    # run tests
uv run python scripts/plot_rd.py # results of script shown above
```

## Running (CUDA)

Needs NVIDIA GPU and CUDA >= 12; tests skip if module not built.

``` bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(uv run python -m pybind11 --cmakedir)
cmake --build ./build
uv run pytest tests/test_cuda.py
```

## Timings (RTX 3060, 8x128x400 frame, float32 CUDA vs float64 NumPy)

| stage              | NumPy ms | CUDA ms (wall, per call) | CUDA ms (kernel only) |
|--------------------|---------:|-------------------------:|----------------------:|
| range + Doppler FFT|    13.3  |   2.15                   | 0.18                  |
| power map          |     2.1  |   0.69                   | 0.012                 |
| CA-CFAR (naive)    |     1.9  |   0.14                   | 0.065                 |
| full chain         |    18.3  |   0.80 (pipeline)        | 0.26                  |

Wall-clock includes host-to-device (0.48 ms for the 3.3 MB cube) and device-to-host (0.07 ms).
Per-stage "wall" rows allocate and transfer per call; the pipeline row uploads once, runs three kernels, downloads `power` and `mask`. Kernel times are `cudaEvent` deltas. `scripts/bench.py` reproduces the table.

## Status

- [x] NumPy reference chain
- [x] cuFFT range-Doppler; matches NumPy to 2e-6 (float32)
- [ ] CA-CFAR stencil kernel, checked against NumPy reference
- [x] per-stage timings
- [ ] real frames (RaDICaL / ColoRadar)
