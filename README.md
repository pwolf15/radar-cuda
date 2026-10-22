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

## Timings

RTX 3060 on an HP Z230 (PCIe Gen3 x16), one 8 x 128 x 400 frame, float32 CUDA vs. float 64 NumPy. Wall-clock is per Python call; GPU stage times are `cudaEvent` medains over 500 calls. `script/bench.py` reproduces both.

| stage                    | NumPy ms | CUDA ms (wall) |
|--------------------------|---------:|---------------:|
| range + Doppler FFT      |    13.69 |           1.94 |
| power map                |     2.11 |           0.70 |
| CA-CFAR (naive)          |     1.90 |           0.14 |
| full chain, per-stage calls |  18.71 |           2.98 |
| pipeline, pageable input |    18.71 |           0.71 |
| pipeline, pinned input   |    18.71 |           0.60 |

The per-stage rows allocate and transfer on every call; the pipeline uploads once, runs all kernels on device, and downloads only `power` and `mask`.

### Optimization steps

| step                                         | fft ms | h2d ms | pipeline ms |
|----------------------------------------------|-------:|-------:|------------:|
| baseline                                     |  0.184 |  0.48  |        0.80 |
| fold fftshift into the window, merge kernels |  0.112 |  0.47  |        0.73 |
| run in place, drop the device copy           |  0.091 |  0.47  |        0.72 |
| pinned input buffer                          |  0.089 |  0.36  |        0.60 |

* **fftshift.**. Multiplying the slow-time window by (1)^m shifts the Doppler spectrum by M/2, which is fftshift for even M. Following this, I merged both windows into one kernel, and running in place removed four passes over the 3.3 MB cube (~22 us each, ~270 Gb/s, which nears card's memory bandwidth).
* **Pinned input.** `RadarPipeline.input` is a NumPy view onto a `cudaMallocHost` buffer; a producer writes into it and `process()` uploads without staging. Host-to-device reaches 9.2 GB/s on Gen3 x16.

At current frame size, transfer costs four times FFT work: pipeline is bound by PCIe.

## Status

- [x] NumPy reference chain
- [x] cuFFT range-Doppler; matches NumPy to 2e-6 (float32)
- [x] CA-CFAR stencil kernel, checked against NumPy reference
- [x] per-stage timings
- [ ] real frames (RaDICaL / ColoRadar)
