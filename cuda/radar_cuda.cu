#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cufft.h>
#include <numbers>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

static std::string device_name() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  return p.name;
}

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess)                                                      \
      throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e)); \
  } while (0)
#define CUFFT_CHECK(x)                                                         \
  do {                                                                         \
    cufftResult r = (x);                                                       \
    if (r != CUFFT_SUCCESS)                                                    \
      throw std::runtime_error("cuFFT error " + std::to_string(r));            \
  } while (0)

template <class T> struct DevBuf {
  T *p = nullptr;
  size_t n = 0;
  explicit DevBuf(size_t n_) : n(n_) {
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  }
  ~DevBuf() { cudaFree(p); }
  DevBuf(const DevBuf &) = delete;
  DevBuf &operator=(const DevBuf &) = delete;
};

__global__ void apply_windows(const float *w_s, const float *w_f,
                              const cuFloatComplex *x, cuFloatComplex *y, int M,
                              int N, size_t total) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < total) {
    float s_s = w_s[(i / N) % M], s_f = w_f[i % N];
    y[i].x = x[i].x * (s_s * s_f);
    y[i].y = x[i].y * (s_s * s_f);
  }
}

__global__ void power_map_kernel(const cuFloatComplex *rd, float *power, int C,
                                 int M, int N) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)M * N)
    return;
  float acc = 0.f;
  for (int c = 0; c < C; ++c) {
    cuFloatComplex v = rd[(size_t)c * M * N + i];
    acc += v.x * v.x + v.y * v.y;
  }
  power[i] = acc;
}

__device__ inline int wrap(int i, int n) { return (i % n + n) % n; }
__device__ inline int clampi(int i, int n) {
  return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// one thread per cell
// wrap on Doppler, clamp on range
// For each cell under test (CUT), noise level is mean of training
//   cells: a (2(g_d+t_d)+1) x (2(g_r+t_r)+1) box minus the (2g_d+1) x (2g_r+1)
//   guard box centered on the CUT. guard/train are (doppler,range) half-widths
// alpha: exact threshold for exponentially distributed (square-law,
// single-channel) noise
__global__ void ca_cfar_kernel(const float *power, uint8_t *mask, int M, int N,
                               int g_d, int g_r, int t_d, int t_r, int n_train,
                               float alpha) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)M * N)
    return;
  int m = i / N;
  int n = i % N;
  float sum_outer = 0.0f, sum_inner = 0.0f;
  for (int dm = -(g_d + t_d); dm <= g_d + t_d; ++dm) {
    for (int dn = -(g_r + t_r); dn <= g_r + t_r; ++dn) {
      float v = power[wrap(m + dm, M) * N + clampi(n + dn, N)];
      sum_outer += v;
      if (abs(dm) <= g_d && abs(dn) <= g_r) {
        sum_inner += v;
      }
    }
  }
  float sum_train = sum_outer - sum_inner;
  float noise = sum_train / n_train;
  mask[i] = power[i] > alpha * noise ? 1 : 0;
}

static float compute_alpha(int n_train, float pfa) {
  return n_train * (pow(pfa, -1.0f / n_train) - 1);
}

struct RangeDopplerPlan {
  int C, M, N;
  cufftHandle plan_range, plan_doppler;
  float *d_w_fast, *d_w_slow_shift;
  RangeDopplerPlan(int C, int M, int N) : C(C), M(M), N(N) {

    // M must be even (FFT shift logic)
    if (M % 2)
      throw std::invalid_argument(
          "M must be even: fftshift is folded into window as (-1)^m");

    // create window data and upload

    // symmetric Hann, matches np.hanning
    // periodic variant divides by N, not N-1
    std::vector<float> w_fast(N);
    for (int i = 0; i < N; ++i) {
      w_fast[i] = 0.5 * (1 - cos(2 * std::numbers::pi * i / (N - 1)));
    }
    std::vector<float> w_slow_shift(M);
    for (int i = 0; i < M; ++i) {
      w_slow_shift[i] = 0.5 * (1 - cos(2 * std::numbers::pi * i / (M - 1))) *
                        (i % 2 ? -1.0f : 1.0f);
    }
    CUDA_CHECK(cudaMalloc(&d_w_fast, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_w_slow_shift, sizeof(float) * M));
    CUDA_CHECK(cudaMemcpy(d_w_fast, w_fast.data(), sizeof(float) * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w_slow_shift, w_slow_shift.data(),
                          sizeof(float) * M, cudaMemcpyHostToDevice));

    // create 2 cuFFT plans
    // range: CxM FFTs of size N
    //  plan: plan_range
    //  rank: 1 (1d transform)
    //  n: 1D array {N}
    //  inembed: N
    //  istride: 1
    //  idist: N
    //  onembed: N
    //  ostride: 1
    //  odist: N
    //  type: C2C
    //  batch: CxM
    int rank = 1;
    int n = N;
    int inembed = N;
    int istride = 1;
    int idist = N;
    int onembed = N;
    int ostride = 1;
    int odist = N;
    int batch = C * M;
    CUFFT_CHECK(cufftPlanMany(&plan_range, rank, &n, &inembed, istride, idist,
                              &onembed, ostride, odist, CUFFT_C2C, batch));

    // doppler: CxN FFTs of size M
    //  plan: plan_doppler
    //  rank: 1 (1d transform)
    //  n: 1D array {M}
    //  inembed: M
    //  istride: N
    //  idist: 1
    //  onembed: M
    //  ostride: N
    //  odist: 1
    //  type: C2C
    //  batch: N
    rank = 1;
    n = M;
    inembed = M;
    istride = N;
    idist = 1;
    onembed = M;
    ostride = N;
    odist = 1;
    batch = N;
    CUFFT_CHECK(cufftPlanMany(&plan_doppler, rank, &n, &inembed, istride, idist,
                              &onembed, ostride, odist, CUFFT_C2C, batch));
  }
  ~RangeDopplerPlan() {
    cufftDestroy(plan_range);
    cufftDestroy(plan_doppler);
    cudaFree(d_w_fast);
    cudaFree(d_w_slow_shift);
  }
  RangeDopplerPlan(const RangeDopplerPlan &) = delete;
  // can run in-place with d_in == d_out
  void run(const cuFloatComplex *d_in, cuFloatComplex *d_out) const {

    size_t total = static_cast<size_t>(C) * M * N;

    // apply windows
    apply_windows<<<(total + 255) / 256, 256>>>(d_w_slow_shift, d_w_fast, d_in,
                                                d_out, M, N, total);
    CUDA_CHECK(cudaGetLastError());

    // run range FFT
    CUFFT_CHECK(cufftExecC2C(plan_range, d_out, d_out, CUFFT_FORWARD));

    // run doppler fft
    for (int c = 0; c < C; ++c) {
      CUFFT_CHECK(cufftExecC2C(plan_doppler, d_out + c * M * N,
                               d_out + c * M * N, CUFFT_FORWARD));
    }
  }
};

struct RadarPipeline {
  int C, M, N;
  int g_d, g_r, t_d, t_r;
  int n_train;
  float alpha;
  RangeDopplerPlan fft;
  DevBuf<cuFloatComplex> d_cube;
  DevBuf<float> d_power;
  DevBuf<uint8_t> d_mask;
  cudaEvent_t ev[6]; // timing

  RadarPipeline(int C, int M, int N, std::pair<int, int> guard,
                std::pair<int, int> train, float pfa)
      : C(C), M(M), N(N), g_d(guard.first), g_r(guard.second), t_d(train.first),
        t_r(train.second), fft(C, M, N), d_cube(C * M * N), d_power(N * M),
        d_mask(M * N) {

    // n_train, alpha
    int n_inner = (2 * g_d + 1) * (2 * g_r + 1);
    int n_outer = (2 * (g_d + t_d) + 1) * (2 * (g_r + t_r) + 1);
    n_train = n_outer - n_inner;
    alpha = compute_alpha(n_train, pfa);

    for (int i = 0; i < 6; ++i) {
      CUDA_CHECK(cudaEventCreate(&ev[i]));
    }
  }

  ~RadarPipeline() {
    for (int i = 0; i < 6; ++i) {
      cudaEventDestroy(ev[i]);
    }
  }

  RadarPipeline(const RadarPipeline &) = delete;

  py::dict process(py::array_t<std::complex<float>,
                               py::array::c_style | py::array::forcecast>
                       cube) {
    float ms_h2d, ms_fft, ms_power, ms_cfar, ms_d2h;

    CUDA_CHECK(cudaEventRecord(ev[0]));

    auto buf = cube.request();
    size_t n = size_t(C) * M * N;
    CUDA_CHECK(cudaMemcpy(d_cube.p, buf.ptr, n * sizeof(cuFloatComplex),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(ev[1]));

    // run range-doppler FFTs
    fft.run(d_cube.p, d_cube.p);

    CUDA_CHECK(cudaEventRecord(ev[2]));

    // power map
    const int total = M * N;
    power_map_kernel<<<(total + 255) / 256, 256>>>(d_cube.p, d_power.p, C, M,
                                                   N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev[3]));

    // cfar kernel creation
    // bboxes for train and guard
    ca_cfar_kernel<<<(total + 255) / 256, 256>>>(d_power.p, d_mask.p, M, N, g_d,
                                                 g_r, t_d, t_r, n_train, alpha);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev[4]));

    // write data to NumPy buffers
    auto power = py::array_t<float>({M, N});
    CUDA_CHECK(cudaMemcpy(power.mutable_data(), d_power.p,
                          total * sizeof(float), cudaMemcpyDeviceToHost));
    auto mask = py::array_t<bool>({M, N});
    CUDA_CHECK(cudaMemcpy(mask.mutable_data(), d_mask.p,
                          total * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(ev[5]));
    CUDA_CHECK(cudaEventSynchronize(ev[5]));

    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, ev[0], ev[1]));
    CUDA_CHECK(cudaEventElapsedTime(&ms_fft, ev[1], ev[2]));
    CUDA_CHECK(cudaEventElapsedTime(&ms_power, ev[2], ev[3]));
    CUDA_CHECK(cudaEventElapsedTime(&ms_cfar, ev[3], ev[4]));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, ev[4], ev[5]));

    py::dict out;
    out["power"] = power;
    out["mask"] = mask;

    py::dict t;
    t["h2d"] = ms_h2d;
    t["fft"] = ms_fft;
    t["power"] = ms_power;
    t["cfar"] = ms_cfar;
    t["d2h"] = ms_d2h;
    out["t_ms"] = t;
    return out;
  }
};

py::array_t<std::complex<float>> range_doppler(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>
        cube) {

  // read data from NumPy buffer
  auto buf = cube.request();
  int C = buf.shape[0], M = buf.shape[1], N = buf.shape[2];
  size_t n = size_t(C) * M * N;
  DevBuf<cuFloatComplex> d_in(n), d_out(n);
  CUDA_CHECK(cudaMemcpy(d_in.p, buf.ptr, n * sizeof(cuFloatComplex),
                        cudaMemcpyHostToDevice));

  // run range-doppler FFTs
  RangeDopplerPlan plan(C, M, N);
  plan.run(d_in.p, d_out.p);

  // write data to NumPy buffers
  auto out = py::array_t<std::complex<float>>({C, M, N});
  CUDA_CHECK(cudaMemcpy(out.mutable_data(), d_out.p, n * sizeof(cuFloatComplex),
                        cudaMemcpyDeviceToHost));
  return out;
}

py::array_t<float> power_map(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>
        rd) {

  // read data from NumPy buffer
  auto buf = rd.request();
  int C = buf.shape[0], M = buf.shape[1], N = buf.shape[2];
  size_t n = size_t(C) * M * N;
  DevBuf<cuFloatComplex> d_in(n);
  DevBuf<float> d_out(M * N);
  CUDA_CHECK(cudaMemcpy(d_in.p, buf.ptr, n * sizeof(cuFloatComplex),
                        cudaMemcpyHostToDevice));

  // run power_map kernel
  int total = M * N;
  power_map_kernel<<<(total + 255) / 256, 256>>>(d_in.p, d_out.p, C, M, N);
  CUDA_CHECK(cudaGetLastError());

  // write data to NumPy buffers
  auto out = py::array_t<float>({M, N});
  CUDA_CHECK(cudaMemcpy(out.mutable_data(), d_out.p, M * N * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return out;
}

py::array_t<bool>
ca_cfar_2d(py::array_t<float, py::array::c_style | py::array::forcecast> power,
           std::pair<int, int> guard = {2, 2},
           std::pair<int, int> train = {4, 8}, float pfa = 1e-4) {

  static_assert(sizeof(bool) == 1);

  // read data from NumPy buffer
  auto buf = power.request();
  int M = buf.shape[0], N = buf.shape[1];
  DevBuf<float> d_in(M * N);
  DevBuf<uint8_t> d_out(M * N);
  CUDA_CHECK(cudaMemcpy(d_in.p, buf.ptr, sizeof(float) * M * N,
                        cudaMemcpyHostToDevice));

  // run ca_cfar kernel
  int total = M * N;

  // bboxes for train and guard
  auto [g_d, g_r] = guard;
  auto [t_d, t_r] = train;
  int n_inner = (2 * g_d + 1) * (2 * g_r + 1);
  int n_outer = (2 * (g_d + t_d) + 1) * (2 * (g_r + t_r) + 1);
  int n_train = n_outer - n_inner;

  float alpha = compute_alpha(n_train, pfa);
  ca_cfar_kernel<<<(total + 255) / 256, 256>>>(d_in.p, d_out.p, M, N, g_d, g_r,
                                               t_d, t_r, n_train, alpha);
  CUDA_CHECK(cudaGetLastError());

  // write data to NumPy buffers
  auto out = py::array_t<bool>({M, N});
  CUDA_CHECK(cudaMemcpy(out.mutable_data(), d_out.p, M * N * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost));
  return out;
}

PYBIND11_MODULE(radar_cuda, m) {
  m.def("device_name", &device_name);
  m.def("range_doppler", &range_doppler);
  m.def("power_map", &power_map);
  m.def("ca_cfar_2d", &ca_cfar_2d, py::arg("power"),
        py::arg("guard") = std::make_pair(2, 2),
        py::arg("train") = std::make_pair(4, 8), py::arg("pfa") = 1e-4f);
  py::class_<RadarPipeline>(m, "RadarPipeline")
      .def(py::init<int, int, int, std::pair<int, int>, std::pair<int, int>,
                    float>(),
           py::arg("C"), py::arg("M"), py::arg("N"),
           py::arg("guard") = std::make_pair(2, 2),
           py::arg("train") = std::make_pair(4, 8), py::arg("pfa") = 1e-4)
      .def("process", &RadarPipeline::process, py::arg("cube"));
}
