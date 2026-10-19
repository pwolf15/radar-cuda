#include <cmath>
#include <cuda_runtime.h>
#include <numbers>
#include <stdexcept>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <cufft.h>

namespace py = pybind11;

static std::string device_name() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    return p.name;
}

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e)); } while (0)
#define CUFFT_CHECK(x) do { cufftResult r = (x); if (r != CUFFT_SUCCESS) \
    throw std::runtime_error("cuFFT error " + std::to_string(r)); } while (0)

template <class T> struct DevBuf {
    T* p = nullptr; size_t n = 0;
    explicit DevBuf(size_t n_) : n(n_) { CUDA_CHECK(cudaMalloc(&p, n*sizeof(T))); }
    ~DevBuf() { cudaFree(p); }
    DevBuf(const DevBuf&) = delete; DevBuf& operator=(const DevBuf&) = delete;
};

__global__ void apply_window_fast(cuFloatComplex* x, const float* w, int N, size_t total) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < total) { float s = w[i % N]; x[i].x *= s; x[i].y *= s; }
}

__global__ void apply_window_slow(cuFloatComplex* x, const float* w, int M, int N, size_t total) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < total) { float s = w[(i / N) % M]; x[i].x *= s; x[i].y *= s;}
}

__global__ void fftshift(cuFloatComplex* out, const cuFloatComplex* in, int M, int N, size_t total) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < total) {
        int n = i % N;
        int m = (i / N) % M;
        int c = i / (M * N);
        int src_m = (m + M/2) % M;
        out[i] = in[c * M * N + src_m * N + n];
    }
}

struct RangeDopplerPlan {
    int C, M, N;
    cufftHandle plan_range, plan_doppler;
    float *d_w_fast, *d_w_slow;
    RangeDopplerPlan(int C, int M, int N): C(C), M(M), N(N) {
        // create window data and upload

        // symmetric Hann, matches np.hanning
        // periodic variant divides by N, not N-1
        std::vector<float> w_fast(N);
        for (int i = 0; i < N; ++i) {
            w_fast[i] = 0.5 * (1 - cos(2*std::numbers::pi*i/(N-1)));
        }
        std::vector<float> w_slow(M);
        for (int i = 0; i < M; ++i) {
            w_slow[i] = 0.5 * (1 - cos(2*std::numbers::pi*i/(M-1)));
        }
        CUDA_CHECK(cudaMalloc(&d_w_fast, sizeof(float) * N));
        CUDA_CHECK(cudaMalloc(&d_w_slow, sizeof(float) * M));
        CUDA_CHECK(cudaMemcpy(d_w_fast, w_fast.data(), sizeof(float)*N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w_slow, w_slow.data(), sizeof(float)*M, cudaMemcpyHostToDevice));

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
        CUFFT_CHECK(cufftPlanMany(&plan_range, rank, &n, &inembed, istride, idist, &onembed, ostride, odist, CUFFT_C2C, batch));

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
        CUFFT_CHECK(cufftPlanMany(&plan_doppler, rank, &n, &inembed, istride, idist, &onembed, ostride, odist, CUFFT_C2C, batch));
    }
    ~RangeDopplerPlan() {
        cufftDestroy(plan_range);
        cufftDestroy(plan_doppler);
        cudaFree(d_w_fast);
        cudaFree(d_w_slow);
    }
    void run(const cuFloatComplex* d_in, cuFloatComplex* d_out, cuFloatComplex* d_tmp) const {

        size_t total = C*M*N;
        CUDA_CHECK(cudaMemcpy(d_out, d_in, sizeof(cuFloatComplex)*total, cudaMemcpyDeviceToDevice));

        // apply window-fast
        apply_window_fast<<<(total + 255) / 256, 256>>>(d_out, d_w_fast, N, total);
        CUDA_CHECK(cudaGetLastError());

        // run range FFT
        CUFFT_CHECK(cufftExecC2C(plan_range, d_out, d_out, CUFFT_FORWARD));

        // apply window-slow
        apply_window_slow<<<(total + 255) / 256, 256>>>(d_out, d_w_slow, M, N, total);
        CUDA_CHECK(cudaGetLastError());

        // run doppler fft
        for (int c = 0; c < C; ++c) {
            CUFFT_CHECK(cufftExecC2C(plan_doppler, d_out + c * M * N, d_out + c * M * N, CUFFT_FORWARD));
        }
        // shift elements
        fftshift<<<(total + 255) / 256, 256>>>(d_tmp, d_out, M, N, total);
        CUDA_CHECK(cudaGetLastError());

        //@TODO: avoid copy
        CUDA_CHECK(cudaMemcpy(d_out, d_tmp, sizeof(cuFloatComplex)*total, cudaMemcpyDeviceToDevice));
    }
};

py::array_t<std::complex<float>> range_doppler(py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> cube) {

    // read data from NumPy buffer
    auto buf = cube.request();
    int C = buf.shape[0], M = buf.shape[1], N = buf.shape[2];
    size_t n = size_t(C) * M * N;
    DevBuf<cuFloatComplex> d_in(n), d_out(n), d_tmp(n);
    CUDA_CHECK(cudaMemcpy(d_in.p, buf.ptr, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

    // run range-doppler FFTs
    RangeDopplerPlan plan(C, M, N);
    plan.run(d_in.p, d_out.p, d_tmp.p);

    // write data to NumPy buffers
    auto out = py::array_t<std::complex<float>>({C, M, N});
    CUDA_CHECK(cudaMemcpy(out.mutable_data(), d_out.p, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    return out;
}


PYBIND11_MODULE(radar_cuda, m) {
    m.def("device_name", &device_name);
    m.def("range_doppler", &range_doppler);
}
