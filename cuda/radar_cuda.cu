#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

static std::string device_name() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    return p.name;
}

PYBIND11_MODULE(radar_cuda, m) {
    m.def("device_name", &device_name);
}