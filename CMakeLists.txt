cmake_minimum_required(VERSION 3.24)

project(radar_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit 13.0 REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(radar_cuda cuda/radar_cuda.cu)
target_link_libraries(radar_cuda PRIVATE CUDA::cudart CUDA::cufft)
