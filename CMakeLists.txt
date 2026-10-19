cmake_minimum_required(VERSION 3.24)

project(radar_cuda LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit 13.0 REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(radar_cuda cuda/radar_cuda.cu)
target_link_libraries(radar_cuda PRIVATE CUDA::cudart CUDA::cufft)
