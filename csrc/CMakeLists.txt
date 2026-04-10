cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(visual_tokenizer_ops LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)

# Set strictly optimized flags for production
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -shared -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")

add_library(visual_ops SHARED
    bindings.cpp
    conv_kernels.cu
)

target_link_libraries(visual_ops "${TORCH_LIBRARIES}")
set_property(TARGET visual_ops PROPERTY CXX_STANDARD 17)