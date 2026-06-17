# setup.py
import os
import sys
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

print("--- LumenBridge Build System Initializing ---")

# 1. Hardware Detection Router
# We check if PyTorch detects a local NVIDIA GPU or if the user forces CUDA compilation via CI/CD
build_cuda = torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1"

# 2. Base Configuration (Shared across all platforms)
sources = [
    "src/bindings.cpp",
    "src/encoder.cpp"
]

# PyTorch extensions require a dictionary format when supporting multiple compilers (C++ vs NVCC)
extra_compile_args = {
    "cxx": ["-O3", "-Wall", "-std=c++17"]
}

# 3. macOS Specific Overrides (Apple Silicon / Intel Mac)
if sys.platform == "darwin":
    extra_compile_args["cxx"] += ["-stdlib=libc++"]
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"

# 4. Dynamic Extension Injection
if build_cuda:
    print("[Router] NVIDIA Hardware Detected: Compiling deep CUDA optimizations.")
    # Inject the future CUDA file into the compilation sources
    sources.append("src/cuda/conv_kernels.cu")
    
    # Tell the C++ compiler that CUDA is active
    extra_compile_args["cxx"].append("-DUSE_CUDA")

    # Add strict, high-performance flags specifically for the NVIDIA compiler (nvcc)
    extra_compile_args["nvcc"] = [
        "-O3",
        "--use_fast_math", # Crucial for aggressive FLOP optimization
        "-lineinfo",       # Allows profiling tools (Nsight) to read the binary
        "-std=c++17"
    ]
    # Swap the extension engine to handle .cu files
    ExtensionClass = CUDAExtension
else:
    print("[Router] No CUDA detected (macOS/CPU environment). Falling back to standard C++ build.")
    # Keep the standard engine we used in Phase 2
    ExtensionClass = CppExtension

# 5. Execute Compilation
setup(
    name="lumenbridge_core",
    ext_modules=[
        ExtensionClass(
            name="lumenbridge_core",
            sources=sources,
            include_dirs=["include"],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)