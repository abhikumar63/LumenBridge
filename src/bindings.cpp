// src/bindings.cpp
#include <torch/extension.h>
#include "encoder.hpp"

// Define the extension module name exactly as declared in setup.py
PYBIND11_MODULE(lumenbridge_core, m) {
    m.doc() = "LumenBridge Core C++ High-Performance Inference Extension";
    
    // Bind the project_patches function under the python module namespace
    m.def(
        "project_patches", 
        &lumenbridge::core::project_patches, 
        "Processes raw image tensors into an LLM-compatible contiguous token sequence layout",
        py::arg("input"),
        py::arg("d_model"),
        py::arg("kernel_size"),
        py::arg("stride")
    );
}