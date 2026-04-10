#include <torch/extension.h>

// Forward declaration of the CUDA kernel launcher
torch::Tensor custom_dw_conv_cuda(torch::Tensor input, torch::Tensor weight);

// C++ frontend to check tensor sanity before launching the kernel
torch::Tensor custom_dw_conv(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    return custom_dw_conv_cuda(input, weight);
}

// Bind to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depthwise_conv", &custom_dw_conv, "Custom CUDA Depthwise Convolution");
}