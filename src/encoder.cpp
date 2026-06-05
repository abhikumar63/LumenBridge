#include "encoder.hpp"

// Forward declaration telling the compiler this function exists in the .cu file
torch::Tensor launch_depthwise_conv_cuda(const torch::Tensor& input, int64_t d_model, int64_t kernel_size, int64_t stride);

namespace lumenbridge {
namespace core {

torch::Tensor project_patches(
    const torch::Tensor& input, 
    int64_t d_model, 
    int64_t kernel_size, 
    int64_t stride
) {
    TORCH_CHECK(input.is_contiguous(), "LumenBridge Error: Input tensor must be contiguous.");
    TORCH_CHECK(input.dim() == 4, "LumenBridge Error: Input must have exactly 4 dimensions [B, C, H, W].");

    // --- THE HARDWARE ROUTER ---
    if (input.device().is_cuda()) {
        // PRODUCTION PATH: The tensor is on an NVIDIA GPU. Route to custom kernels.
        return launch_depthwise_conv_cuda(input, d_model, kernel_size, stride);
    } 
    
    // MAC LOCAL DEV PATH: Fall back to ATen C++ operations
    int64_t channels = input.size(1);
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    
    auto dw_weights = torch::randn({channels, 1, kernel_size, kernel_size}, options);
    auto pw_weights = torch::randn({d_model, channels, 1, 1}, options);
    at::Tensor null_bias;

    auto x = at::conv2d(input, dw_weights, null_bias, std::vector<int64_t>{stride, stride}, std::vector<int64_t>{0, 0}, std::vector<int64_t>{1, 1}, channels);
    x = at::conv2d(x, pw_weights, null_bias, std::vector<int64_t>{1, 1}, std::vector<int64_t>{0, 0}, std::vector<int64_t>{1, 1}, 1);

    return x.flatten(2).transpose(1, 2).contiguous();
}

} // namespace core
} // namespace lumenbridge