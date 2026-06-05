// include/encoder.h
#pragma once

#include <torch/extension.h>
#include <vector>

namespace lumenbridge {
namespace core {

/**
 * Validates tensor properties and returns its geometric dimensions.
 * Objectively acts as a safety gate before passing memory to tokenization algorithms.
 * 
 * @param input A 4D Tensor representing raw visual data [Batch, Channels, Height, Width]
 * @param d_model Target embedding dimension size (e.g., 768)
 * @param kernel_size Size of the spatial window (e.g., 4)
 * @param stride Distance the window steps across spatial planes (e.g., 4)
 */
torch::Tensor project_patches(
    const torch::Tensor& input,
    int64_t d_model,
    int64_t kernel_size,
    int64_t stride
);

} // namespace core
} // namespace lumenbridge