// src/cuda/conv_kernels.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

// Define our hardware tile optimizations
#define TILE_WIDTH 16
#define KERNEL_SIZE 3
#define HALO_SIZE (KERNEL_SIZE / 2) // For a 3x3 kernel, the halo is 1 pixel wide
#define SMEM_WIDTH (TILE_WIDTH + KERNEL_SIZE - 1) // 16 + 3 - 1 = 18

__global__ void depthwise_conv_shared_kernel(
    const float* input, 
    const float* weight, 
    float* output, 
    int batch_size, 
    int channels, 
    int height, 
    int width, 
    int stride
) {
    // 1. Allocate the Shared Memory Tile (18x18 to accommodate the 16x16 compute + 1px Halo)
    __shared__ float shared_tile[SMEM_WIDTH][SMEM_WIDTH];

    // Calculate thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global output coordinates mapped by this thread
    int col_out = blockIdx.x * TILE_WIDTH + tx;
    int row_out = blockIdx.y * TILE_WIDTH + ty;
    
    // Map to global input coordinates (accounting for stride)
    int col_in = col_out * stride;
    int row_in = row_out * stride;
    
    int channel = blockIdx.z % channels;
    int b = blockIdx.z / channels;

    // 2. Collaborative Memory Loading (Global to Shared)
    // Every thread loads its primary pixel into the center of the shared tile
    int smem_x = tx + HALO_SIZE;
    int smem_y = ty + HALO_SIZE;
    
    // Safe-load primary pixel
    if (row_in < height && col_in < width) {
        int in_idx = b * (channels * height * width) + channel * (height * width) + row_in * width + col_in;
        shared_tile[smem_y][smem_x] = input[in_idx];
    } else {
        shared_tile[smem_y][smem_x] = 0.0f;
    }

    // --- HALO BOUNDARY LOADING ---
    // The threads on the edges of the 16x16 block must do double-duty and load the halo pixels
    
    // Left Halo
    if (tx < HALO_SIZE) {
        int halo_col = col_in - HALO_SIZE;
        float val = 0.0f;
        if (halo_col >= 0 && row_in < height) {
            val = input[b * (channels * height * width) + channel * (height * width) + row_in * width + halo_col];
        }
        shared_tile[smem_y][tx] = val;
    }
    // Right Halo
    if (tx >= TILE_WIDTH - HALO_SIZE) {
        int halo_col = col_in + HALO_SIZE;
        float val = 0.0f;
        if (halo_col < width && row_in < height) {
            val = input[b * (channels * height * width) + channel * (height * width) + row_in * width + halo_col];
        }
        shared_tile[smem_y][smem_x + HALO_SIZE] = val;
    }
    // Top Halo
    if (ty < HALO_SIZE) {
        int halo_row = row_in - HALO_SIZE;
        float val = 0.0f;
        if (halo_row >= 0 && col_in < width) {
            val = input[b * (channels * height * width) + channel * (height * width) + halo_row * width + col_in];
        }
        shared_tile[ty][smem_x] = val;
    }
    // Bottom Halo
    if (ty >= TILE_WIDTH - HALO_SIZE) {
        int halo_row = row_in + HALO_SIZE;
        float val = 0.0f;
        if (halo_row < height && col_in < width) {
            val = input[b * (channels * height * width) + channel * (height * width) + halo_row * width + col_in];
        }
        shared_tile[smem_y + HALO_SIZE][smem_x] = val;
    }

    // --- CORNER HALO LOADING (The Fix) ---
    // Top-Left Corner
    if (tx < HALO_SIZE && ty < HALO_SIZE) {
        int halo_r = row_in - HALO_SIZE;
        int halo_c = col_in - HALO_SIZE;
        float val = 0.0f;
        if (halo_r >= 0 && halo_c >= 0) {
            val = input[b * (channels * height * width) + channel * (height * width) + halo_r * width + halo_c];
        }
        shared_tile[ty][tx] = val;
    }
    // Top-Right Corner
    if (tx >= TILE_WIDTH - HALO_SIZE && ty < HALO_SIZE) {
        int halo_r = row_in - HALO_SIZE;
        int halo_c = col_in + HALO_SIZE;
        float val = 0.0f;
        if (halo_r >= 0 && halo_c < width) {
            val = input[b * (channels * height * width) + channel * (height * width) + halo_r * width + halo_c];
        }
        shared_tile[ty][smem_x + HALO_SIZE] = val;
    }
    // Bottom-Left Corner
    if (tx < HALO_SIZE && ty >= TILE_WIDTH - HALO_SIZE) {
        int halo_r = row_in + HALO_SIZE;
        int halo_c = col_in - HALO_SIZE;
        float val = 0.0f;
        if (halo_r < height && halo_c >= 0) {
            val = input[b * (channels * height * width) + channel * (height * width) + halo_r * width + halo_c];
        }
        shared_tile[smem_y + HALO_SIZE][tx] = val;
    }
    // Bottom-Right Corner
    if (tx >= TILE_WIDTH - HALO_SIZE && ty >= TILE_WIDTH - HALO_SIZE) {
        int halo_r = row_in + HALO_SIZE;
        int halo_c = col_in + HALO_SIZE;
        float val = 0.0f;
        if (halo_r < height && halo_c < width) {
            val = input[b * (channels * height * width) + channel * (height * width) + halo_r * width + halo_c];
        }
        shared_tile[smem_y + HALO_SIZE][smem_x + HALO_SIZE] = val;
    }

    // 3. Hardware Barrier
    // Wait for all 256 threads in this block to finish fetching their pixels + halo pixels
    __syncthreads();

    // 4. Compute Phase (MAC Operations)
    if (row_out < height / stride && col_out < width / stride) {
        float val = 0.0f;
        
        // Loop over the 3x3 kernel using ONLY the ultra-fast shared memory
        for (int i = 0; i < KERNEL_SIZE; ++i) {
            for (int j = 0; j < KERNEL_SIZE; ++j) {
                // The offset maps the 3x3 kernel perfectly over our central pixel in shared memory
                float pixel = shared_tile[ty + i][tx + j];
                float weight_val = weight[channel * KERNEL_SIZE * KERNEL_SIZE + i * KERNEL_SIZE + j];
                val += pixel * weight_val;
            }
        }
        
        // 5. Write back to Global Memory
        int out_h = height / stride;
        int out_w = width / stride;
        int out_idx = b * (channels * out_h * out_w) + channel * (out_h * out_w) + row_out * out_w + col_out;
        output[out_idx] = val;
    }
}

// This is the Host Function that C++ will call.
torch::Tensor launch_depthwise_conv_cuda(
    const torch::Tensor& input, 
    int64_t d_model, 
    int64_t kernel_size, 
    int64_t stride
) {
    int batch = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int out_h = height / stride;
    int out_w = width / stride;

    // Allocate the output tensor directly on the GPU
    auto output = torch::empty({batch, channels, out_h, out_w}, input.options());
    
    // (In a real system, you'd pass weights in. For this architecture blueprint, we assume standard generation)
    auto weight = torch::randn({channels, 1, kernel_size, kernel_size}, input.options());

    // Calculate Grid and Block dimensions
    // A block is our 16x16 tile of threads
    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    
    // The grid is how many 16x16 blocks we need to cover the whole image, multiplied by channels and batch
    dim3 blocks_per_grid(
        (out_w + TILE_WIDTH - 1) / TILE_WIDTH,
        (out_h + TILE_WIDTH - 1) / TILE_WIDTH,
        batch * channels
    );

    // Launch the CUDA Kernel
    depthwise_conv_shared_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, height, width, stride
    );

    // Flatten and transpose for LLM alignment, just like we did on the CPU
    output = output.flatten(2).transpose(1, 2);
    return output.contiguous();
}