import torch
import time
from lumenbridge.ops import project_visual_patches

def test_local_bridge():
    print("--- LumenBridge Local Sanity Check ---")
    
    # 1. Simulate a standard Vision Transformer input: [Batch, Channels, Height, Width]
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    d_model = 768
    kernel_size = 16
    stride = 16
    
    print(f"Generating dummy image tensor: [{batch_size}, {channels}, {height}, {width}]")
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    # 2. Test 1: The Standard Forward Pass
    try:
        print("\n[Test 1] Passing contiguous tensor to C++ backend...")
        start_time = time.time()
        
        # This will route to the Mac C++ fallback path in encoder.cpp
        output = project_visual_patches(dummy_input, d_model, kernel_size, stride)
        
        elapsed = (time.time() - start_time) * 1000
        print(f"Success! Output Shape: {output.shape} | Latency: {elapsed:.2f} ms")
        
        # Expected shape for 224x224 with 16x16 patches is [Batch, 194, 768] -> 14x14 = 196 patches
        assert output.shape == (batch_size, 196, d_model), "Mathematical Output Shape Mismatch!"
        print("Shape alignment verified.")
        
    except Exception as e:
        print(f"\n[Test 1 FAILED]: {e}")

    # 3. Test 2: The Memory Guardrail Trigger
    try:
        print("\n[Test 2] Forcing non-contiguous memory layout...")
        # Transposing a tensor changes its stride, making it non-contiguous in memory
        bad_tensor = dummy_input.transpose(2, 3) 
        
        # ops.py should catch this and fix it before it hits C++
        output_safe = project_visual_patches(bad_tensor, d_model, kernel_size, stride)
        print("Success! Python Gateway successfully caught and fixed the memory layout.")
        
    except Exception as e:
        print(f"\n[Test 2 FAILED]: The memory guardrail did not protect the C++ engine. {e}")

if __name__ == "__main__":
    test_local_bridge()