import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import time
import sys

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def benchmark_latency(engine, batch_size=1, num_runs=100):
    context = engine.create_execution_context()
    
    # Define input/output shapes (Assuming 224x224 image, 3 channels)
    # Output size depends on your specific patch size configuration.
    # We will assume [Batch, 196, 768] for standard ViT tokens.
    input_shape = (batch_size, 3, 224, 224) 
    
    # Allocate host and device buffers
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(batch_size * 196 * 768, dtype=np.float32)
    
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()
    
    print(f"--- Benchmark Started (Batch Size: {batch_size}) ---")
    
    # Warmup runs (to allow GPU to clock up)
    for _ in range(10):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
    
    stream.synchronize()
    
    # Timing Runs
    start_time = time.time()
    for _ in range(num_runs):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
    
    stream.synchronize()
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / num_runs * 1000 # in milliseconds
    print(f"Average Inference Latency: {avg_latency:.2f} ms")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <engine_file>")
        sys.exit(1)
        
    engine = load_engine(sys.argv[1])
    benchmark_latency(engine)