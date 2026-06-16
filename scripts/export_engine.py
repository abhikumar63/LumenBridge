import tensorrt as trt
import sys

# Define logging
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    print(f"--- LumenBridge TensorRT Compiler ---")
    print(f"Target: {onnx_file_path}")
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Configure the builder
    config = builder.create_builder_config()
    
    # Allow TensorRT to use up to 4GB of workspace memory for compilation
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30)) 
    
    # We are forcing FP16 precision. Since we did INT8 PTQ earlier, 
    # we can later enable INT8 flag here. For this baseline run, FP16 is safer.
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Enabled FP16 Optimization.")

    # Parse ONNX
    print("Parsing ONNX file...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    print("Building TensorRT Engine. This may take a few minutes...")
    engine_bytes = builder.build_serialized_network(network, config)
    
    if engine_bytes is None:
        print("ERROR: Engine compilation failed.")
        return None
        
    print(f"Engine built successfully! Saving to {engine_file_path}")
    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)
        
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_engine.py <input.onnx> <output.engine>")
        sys.exit(1)
        
    build_engine(sys.argv[1], sys.argv[2])