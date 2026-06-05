# scripts/quantize.py
import os
import torch
import torch.ao.quantization as quant
from lumenbridge.reference import ReferenceLumenBridgeStem

def run_ptq_calibration():
    print("--- LumenBridge PTQ Calibration (Native PyTorch) ---")
    
    # 1. Hardware Architecture Fix for Apple Silicon (ARM64)
    # Explicitly set the global quantized engine to qnnpack
    torch.backends.quantized.engine = 'qnnpack'
    
    # 2. Instantiate the FP32 Reference Model
    model = ReferenceLumenBridgeStem(d_model=768, patch_size=16)
    model.eval()
    
    # Track original size
    torch.save(model.state_dict(), "fp32_model.pth")
    fp32_size = os.path.getsize("fp32_model.pth") / 1e6
    print(f"[Profiling] Original FP32 Model Size: {fp32_size:.2f} MB")

    # 3. Attach the QConfig for ARM architectures
    model.qconfig = quant.get_default_qconfig('qnnpack')
    
    # 4. Prepare: Injects Observers into the model
    # (Note: PyTorch will throw deprecation warnings here because they are migrating 
    # tools to `torchao` for PT2.0, but this classic API is perfectly stable for our local proof-of-concept)
    quant.prepare(model, inplace=True)
    print("[Calibration] Observers injected. Running calibration batch...")

    # 5. Calibration: Pass representative data
    with torch.no_grad():
        for _ in range(10):
            dummy_calibration_data = torch.rand(4, 3, 224, 224)
            model(dummy_calibration_data)

    # 6. Convert: Swaps FP32 layers for INT8 layers
    quant.convert(model, inplace=True)
    print("[Calibration] Model converted to INT8 successfully.")

    # Track new size
    torch.save(model.state_dict(), "int8_model.pth")
    int8_size = os.path.getsize("int8_model.pth") / 1e6
    print(f"[Profiling] Quantized INT8 Model Size: {int8_size:.2f} MB")
    
    # Avoid division by zero if the model is so small it registers as 0.00 MB
    if fp32_size > 0:
        print(f"[Profiling] Memory Reduction: {((fp32_size - int8_size) / fp32_size) * 100:.1f}%")

    # 7. Verify Output Shape
    test_tensor = torch.rand(1, 3, 224, 224)
    out = model(test_tensor)
    print(f"\n✓ SUCCESS: INT8 Pipeline verified. Output shape: {list(out.shape)}")

if __name__ == "__main__":
    run_ptq_calibration()