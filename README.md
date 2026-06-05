# LumenBridge 🌉

**LumenBridge** is a high-performance, hybrid Python/C++ visual tokenizer engineered for high-frequency and low-latency AI inference (e.g., real-time document tokenization, financial chart parsing). 

Standard PyTorch Python layers suffer from Global Interpreter Lock (GIL) overhead and fail silently on uncoalesced memory reads. LumenBridge solves this by bypassing standard deep learning wrappers, executing **Strided Depthwise Separable Convolutions** directly at the bare-metal C++/CUDA level using LibTorch and ATen.

It efficiently maps raw 2D image footprints `[Batch, 3, 224, 224]` into flattened, 1D sequential tokens `[Batch, 196, 768]`, making visual data instantly readable by standard downstream LLM Attention blocks.

---

## 🚀 Architectural Philosophy

1. **Eradicating Python Overhead:** Heavy tensor manipulations (spatial flattening, channel projections) are shifted to native C++, running at bare-metal speeds.
2. **Strict Hardware Guardrails:** Embedded `TORCH_CHECK` macros validate memory contiguity at the boundary layer, preventing silent data corruption and ensuring maximum cache-line utilization.
3. **Hardware-Awareness:** The build system dynamically routes compilation based on the host machine—using Clang/C++17 for local macOS development, and seamlessly pivoting to NVIDIA's `nvcc` compiler for Linux GPU servers.

---

## 🏗️ System Tiers

The codebase is structured across three distinct operational layers:

1. **Python Object Layer (`lumenbridge/model.py`):** A user-friendly PyTorch `nn.Module` (`LumenBridgeStem`) that handles LayerNorm and configuration routing. Drops cleanly into any existing transformer architecture.
2. **Operator Gateway (`lumenbridge/ops.py`):** The safety checkpoint. Forces physical memory reallocation if non-contiguous tensors attempt to cross the language boundary.
3. **Native Binary Core (`src/`):** The compiled dynamic library. Routes execution to standard C++ ATen ops on CPU/Mac, or custom Shared Memory Tiling kernels on NVIDIA GPUs.

---

## 📁 Directory Structure

```text
LumenBridge/
├── include/
│   └── encoder.hpp            # C++ Header declarations and hardware routing signatures
├── src/
│   ├── bindings.cpp           # PyBind11 Python-to-C++ mapping logic
│   ├── encoder.cpp            # Main hardware router and ATen convolution logic
│   └── cuda/
│       └── conv_kernels.cu    # (Drafted) Shared memory tiling CUDA optimizations
├── lumenbridge/               # Python Package
│   ├── __init__.py
│   ├── model.py               # LumenBridgeStem nn.Module wrapper
│   ├── ops.py                 # Safe operator gateway
│   └── reference.py           # Pure-Python shadow model for INT8 PTQ Calibration
├── scripts/
│   ├── quantize.py            # Local PTQ calibration and footprint verification
│   └── export_onnx.py         # Static graph compilation for TensorRT deployment
├── tests/
│   └── test_pipeline.py       # End-to-end memory safety and mathematical verification
├── setup.py                   # Hardware-aware C++/CUDA build system
└── .gitignore
```

---

## ⚙️ Current Project State & Roadmap

The local development phase (macOS/Apple Silicon) is fully complete, mathematically verified, and tested. The project is currently staged for migration to AWS.

- [x] **Phase 1:** Native C++ Core Engine & PyBind11 Integration
- [x] **Phase 2:** Python Object Abstraction (`nn.Module`) & Memory Guardrails
- [x] **Phase 3:** Hardware-Aware Build System Configuration (`setup.py`)
- [x] **Phase 4:** Custom CUDA Kernel Architecture Drafted (Shared Memory Tiling)
- [x] **Phase 5:** Local INT8 Quantization & Calibration via `qnnpack` (~75% memory reduction verified)
- [x] **Phase 6:** Static Graph Export (`ONNX`)
- [ ] **Phase 7 (Next Step):** AWS GPU Deployment (g4dn/g5 instances)
- [ ] **Phase 8:** NVIDIA Docker Containerization & TensorRT INT8 Engine Compilation

---

## 💻 Local Installation & Usage

LumenBridge requires a local compilation phase to build the native C++ bindings for your specific machine.

**1. Environment Setup & Build:**
```bash
# Activate your environment
source .venv/bin/activate

# Install core dependencies
pip install torch numpy

# Build the C++ Extension (Editable mode for development)
pip install -e .
```

**2. Verify the Pipeline:**
```bash
python tests/test_pipeline.py
```

**3. Implementation Example:**
```python
import torch
from lumenbridge import LumenBridgeStem

# Instantiate the high-performance visual stem
tokenizer = LumenBridgeStem(d_model=768, patch_size=16)

# Simulate raw image inputs [Batch, Channels, Height, Width]
raw_images = torch.rand(4, 3, 224, 224)

# Execute fast forward pass through the C++ engine
visual_tokens = tokenizer(raw_images)

print(f"Token Output Shape: {visual_tokens.shape}")
# Expected Output: torch.Size([4, 196, 768])
```