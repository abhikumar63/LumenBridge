# Use NVIDIA's official PyTorch + TensorRT base image
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /workspace/lumenbridge

# Copy the repository into the container
COPY . .

# Force the hardware router in setup.py to compile the .cu files
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="7.5" 
# Note: 7.5 is the architecture for AWS g4dn (T4 GPUs). 

# Install dependencies and build the C++ / CUDA extensions
RUN pip install --upgrade pip
RUN pip install tensorrt
RUN pip install -e .

# Default command: Open bash so we can run the TensorRT export script
CMD ["/bin/bash"]