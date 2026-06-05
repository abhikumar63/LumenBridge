# lumenbridge/reference.py
import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub

class ReferenceLumenBridgeStem(nn.Module):
    """
    A pure-Python shadow of our C++ engine, strictly used for 
    PTQ Calibration and exporting INT8 scaling factors.
    """
    def __init__(self, in_channels=3, d_model=768, patch_size=16):
        super().__init__()
        
        # Entry point: Converts FP32 inputs to INT8
        self.quant = QuantStub()
        
        # Phase 1: Depthwise Convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=patch_size, 
            stride=patch_size, 
            groups=in_channels, 
            bias=False
        )
        
        # Phase 2: Pointwise Convolution
        self.pointwise = nn.Conv2d(
            in_channels, d_model, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        
        # Exit point: Converts INT8 back to FP32 for the LLM Attention layers
        self.dequant = DeQuantStub()

    def forward(self, x):
        # 1. Quantize
        x = self.quant(x)
        
        # 2. INT8 Math
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        # 3. Dequantize
        x = self.dequant(x)
        
        # 4. Flatten and permute to sequence (just like our C++ engine)
        x = x.flatten(2).transpose(1, 2)
        return x.contiguous()