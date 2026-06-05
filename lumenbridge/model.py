# lumenbridge/model.py
import torch
import torch.nn as nn
from .ops import project_visual_patches

class LumenBridgeStem(nn.Module):
    def __init__(self, d_model: int = 768, patch_size: int = 16):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        
        # Layer normalization to apply directly to the generated tokens
        # to ensure numerical stability before feeding them into deep LLM layers
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Ingests a raw image tensor and returns normalized linguistic-style tokens.
        
        Input:  [Batch, 3, Height, Width]
        Output: [Batch, Sequence_Length, D_model]
        """
        # 1. Execute the high-performance C++ Strided Convolution & Flattening Engine
        tokens = project_visual_patches(
            pixel_values, 
            d_model=self.d_model, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # 2. Apply standard transformer layer normalization
        return self.layer_norm(tokens)