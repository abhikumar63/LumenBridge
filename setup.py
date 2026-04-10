import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Depthwise: processes each channel independently
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, 
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        # Pointwise: projects across channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, 
            stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))

class VisualTokenizerStem(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        # Aggressive early downsampling (e.g., stride 4)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, d_model, stride=2)
        )

    def forward(self, x):
        # Input: [B, 3, H, W]
        x = self.stem(x) # Output: [B, D_model, H', W']
        
        B, C, H_prime, W_prime = x.shape
        
        # Flatten spatial dimensions to create a sequence
        # Permute to [B, Seq_Len, D_model]
        x = x.flatten(2).transpose(1, 2)
        
        # CRITICAL: Force contiguous memory layout before handing off to C++/CUDA or next layers
        return x.contiguous()