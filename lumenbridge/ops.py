# lumenbridge/ops.py
import torch
import lumenbridge_core

def project_visual_patches(tensor: torch.Tensor, d_model: int, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Safe wrapper for the native lumenbridge_core C++ extension.
    Ensures input requirements are pristine before dropping to binary compilation paths.
    """
    # System verification check before crossing the runtime boundary
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
        
    return lumenbridge_core.project_patches(tensor, d_model, kernel_size, stride)