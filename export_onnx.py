import torch
from lumenbridge.reference import ReferenceLumenBridgeStem

model = ReferenceLumenBridgeStem(d_model=768, patch_size=16)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, 
    dummy_input, 
    "lumenbridge_stem.onnx", 
    export_params=True, 
    opset_version=11, 
    do_constant_folding=True, 
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("Exported to lumenbridge_stem.onnx")
