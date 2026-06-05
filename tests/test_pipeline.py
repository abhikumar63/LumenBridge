# tests/test_pipeline.py
import torch
import lumenbridge_core
# Import our unified object framework directly from the module namespace
from lumenbridge import LumenBridgeStem

def run_sanity_check():
    print("=== TEST 1: Raw C++ Core Validation ===")
    try:
        dummy_images = torch.rand(4, 3, 224, 224)
        print(f"[Python] Instantiated dummy images: {list(dummy_images.shape)}")
        
        # Testing raw binary module
        returned_tensor = lumenbridge_core.project_patches(dummy_images, 768, 16, 16)
        print(f"[C++ Core] Returned sequence layout shape: {list(returned_tensor.shape)}")
        
        expected_shape = [4, 196, 768]
        assert list(returned_tensor.shape) == expected_shape, "Math mismatch on raw core!"
        print("✓ SUCCESS: C++ tokenization math matches exactly.")
    except Exception as e:
        print(f"❌ FAILURE in Core Test: {e}")


    print("\n=== TEST 2: LumenBridge Folder Integration (Object-Oriented) ===")
    try:
        # Initialize our object-wrapped tokenizer stem
        # This simulates exactly how a downstream LLM pipeline would instantiate your project
        tokenizer = LumenBridgeStem(d_model=768, patch_size=16)
        print("[Python Module] Initialized LumenBridgeStem layer configuration.")
        
        # Pass a standard batch of images
        sample_batch = torch.rand(2, 3, 224, 224)
        print(f"[Python Module] Ingesting test batch into pipeline: {list(sample_batch.shape)}")
        
        # Execute the forward pass through model.py -> ops.py -> C++ -> LayerNorm
        final_tokens = tokenizer(sample_batch)
        print(f"[Python Module] Forward execution complete.")
        print(f"[Python Module] Outputs are ready for LLM Attention blocks.")
        print(f"  -> Final Token Grid Shape: {list(final_tokens.shape)}")
        
        # Expected shape for batch size 2: [2, 196, 768]
        assert list(final_tokens.shape) == [2, 196, 768], "Integration shape mismatch!"
        print("\n✓ SUCCESS: High-level Python abstraction layer fully verified.")
        
    except Exception as e:
        print(f"❌ FAILURE in Module Integration Test: {e}")


    print("\n=== TEST 3: Guardrails with a Non-Contiguous Tensor ===")
    try:
        non_contiguous_tensor = torch.rand(4, 224, 224, 3).permute(0, 3, 1, 2)
        print("[Python] Attempting to force non-contiguous memory...")
        lumenbridge_core.project_patches(non_contiguous_tensor, 768, 16, 16)
    except RuntimeError as e:
        print(f"✓ SUCCESS: Guardrail successfully caught exception:\n  -> {e}")

if __name__ == "__main__":
    run_sanity_check()