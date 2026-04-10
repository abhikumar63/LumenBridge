import torch
import numpy as np

def check_mps():
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
        return False

    print("MPS is available!")
    
    # Run a quick test
    try:
        mps_device = torch.device("mps")
        x = torch.ones(5, device=mps_device)
        print("Successfully created a tensor on the MPS device:")
        print(x)
        return True
    except Exception as e:
        print(f"Failed to create tensor on MPS: {e}")
        return False

if __name__ == "__main__":
    check_mps()
