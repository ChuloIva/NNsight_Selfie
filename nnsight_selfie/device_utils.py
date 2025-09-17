"""
Device detection and management utilities for cross-platform compatibility.

This module provides utilities for automatically detecting the best available
device (MPS for Mac, CUDA for NVIDIA GPUs, CPU fallback) and managing
device-specific operations.
"""

import torch
import platform
import warnings
from typing import Optional, Union


def get_optimal_device() -> str:
    """
    Automatically detect the best available device for the current system.
    
    Priority order:
    1. MPS (Metal Performance Shaders) on Apple Silicon Macs
    2. CUDA on NVIDIA GPUs
    3. CPU as fallback
    
    Returns:
        String representing the optimal device ("mps", "cuda", or "cpu")
    """
    # Check for MPS (Apple Silicon Mac support)
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return "mps"
        else:
            warnings.warn(
                "MPS is available but not built in this PyTorch installation. "
                "Falling back to CPU. Consider updating PyTorch for MPS support."
            )
    
    # Check for CUDA
    if torch.cuda.is_available():
        return "cuda"
    
    # Fallback to CPU
    return "cpu"


def get_device_map(device: Optional[Union[str, torch.device]] = None) -> Union[str, dict]:
    """
    Get appropriate device mapping for model loading.
    
    Args:
        device: Optional device specification. If None, auto-detects optimal device.
        
    Returns:
        Device mapping suitable for use with transformers models
    """
    if device is None:
        device = get_optimal_device()
    
    if isinstance(device, torch.device):
        device = str(device)
    
    # Handle device string conversion
    if device == "mps":
        # For MPS, we need to use "auto" for model loading but ensure tensors go to MPS
        return "auto"
    elif device == "cuda":
        return "auto"
    elif device == "cpu":
        return "cpu" 
    else:
        return device


def ensure_device_compatibility(tensor: torch.Tensor, target_device: str) -> torch.Tensor:
    """
    Ensure tensor is on the correct device, handling device-specific quirks.
    
    Args:
        tensor: Input tensor
        target_device: Target device string
        
    Returns:
        Tensor moved to target device
    """
    if target_device == "mps":
        # MPS has some limitations, handle gracefully
        try:
            return tensor.to("mps")
        except Exception as e:
            warnings.warn(f"Failed to move tensor to MPS: {e}. Using CPU.")
            return tensor.to("cpu")
    else:
        return tensor.to(target_device)


def get_system_info() -> dict:
    """
    Get system information relevant for device selection.
    
    Returns:
        Dictionary with system information
    """
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }
    
    # Add device-specific info
    if torch.backends.mps.is_available():
        info["mps_available"] = True
        info["mps_built"] = torch.backends.mps.is_built()
    else:
        info["mps_available"] = False
        info["mps_built"] = False
        
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_version"] = torch.version.cuda
        info["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.device_count() > 0:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    else:
        info["cuda_available"] = False
        
    return info


def print_device_info():
    """Print detailed device information for debugging."""
    info = get_system_info()
    optimal_device = get_optimal_device()
    
    print("=== Device Information ===")
    print(f"Platform: {info['platform']} {info['machine']}")
    print(f"Python: {info['python_version']}")
    print(f"PyTorch: {info['pytorch_version']}")
    print(f"Optimal Device: {optimal_device}")
    print()
    
    print("=== MPS Support ===")
    print(f"MPS Available: {info['mps_available']}")
    print(f"MPS Built: {info['mps_built']}")
    if info['mps_available'] and not info['mps_built']:
        print("⚠️  MPS is available but not built in this PyTorch installation")
        print("   Consider updating PyTorch: pip install torch torchvision torchaudio")
    print()
    
    print("=== CUDA Support ===")
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"Device Count: {info['cuda_device_count']}")
        if info.get('cuda_device_name'):
            print(f"Primary Device: {info['cuda_device_name']}")
    print()


def optimize_model_for_device(model, device: str):
    """
    Apply device-specific optimizations to a model.
    
    Args:
        model: The model to optimize
        device: Target device string
        
    Returns:
        Optimized model
    """
    if device == "mps":
        # MPS-specific optimizations
        # Note: Some operations may not be supported on MPS
        try:
            model = model.to("mps")
            # You might want to disable certain features that don't work on MPS
            if hasattr(model, 'config'):
                # Some models have compilation options that don't work on MPS
                if hasattr(model.config, 'use_cache'):
                    # Cache might cause issues on MPS in some cases
                    pass
        except Exception as e:
            warnings.warn(f"MPS optimization failed: {e}. Using CPU.")
            model = model.to("cpu")
            
    elif device == "cuda":
        # CUDA-specific optimizations
        model = model.to("cuda")
        # Enable CUDA optimizations if available
        if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
            torch.backends.cuda.flash_sdp_enabled(True)
            
    else:
        # CPU optimizations
        model = model.to("cpu")
        
    return model


class DeviceManager:
    """
    Context manager for handling device-specific operations.
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or get_optimal_device()
        self.original_device = None
        
    def __enter__(self):
        if self.device == "mps":
            # Set MPS as default device if possible
            try:
                if hasattr(torch, 'set_default_device'):
                    self.original_device = torch.get_default_device()
                    torch.set_default_device("mps")
            except Exception:
                pass
        return self.device
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_device is not None:
            try:
                torch.set_default_device(self.original_device)
            except Exception:
                pass