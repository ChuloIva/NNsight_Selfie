"""
MPS (Metal Performance Shaders) compatibility examples for Apple Silicon Macs.

This script demonstrates how NNsight Selfie automatically detects and uses MPS
on Apple Silicon Macs, with graceful fallbacks to CPU when needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nnsight_selfie import (
    ModelAgnosticSelfie, 
    InterpretationPrompt, 
    print_device_info, 
    get_optimal_device,
    DeviceManager
)
import torch
import platform


def check_mps_availability():
    """Check MPS availability and print detailed information."""
    print("=== MPS Availability Check ===")
    
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        print("✅ MPS is available")
        if torch.backends.mps.is_built():
            print("✅ MPS is built and ready to use")
        else:
            print("⚠️  MPS is available but not built in this PyTorch installation")
            print("   Consider updating PyTorch for MPS support:")
            print("   pip install torch torchvision torchaudio")
    else:
        print("❌ MPS is not available")
        if platform.system() == "Darwin" and "arm" in platform.machine().lower():
            print("   You're on Apple Silicon but MPS is not available.")
            print("   Make sure you have PyTorch >= 1.12.0 with MPS support.")
        else:
            print("   MPS is only available on Apple Silicon Macs.")
    
    print()


def example_automatic_device_selection():
    """Example showing automatic device selection with MPS priority."""
    print("=== Automatic Device Selection ===")
    
    optimal_device = get_optimal_device()
    print(f"Optimal device: {optimal_device}")
    
    # Initialize model with automatic device detection
    print("Initializing model with automatic device detection...")
    selfie = ModelAgnosticSelfie("sshleifer/tiny-gpt2")  # Use tiny model for speed
    
    print(f"Model device: {selfie.device}")
    print(f"Model loaded successfully!")
    print()
    
    return selfie


def example_explicit_device_selection():
    """Example showing explicit device selection."""
    print("=== Explicit Device Selection ===")
    
    devices_to_try = ["mps", "cuda", "cpu"]
    
    for device in devices_to_try:
        try:
            print(f"Trying device: {device}")
            selfie = ModelAgnosticSelfie("sshleifer/tiny-gpt2", device=device)
            print(f"✅ Successfully loaded on {selfie.device}")
            return selfie
        except Exception as e:
            print(f"❌ Failed to load on {device}: {e}")
            continue
    
    print("Could not load model on any device")
    return None


def example_device_manager():
    """Example using DeviceManager context manager."""
    print("=== Device Manager Example ===")
    
    with DeviceManager() as device:
        print(f"Using device: {device}")
        
        # Create some tensors that will be on the optimal device
        tensor1 = torch.randn(10, 768)
        tensor2 = torch.randn(10, 768)
        
        print(f"Tensor device: {tensor1.device}")
        
        # Perform operations
        result = torch.mm(tensor1, tensor2.T)
        print(f"Result shape: {result.shape}")
        print(f"Operation completed successfully on {device}")
    
    print()


def example_mps_specific_interpretation():
    """Example running interpretation specifically on MPS if available."""
    print("=== MPS-Specific Interpretation Example ===")
    
    if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
        print("MPS not available, skipping MPS-specific example")
        return
    
    # Force MPS device
    try:
        selfie = ModelAgnosticSelfie("sshleifer/tiny-gpt2", device="mps")
        print(f"Model loaded on: {selfie.device}")
        
        # Create interpretation prompt
        interpretation_prompt = InterpretationPrompt.create_concept_prompt(
            selfie.model.tokenizer
        )
        
        # Test basic interpretation
        prompt = "Hello world"
        print(f"Testing interpretation with prompt: '{prompt}'")
        
        # Get activations
        activations = selfie.get_activations(prompt, layer_indices=[0, 1])
        print(f"Extracted activations from {len(activations)} layers")
        
        for layer_idx, activation in activations.items():
            print(f"Layer {layer_idx}: {activation.shape}, device: {activation.device}")
        
        # Test interpretation
        results = selfie.interpret(
            original_prompt=prompt,
            interpretation_prompt=interpretation_prompt,
            tokens_to_interpret=[(1, 0)],  # Layer 1, first token
            max_new_tokens=5
        )
        
        print(f"Interpretation result: {results['interpretation'][0]}")
        print("✅ MPS interpretation completed successfully!")
        
    except Exception as e:
        print(f"❌ MPS interpretation failed: {e}")
        print("This might happen with models that have operations not yet supported on MPS")
    
    print()


def example_performance_comparison():
    """Compare performance across different devices (if available)."""
    print("=== Performance Comparison ===")
    
    import time
    
    devices = ["cpu"]
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    
    results = {}
    
    for device in devices:
        try:
            print(f"Testing performance on {device}...")
            
            start_time = time.time()
            
            # Initialize model
            selfie = ModelAgnosticSelfie("sshleifer/tiny-gpt2", device=device)
            
            # Run a simple operation
            activations = selfie.get_activations(
                "The quick brown fox jumps", 
                layer_indices=[0, 1, 2]
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            results[device] = elapsed
            print(f"  {device}: {elapsed:.2f} seconds")
            
        except Exception as e:
            print(f"  {device}: Failed - {e}")
    
    if len(results) > 1:
        fastest = min(results, key=results.get)
        print(f"\nFastest device: {fastest} ({results[fastest]:.2f}s)")
    
    print()


def troubleshooting_tips():
    """Print troubleshooting tips for MPS issues."""
    print("=== Troubleshooting Tips ===")
    
    print("If you encounter MPS-related issues:")
    print()
    print("1. Update PyTorch:")
    print("   pip install --upgrade torch torchvision torchaudio")
    print()
    print("2. Check your macOS version:")
    print("   MPS requires macOS 12.3+ and Apple Silicon (M1/M2/M3)")
    print()
    print("3. Some operations may not be supported on MPS yet:")
    print("   The library will automatically fall back to CPU in such cases")
    print()
    print("4. Memory issues:")
    print("   MPS shares memory with the system, try smaller models or batch sizes")
    print()
    print("5. Force CPU if needed:")
    print("   selfie = ModelAgnosticSelfie(model_name, device='cpu')")
    print()


if __name__ == "__main__":
    print("NNsight Selfie - MPS Compatibility Examples")
    print("=" * 60)
    
    # First show system information
    print_device_info()
    
    try:
        check_mps_availability()
        selfie = example_automatic_device_selection()
        
        if selfie:
            example_explicit_device_selection()
            example_device_manager()
            example_mps_specific_interpretation()
            example_performance_comparison()
        
        troubleshooting_tips()
        
        print("=== MPS compatibility examples completed! ===")
        
    except Exception as e:
        print(f"Error running MPS examples: {e}")
        import traceback
        traceback.print_exc()