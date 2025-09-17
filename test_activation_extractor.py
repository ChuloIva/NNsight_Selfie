#!/usr/bin/env python3
"""
Test the activation extractor specifically.
"""

# FOR AMD GPU
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["AMD_SERIALIZE_KERNEL"] = "3"
os.environ["TORCH_USE_HIP_DSA"] = "1"

import torch
from transformers import AutoTokenizer
import nnsight
import sys
import traceback

def test_activation_extractor():
    """Test our activation extractor with the fixes."""
    print("🔍 Testing Activation Extractor...")
    
    # Use small model for testing
    model_name = "microsoft/DialoGPT-small"
    print(f"Loading model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = nnsight.LanguageModel(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False
    )
    
    print("✅ Model and tokenizer loaded")
    
    # Test the extractor
    from nnsight_selfie.repeng_activation_extractor import RepengActivationExtractor
    
    # Create extractor with just first 3 layers
    extractor = RepengActivationExtractor(model, tokenizer, layer_indices=[0, 1, 2])
    
    print("✅ Extractor created")
    
    # Test with simple inputs
    test_inputs = ["Hello world", "How are you", "Good morning"]
    
    print("🧪 Testing activation extraction...")
    
    try:
        activations = extractor.extract_last_token_activations(
            test_inputs, 
            batch_size=2,
            show_progress=True
        )
        
        print("✅ Activation extraction successful!")
        for layer_idx, acts in activations.items():
            print(f"   Layer {layer_idx}: {acts.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Activation extraction failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_activation_extractor()
    sys.exit(0 if success else 1)