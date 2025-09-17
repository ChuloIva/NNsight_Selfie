"""
Basic usage examples for NNsight Selfie.

This script demonstrates how to use the ModelAgnosticSelfie class
for neural network interpretation across different model architectures.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nnsight_selfie import ModelAgnosticSelfie, InterpretationPrompt, print_device_info, get_optimal_device
import torch


def example_device_detection():
    """Example showing automatic device detection."""
    print("=== Device Detection Example ===")
    
    # Show available devices
    print_device_info()
    
    optimal_device = get_optimal_device()
    print(f"Optimal device detected: {optimal_device}")
    print()


def example_gpt2_interpretation():
    """Example using GPT-2 for token interpretation with automatic device selection."""
    print("=== GPT-2 Token Interpretation Example ===")
    
    # Initialize model with automatic device detection
    # This will use MPS on Apple Silicon Macs, CUDA on NVIDIA GPUs, or CPU as fallback
    selfie = ModelAgnosticSelfie("openai-community/gpt2")
    print(f"Model initialized on: {selfie.device}")
    print()
    
    # Create interpretation prompt
    interpretation_prompt = InterpretationPrompt.create_concept_prompt(selfie.model.tokenizer)
    
    # Define prompt and tokens to interpret
    original_prompt = "The Eiffel Tower is located in Paris"
    tokens_to_interpret = [(5, 2), (8, 4)]  # (layer, token_position) pairs
    
    # Perform interpretation
    results = selfie.interpret(
        original_prompt=original_prompt,
        interpretation_prompt=interpretation_prompt,
        tokens_to_interpret=tokens_to_interpret,
        max_new_tokens=10
    )
    
    # Display results
    for i in range(len(results['prompt'])):
        print(f"Token: '{results['token_decoded'][i]}' (Layer {results['layer'][i]}, Position {results['token'][i]})")
        print(f"Interpretation: {results['interpretation'][i]}")
        print("-" * 50)


def example_vector_interpretation():
    """Example of interpreting arbitrary activation vectors."""
    print("\n=== Vector Interpretation Example ===")
    
    # Initialize model with automatic device selection
    selfie = ModelAgnosticSelfie("openai-community/gpt2")
    print(f"Model device: {selfie.device}")
    
    # Create interpretation prompt
    interpretation_prompt = InterpretationPrompt.create_sentiment_prompt(selfie.model.tokenizer)
    
    # Extract some activations to use as vectors
    prompt = "I love sunny days"
    activations = selfie.get_activations(prompt, layer_indices=[6], token_indices=[2, 3])
    
    # Convert to list of vectors
    vectors = [activations[6][:, i, :] for i in range(2)]
    
    # Interpret vectors
    interpretations = selfie.interpret_vectors(
        vectors=vectors,
        interpretation_prompt=interpretation_prompt,
        injection_layer=2,
        max_new_tokens=8
    )
    
    # Display results
    for i, interpretation in enumerate(interpretations):
        print(f"Vector {i}: {interpretation}")


def example_different_model_architectures():
    """Example showing usage with different model types."""
    print("\n=== Different Model Architecture Example ===")
    
    model_names = [
        "openai-community/gpt2",  # GPT-style
        # "microsoft/DialoGPT-small",  # Another GPT-style
        # "distilbert-base-uncased",  # BERT-style (if available)
    ]
    
    for model_name in model_names:
        try:
            print(f"\n--- Testing {model_name} ---")
            selfie = ModelAgnosticSelfie(model_name)
            
            # Show detected layer structure
            print(f"Detected {len(selfie.layer_paths)} layers:")
            for i, path in enumerate(selfie.layer_paths[:3]):  # Show first 3
                print(f"  Layer {i}: {path}")
            if len(selfie.layer_paths) > 3:
                print(f"  ... and {len(selfie.layer_paths) - 3} more")
            
            # Quick activation extraction test
            activations = selfie.get_activations("Hello world", layer_indices=[0])
            print(f"Activation shape: {activations[0].shape}")
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")


def example_custom_interpretation_prompt():
    """Example of creating custom interpretation prompts."""
    print("\n=== Custom Interpretation Prompt Example ===")
    
    selfie = ModelAgnosticSelfie("openai-community/gpt2")
    
    # Create custom prompt with multiple placeholders
    custom_sequence = [
        "The activation ", 
        None,  # First placeholder
        " represents the concept of ",
        None,  # Second placeholder  
        " in the context of"
    ]
    
    custom_prompt = InterpretationPrompt(selfie.model.tokenizer, custom_sequence)
    print(f"Custom prompt: '{custom_prompt.get_prompt()}'")
    print(f"Insert locations: {custom_prompt.get_insert_locations()}")
    
    # Note: This would need special handling in the main interpretation function
    # for multiple placeholders, which could be added as an enhancement


def example_activation_analysis():
    """Example of analyzing activations across layers and tokens."""
    print("\n=== Activation Analysis Example ===")
    
    selfie = ModelAgnosticSelfie("openai-community/gpt2")
    
    prompt = "The capital of France is Paris"
    
    # Get activations for multiple layers
    layer_indices = [0, 3, 6, 9]  # Sample layers
    activations = selfie.get_activations(prompt, layer_indices=layer_indices)
    
    print(f"Analyzing prompt: '{prompt}'")
    print(f"Token count: {len(selfie.model.tokenizer.encode(prompt))}")
    
    for layer_idx, activation in activations.items():
        print(f"Layer {layer_idx}: {activation.shape}")
        
        # Show activation statistics
        mean_activation = activation.mean().item()
        std_activation = activation.std().item()
        print(f"  Mean: {mean_activation:.4f}, Std: {std_activation:.4f}")


if __name__ == "__main__":
    print("NNsight Selfie Examples")
    print("=" * 50)
    
    try:
        # Show device information first
        example_device_detection()
        
        example_gpt2_interpretation()
        example_vector_interpretation() 
        example_different_model_architectures()
        example_custom_interpretation_prompt()
        example_activation_analysis()
        
        print("\n=== All examples completed successfully! ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have installed nnsight and have internet connection for model downloads.")
        print("On Apple Silicon Macs, ensure you have a recent version of PyTorch with MPS support:")