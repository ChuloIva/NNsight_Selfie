"""
Advanced usage examples for NNsight Selfie.

This script demonstrates more sophisticated interpretation techniques
and analysis workflows.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nnsight_selfie import ModelAgnosticSelfie, InterpretationPrompt
from nnsight_selfie.utils import (
    create_token_grid, 
    aggregate_activations,
    compute_activation_similarity,
    batch_process_interpretations
)
import torch
import numpy as np


def example_systematic_layer_analysis():
    """Systematically analyze interpretations across different layers."""
    print("=== Systematic Layer Analysis ===")
    
    selfie = ModelAgnosticSelfie("openai-community/gpt2", device_map="cpu")
    interpretation_prompt = InterpretationPrompt.create_concept_prompt(selfie.model.tokenizer)
    
    prompt = "The cat sat on the mat"
    target_token = 4  # "mat"
    
    print(f"Analyzing token '{selfie.model.tokenizer.decode([selfie.model.tokenizer.encode(prompt)[target_token]])}'")
    print("across different layers...")
    
    # Test multiple layers
    layer_range = range(0, min(8, len(selfie.layer_paths)), 2)  # Every 2nd layer
    
    for layer_idx in layer_range:
        tokens_to_interpret = [(layer_idx, target_token)]
        
        results = selfie.interpret(
            original_prompt=prompt,
            interpretation_prompt=interpretation_prompt,
            tokens_to_interpret=tokens_to_interpret,
            max_new_tokens=6
        )
        
        interpretation = results['interpretation'][0].strip()
        print(f"Layer {layer_idx:2d}: {interpretation}")


def example_activation_similarity_analysis():
    """Analyze similarity between activations of different words."""
    print("\n=== Activation Similarity Analysis ===")
    
    selfie = ModelAgnosticSelfie("openai-community/gpt2", device_map="cpu")
    
    # Compare activations for similar/different concepts
    prompts = [
        "The dog ran quickly",
        "The cat ran quickly", 
        "The car moved quickly",
        "The dog walked slowly"
    ]
    
    target_token = 1  # Second token (dog/cat/car/dog)
    layer_idx = 6
    
    print("Comparing activations for different entities:")
    
    activations = []
    for prompt in prompts:
        acts = selfie.get_activations(prompt, layer_indices=[layer_idx], token_indices=[target_token])
        activations.append(acts[layer_idx][:, 0, :])  # Remove token dimension
    
    # Compute pairwise similarities
    for i, prompt1 in enumerate(prompts):
        for j, prompt2 in enumerate(prompts[i+1:], i+1):
            similarity = compute_activation_similarity(
                activations[i], 
                activations[j], 
                metric='cosine'
            )
            token1 = prompt1.split()[1]
            token2 = prompt2.split()[1] 
            print(f"'{token1}' vs '{token2}': {similarity:.3f}")


def example_intervention_strength_analysis():
    """Analyze how intervention strength affects interpretations."""
    print("\n=== Intervention Strength Analysis ===")
    
    selfie = ModelAgnosticSelfie("openai-community/gpt2", device_map="cpu")
    interpretation_prompt = InterpretationPrompt.create_concept_prompt(selfie.model.tokenizer)
    
    prompt = "Paris is a beautiful city"
    token_to_interpret = (6, 0)  # Layer 6, first token
    
    strengths = [0.2, 0.5, 0.8, 1.0]
    
    print(f"Testing different intervention strengths for token interpretation:")
    
    for strength in strengths:
        results = selfie.interpret(
            original_prompt=prompt,
            interpretation_prompt=interpretation_prompt,
            tokens_to_interpret=[token_to_interpret],
            overlay_strength=strength,
            max_new_tokens=8
        )
        
        interpretation = results['interpretation'][0].strip()
        print(f"Strength {strength}: {interpretation}")


def example_activation_arithmetic():
    """Perform arithmetic operations on activations."""
    print("\n=== Activation Arithmetic ===") 
    
    selfie = ModelAgnosticSelfie("openai-community/gpt2", device_map="cpu")
    interpretation_prompt = InterpretationPrompt.create_concept_prompt(selfie.model.tokenizer)
    
    # Get activations for different concepts
    concepts = {
        "king": "The king ruled wisely",
        "queen": "The queen was elegant", 
        "man": "The man walked home",
        "woman": "The woman read quietly"
    }
    
    layer_idx = 6
    activations = {}
    
    for concept, prompt in concepts.items():
        acts = selfie.get_activations(prompt, layer_indices=[layer_idx], token_indices=[1])
        activations[concept] = acts[layer_idx][:, 0, :]
    
    # Perform king - man + woman ≈ queen
    try:
        result_vector = (activations["king"] - activations["man"] + activations["woman"])
        
        # Interpret the result
        interpretations = selfie.interpret_vectors(
            vectors=[result_vector],
            interpretation_prompt=interpretation_prompt,
            injection_layer=2,
            max_new_tokens=8
        )
        
        print("King - Man + Woman =", interpretations[0].strip())
        
    except Exception as e:
        print(f"Activation arithmetic failed: {e}")


def example_batch_interpretation():
    """Process multiple prompts efficiently in batches."""
    print("\n=== Batch Interpretation ===")
    
    selfie = ModelAgnosticSelfie("openai-community/gpt2", device_map="cpu")
    interpretation_prompt = InterpretationPrompt.create_entity_prompt(selfie.model.tokenizer)
    
    # Multiple prompts to process
    prompts = [
        "London is the capital of England",
        "Tokyo is a bustling metropolis", 
        "New York has tall skyscrapers",
        "Paris is known for its art"
    ]
    
    # Define tokens to interpret for each prompt
    tokens_per_prompt = [
        [(4, 0)],  # "London" 
        [(4, 0)],  # "Tokyo"
        [(4, 0), (4, 1)],  # "New York" 
        [(4, 0)]   # "Paris"
    ]
    
    print("Processing multiple prompts in batch...")
    
    results = batch_process_interpretations(
        selfie,
        prompts,
        interpretation_prompt,
        tokens_per_prompt,
        batch_size=2
    )
    
    # Display results
    for i, result in enumerate(results):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        for j in range(len(result['interpretation'])):
            token = result['token_decoded'][j]
            interp = result['interpretation'][j].strip()
            print(f"  '{token}' -> {interp}")


def example_layer_activation_patterns():
    """Analyze how activation patterns change across layers."""
    print("\n=== Layer Activation Patterns ===")
    
    selfie = ModelAgnosticSelfie("openai-community/gpt2", device_map="cpu")
    
    prompt = "The scientist discovered something amazing"
    token_idx = 2  # "discovered"
    
    # Get activations from multiple layers
    layer_indices = list(range(0, min(len(selfie.layer_paths), 12), 2))
    activations = selfie.get_activations(prompt, layer_indices=layer_indices, token_indices=[token_idx])
    
    print(f"Activation statistics for token '{selfie.model.tokenizer.decode([selfie.model.tokenizer.encode(prompt)[token_idx]])}':")
    
    for layer_idx in layer_indices:
        activation = activations[layer_idx][:, 0, :]  # Remove batch and token dims
        
        # Compute statistics
        mean_val = activation.mean().item()
        std_val = activation.std().item()
        max_val = activation.max().item()
        sparsity = (activation.abs() < 0.1).float().mean().item()
        
        print(f"Layer {layer_idx:2d}: Mean={mean_val:6.3f}, Std={std_val:6.3f}, Max={max_val:6.3f}, Sparsity={sparsity:.3f}")


if __name__ == "__main__":
    print("NNsight Selfie Advanced Examples")
    print("=" * 50)
    
    try:
        example_systematic_layer_analysis()
        example_activation_similarity_analysis()
        example_intervention_strength_analysis()
        example_activation_arithmetic()
        example_batch_interpretation()
        example_layer_activation_patterns()
        
        print("\n=== All advanced examples completed successfully! ===")
        
    except Exception as e:
        print(f"Error running advanced examples: {e}")
        import traceback
        traceback.print_exc()