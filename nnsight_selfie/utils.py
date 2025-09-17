"""
Utility functions for model-agnostic selfie functionality.

This module provides helper functions for working with different model architectures,
extracting layer paths, and performing common operations.
"""

from typing import List, Dict, Any, Optional, Union
import torch
from tqdm import tqdm


def get_model_layers(model) -> List[str]:
    """
    Get layer paths for different model architectures.
    
    Args:
        model: NNsight LanguageModel instance
        
    Returns:
        List of layer path strings for accessing transformer layers
    """
    # Try to detect model architecture and return appropriate layer paths
    model_config = model.config
    model_type = getattr(model_config, 'model_type', '').lower()
    
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-style models (GPT-2, GPT-Neo, etc.)
        num_layers = len(model.transformer.h)
        return [f'transformer.h.{i}' for i in range(num_layers)]
    
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLaMA-style models
        num_layers = len(model.model.layers)
        return [f'model.layers.{i}' for i in range(num_layers)]
    
    elif hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
        # BERT-style models
        num_layers = len(model.bert.encoder.layer)
        return [f'bert.encoder.layer.{i}' for i in range(num_layers)]
    
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        # T5/BART encoder
        num_layers = len(model.encoder.layer)
        return [f'encoder.layer.{i}' for i in range(num_layers)]
    
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        # T5/BART decoder  
        num_layers = len(model.decoder.layers)
        return [f'decoder.layers.{i}' for i in range(num_layers)]
    
    else:
        # Fallback: try to find layers automatically
        return _auto_detect_layers(model)


def _auto_detect_layers(model) -> List[str]:
    """
    Automatically detect layer paths by traversing the model structure.
    
    Args:
        model: NNsight LanguageModel instance
        
    Returns:
        List of detected layer paths
    """
    layer_paths = []
    
    def find_layers(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Look for common transformer layer patterns
            if any(pattern in name.lower() for pattern in ['layer', 'block', 'h']):
                if hasattr(child, '__len__'):  # It's a list/sequential of layers
                    for i in range(len(child)):
                        layer_paths.append(f"{full_name}.{i}")
                else:
                    layer_paths.append(full_name)
            else:
                find_layers(child, full_name)
    
    find_layers(model)
    return layer_paths


def get_layer_by_path(model, layer_path: str):
    """
    Get a layer object by its path string.
    
    Args:
        model: NNsight LanguageModel instance
        layer_path: String path to the layer (e.g., 'transformer.h.0')
        
    Returns:
        The layer object
    """
    parts = layer_path.split('.')
    current = model
    
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    
    return current


def interpret_vectors(
    model,
    vectors: List[torch.Tensor],
    interpretation_prompt,
    injection_layer: int = 3,
    batch_size: int = 8,
    max_new_tokens: int = 30,
    overlay_strength: float = 1.0
) -> List[str]:
    """
    Interpret a list of activation vectors using a model-agnostic approach.
    
    Args:
        model: ModelAgnosticSelfie instance
        vectors: List of activation tensors
        interpretation_prompt: InterpretationPrompt instance
        injection_layer: Layer index to inject at
        batch_size: Batch size for processing
        max_new_tokens: Maximum tokens to generate
        overlay_strength: Strength of intervention
        
    Returns:
        List of interpretation strings
    """
    return model.interpret_vectors(
        vectors,
        interpretation_prompt,
        injection_layer,
        batch_size,
        max_new_tokens,
        overlay_strength
    )


def batch_process_interpretations(
    model,
    prompts: List[str],
    interpretation_prompt,
    tokens_to_interpret_per_prompt: List[List[tuple]],
    batch_size: int = 4
) -> List[Dict[str, Any]]:
    """
    Process multiple prompts for interpretation in batches.
    
    Args:
        model: ModelAgnosticSelfie instance
        prompts: List of input prompts
        interpretation_prompt: InterpretationPrompt instance
        tokens_to_interpret_per_prompt: List of token lists for each prompt
        batch_size: Batch size for processing
        
    Returns:
        List of interpretation result dictionaries
    """
    results = []
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_tokens = tokens_to_interpret_per_prompt[i:i + batch_size]
        
        for prompt, tokens_to_interpret in zip(batch_prompts, batch_tokens):
            result = model.interpret(
                prompt,
                interpretation_prompt,
                tokens_to_interpret
            )
            results.append(result)
    
    return results


def create_token_grid(
    prompt: str,
    tokenizer,
    layer_range: Optional[tuple] = None,
    token_range: Optional[tuple] = None
) -> List[tuple]:
    """
    Create a grid of (layer, token) pairs for systematic interpretation.
    
    Args:
        prompt: Input prompt to tokenize
        tokenizer: Tokenizer to use
        layer_range: Tuple of (start_layer, end_layer) or None for all layers
        token_range: Tuple of (start_token, end_token) or None for all tokens
        
    Returns:
        List of (layer_idx, token_idx) tuples
    """
    tokens = tokenizer.encode(prompt)
    num_tokens = len(tokens)
    
    if token_range is None:
        token_indices = list(range(num_tokens))
    else:
        token_indices = list(range(
            max(0, token_range[0]),
            min(num_tokens, token_range[1])
        ))
    
    # Default layer range (would need model info for actual range)
    if layer_range is None:
        layer_indices = list(range(12))  # Default for medium-sized models
    else:
        layer_indices = list(range(layer_range[0], layer_range[1]))
    
    return [(layer, token) for layer in layer_indices for token in token_indices]


def aggregate_activations(
    activations: Dict[int, torch.Tensor],
    method: str = 'mean'
) -> torch.Tensor:
    """
    Aggregate activations across layers or tokens.
    
    Args:
        activations: Dictionary of layer_idx -> activation tensors
        method: Aggregation method ('mean', 'sum', 'max', 'last')
        
    Returns:
        Aggregated activation tensor
    """
    activation_list = list(activations.values())
    
    if method == 'mean':
        return torch.stack(activation_list).mean(dim=0)
    elif method == 'sum':
        return torch.stack(activation_list).sum(dim=0)
    elif method == 'max':
        return torch.stack(activation_list).max(dim=0)[0]
    elif method == 'last':
        return activation_list[-1]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def compute_activation_similarity(
    activation1: torch.Tensor,
    activation2: torch.Tensor,
    metric: str = 'cosine'
) -> float:
    """
    Compute similarity between two activation vectors.
    
    Args:
        activation1: First activation tensor
        activation2: Second activation tensor  
        metric: Similarity metric ('cosine', 'l2', 'dot')
        
    Returns:
        Similarity score
    """
    if metric == 'cosine':
        return torch.nn.functional.cosine_similarity(
            activation1.flatten(), 
            activation2.flatten(), 
            dim=0
        ).item()
    elif metric == 'l2':
        return -torch.nn.functional.mse_loss(activation1, activation2).item()
    elif metric == 'dot':
        return torch.dot(activation1.flatten(), activation2.flatten()).item()
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")