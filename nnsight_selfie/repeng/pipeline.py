"""
High-level RePENG pipeline to:
- Build datasets from cognitive patterns JSONL
- Extract last-token activations across layers
- Compute layer-wise steering vectors via PCA-diff (and variants)
- Inject multi-layer steering vectors at specified token positions

This module wires together:
- patterns_dataset (dataset building)
- repeng_activation_extractor (activation capture)
- repeng_steering_vectors (PCA-based vectors)
- repeng_multi_injection (token-level multi-layer injection)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass

from .patterns_dataset import build_all_datasets, PairType
from .repeng_activation_extractor import RepengActivationExtractor
from .repeng_steering_vectors import (
    RepengSteeringVectorGenerator,
    SteeringVector,
)
from .repeng_multi_injection import RepengMultiLayerInjector


MethodType = Literal["pca_diff", "pca_center", "mean_diff"]


@dataclass
class PatternSteeringBundle:
    pattern_name: str
    pair_type: PairType
    method: MethodType
    steering_vector: SteeringVector
    inputs_used: List[str]


def compute_pattern_steering_vectors(
    model,
    tokenizer,
    patterns_jsonl_path: str,
    pair_types: Optional[List[PairType]] = None,
    method: MethodType = "pca_diff",
    layer_range: Optional[Tuple[int, int]] = None,
    batch_size: int = 1,
    whiten: bool = False,
    show_progress: bool = True,
    max_patterns: Optional[int] = None,
) -> List[PatternSteeringBundle]:
    """
    For each cognitive pattern and requested pairing, compute multi-layer steering vectors.

    Args:
        model, tokenizer: NNsight LanguageModel and tokenizer
        patterns_jsonl_path: Path to positive_patterns.jsonl
        pair_types: Which pairings to compute (default all)
        method: PCA method (default pca_diff)
        layer_range: Optional (start, end) inclusive/exclusive pattern as in extractor
        batch_size: Activation extraction batch size
        whiten: Whether to whiten PCA
        show_progress: Progress bars for extraction
        max_patterns: Maximum number of patterns to process (for memory optimization)

    Returns:
        List of PatternSteeringBundle entries
    """
    datasets_map = build_all_datasets(patterns_jsonl_path, pair_types, max_patterns)
    extractor = RepengActivationExtractor(model, tokenizer)
    generator = RepengSteeringVectorGenerator(model_type=getattr(model, "model_name", "unknown"))

    bundles: List[PatternSteeringBundle] = []

    for pattern_name, pair_map in datasets_map.items():
        for pair_type, dataset in pair_map.items():
            if not dataset:
                continue

            # Extract activations in alternating order per repeng
            if layer_range is not None:
                # Build alternating inputs manually
                alternating_inputs = []
                for entry in dataset:
                    alternating_inputs.append(entry.positive)
                    alternating_inputs.append(entry.negative)
                
                # Handle specific layer indices vs range
                if hasattr(layer_range, '__iter__') and not isinstance(layer_range, range):
                    # layer_range is a list of specific indices
                    specific_layers = list(layer_range)
                elif isinstance(layer_range, range):
                    # Convert range to list of specific indices
                    specific_layers = list(layer_range)
                else:
                    # Assume start/end format
                    specific_layers = list(range(layer_range[0], layer_range[1] + 1))
                
                # Temporarily set extractor to use specific layers
                original_indices = extractor.layer_indices
                extractor.layer_indices = specific_layers
                
                try:
                    activations = extractor.extract_last_token_activations(
                        alternating_inputs, batch_size=batch_size, show_progress=show_progress
                    )
                finally:
                    # Restore original indices
                    extractor.layer_indices = original_indices
                    
                inputs_used = alternating_inputs
            else:
                activations, inputs_used = extractor.extract_dataset_activations(
                    dataset, batch_size=batch_size, show_progress=show_progress
                )

            # Compute vectors
            steering = generator.generate_steering_vectors(
                activations=activations, method=method, whiten=whiten
            )

            bundles.append(
                PatternSteeringBundle(
                    pattern_name=pattern_name,
                    pair_type=pair_type,
                    method=method,
                    steering_vector=steering,
                    inputs_used=inputs_used,
                )
            )

    return bundles


def inject_with_interpretation_prompt(
    model,
    tokenizer,
    prompt_text: str,
    steering_vector: SteeringVector,
    interpretation_prompt,
    injection_strength: float = 1.0,
    injection_mode: str = "addition",
    max_new_tokens: int = 30,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Inject a multi-layer steering vector at the placeholder token positions defined by
    InterpretationPrompt. This respects your requirement: capture last-token activations
    for training, but inject at specific token(s) defined by the prompt.
    """
    injector = RepengMultiLayerInjector(model, tokenizer)

    insert_positions = interpretation_prompt.get_insert_locations()
    if not insert_positions:
        # Fallback: inject at the last prompt token if no placeholders
        insert_positions = [len(interpretation_prompt.get_tokenized_inputs()["input_ids"][0]) - 1]

    return injector.inject_steering_vector(
        prompt=prompt_text,
        steering_vector=steering_vector,
        injection_positions=insert_positions,
        injection_strength=injection_strength,
        injection_mode=injection_mode,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )


