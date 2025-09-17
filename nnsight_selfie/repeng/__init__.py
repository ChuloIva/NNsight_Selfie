"""
Repeng-style steering vector utilities for neural network manipulation.

This module provides tools for:
1. Creating datasets for steering vector generation
2. Extracting activations across multiple layers
3. Generating steering vectors using PCA and other methods
4. Injecting steering vectors at multiple layers simultaneously
5. Interpreting steering vectors using selfie-style prompts

The repeng approach enables fine-grained control over model behavior by
manipulating internal representations across multiple transformer layers.
"""

# Core repeng functionality
from .repeng_dataset_generator import RepengDatasetGenerator, DatasetEntry, create_quick_dataset
from .repeng_activation_extractor import RepengActivationExtractor, extract_repeng_activations
from .repeng_steering_vectors import RepengSteeringVectorGenerator, SteeringVector, create_steering_vector
from .repeng_multi_injection import RepengMultiLayerInjector, inject_multi_layer

# Combined interpretation utilities
from .repeng_interpretation_utils import (
    RepengInterpretationAnalyzer, 
    SteeringInterpretationResult,
    create_interpretation_comparison_plot,
    quick_steering_interpretation_analysis
)

__version__ = "0.1.0"
__all__ = [
    # Core repeng functionality
    "RepengDatasetGenerator",
    "DatasetEntry", 
    "create_quick_dataset",
    "RepengActivationExtractor",
    "extract_repeng_activations",
    "RepengSteeringVectorGenerator",
    "SteeringVector",
    "create_steering_vector",
    "RepengMultiLayerInjector",
    "inject_multi_layer",
    # Combined interpretation utilities
    "RepengInterpretationAnalyzer",
    "SteeringInterpretationResult",
    "create_interpretation_comparison_plot",
    "quick_steering_interpretation_analysis"
]