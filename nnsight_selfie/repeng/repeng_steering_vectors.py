"""
PCA-based steering vector generation following repeng methodology.

This module provides functionality to:
1. Apply PCA to extracted activations to find principal directions
2. Generate steering vectors for multiple layers
3. Support different averaging methods (pca_diff, pca_center)
4. Automatic sign correction for consistent positive/negative direction
5. Save/load steering vectors for reuse
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import numpy as np
import dataclasses
from sklearn.decomposition import PCA
import warnings
import pickle
from pathlib import Path

try:
    import nnsight
except ImportError:
    raise ImportError("NNsight is required. Install it with: pip install nnsight")

from .repeng_dataset_generator import DatasetEntry


@dataclasses.dataclass
class SteeringVector:
    """
    Container for a multi-layer steering vector.
    
    Attributes:
        model_type: Type/name of the model this vector is for
        directions: Dictionary mapping layer_idx -> direction vector
        metadata: Additional information about how the vector was created
    """
    model_type: str
    directions: Dict[int, np.ndarray]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RepengSteeringVectorGenerator:
    """
    Generate steering vectors using repeng methodology with PCA.
    
    This class implements the three main approaches from repeng:
    1. pca_diff: PCA on differences between positive and negative activations
    2. pca_center: PCA on centered activations around the positive/negative mean
    3. Basic averaging: Simple mean difference between positive and negative
    """
    
    def __init__(self, model_type: str = "unknown"):
        """
        Initialize the steering vector generator.
        
        Args:
            model_type: Identifier for the model type
        """
        self.model_type = model_type
    
    def generate_steering_vectors(
        self,
        activations: Dict[int, torch.Tensor],
        method: str = "pca_diff",
        whiten: bool = False
    ) -> SteeringVector:
        """
        Generate steering vectors from extracted activations.
        
        Args:
            activations: Dictionary mapping layer_idx -> activations tensor
                        Shape: (2*n_examples, hidden_dim) in alternating pos/neg order
            method: Method to use ("pca_diff", "pca_center", "mean_diff")
            whiten: Whether to apply whitening in PCA
            
        Returns:
            SteeringVector object containing directions for each layer
        """
        if method not in ["pca_diff", "pca_center", "mean_diff"]:
            raise ValueError(f"Unknown method: {method}. Choose from 'pca_diff', 'pca_center', 'mean_diff'")
        
        directions = {}
        
        for layer_idx, layer_activations in activations.items():
            # Convert to numpy for processing (bfloat16 -> float32 -> numpy)
            h = layer_activations.cpu().float().detach().numpy()
            
            if method == "pca_diff":
                direction = self._generate_pca_diff_vector(h, whiten)
            elif method == "pca_center":
                direction = self._generate_pca_center_vector(h, whiten)
            elif method == "mean_diff":
                direction = self._generate_mean_diff_vector(h)
            
            # Apply sign correction to ensure consistent positive/negative direction
            direction = self._apply_sign_correction(h, direction)
            
            directions[layer_idx] = direction
        
        steering_vector = SteeringVector(
            model_type=self.model_type,
            directions=directions,
            metadata={
                "method": method,
                "whiten": whiten,
                "num_examples": len(activations[list(activations.keys())[0]]) // 2,
                "layers": list(activations.keys())
            }
        )
        
        print(f"Generated steering vectors for {len(directions)} layers using {method}")
        return steering_vector
    
    def _generate_pca_diff_vector(self, h: np.ndarray, whiten: bool) -> np.ndarray:
        """
        Generate steering vector using PCA on differences between pos/neg activations.
        
        This is repeng's main method - finds the principal component of the 
        difference vectors between positive and negative examples.
        """
        # Extract positive and negative examples (alternating pattern)
        positive_activations = h[::2]  # Even indices
        negative_activations = h[1::2]  # Odd indices
        
        # Compute differences: positive - negative
        differences = positive_activations - negative_activations
        
        # Apply PCA to find principal direction
        pca = PCA(n_components=1, whiten=whiten)
        pca.fit(differences)
        
        direction = pca.components_[0].astype(np.float32)
        return direction
    
    def _generate_pca_center_vector(self, h: np.ndarray, whiten: bool) -> np.ndarray:
        """
        Generate steering vector using PCA on centered activations.
        
        Centers both positive and negative activations around their mean,
        then applies PCA to the combined centered data.
        """
        # Extract positive and negative examples
        positive_activations = h[::2]
        negative_activations = h[1::2]
        
        # Compute center point
        center = (positive_activations + negative_activations) / 2
        
        # Center the activations
        centered_positive = positive_activations - center
        centered_negative = negative_activations - center
        
        # Combine centered activations
        centered_data = np.concatenate([centered_positive, centered_negative], axis=0)
        
        # Apply PCA
        pca = PCA(n_components=1, whiten=whiten)
        pca.fit(centered_data)
        
        direction = pca.components_[0].astype(np.float32)
        return direction
    
    def _generate_mean_diff_vector(self, h: np.ndarray) -> np.ndarray:
        """
        Generate steering vector using simple mean difference.
        
        Computes the mean of positive examples minus mean of negative examples.
        This is the simplest approach and serves as a baseline.
        """
        # Extract positive and negative examples
        positive_activations = h[::2]
        negative_activations = h[1::2]
        
        # Compute mean difference
        positive_mean = np.mean(positive_activations, axis=0)
        negative_mean = np.mean(negative_activations, axis=0)
        
        direction = (positive_mean - negative_mean).astype(np.float32)
        
        # Normalize
        direction = direction / np.linalg.norm(direction)
        
        return direction
    
    def _apply_sign_correction(self, h: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        Apply sign correction to ensure positive examples project higher than negative.
        
        This follows repeng's methodology of checking the sign and flipping if needed.
        """
        # Project activations onto the direction
        projected = np.dot(h, direction)
        
        # Check if positive examples (even indices) have higher projections on average
        positive_projections = projected[::2]
        negative_projections = projected[1::2]
        
        positive_mean_projection = np.mean(positive_projections)
        negative_mean_projection = np.mean(negative_projections)
        
        # If negative projections are higher, flip the direction
        if negative_mean_projection > positive_mean_projection:
            direction = -direction
        
        return direction
    
    def evaluate_steering_vector(
        self,
        steering_vector: SteeringVector,
        activations: Dict[int, torch.Tensor],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the quality of a steering vector.
        
        Args:
            steering_vector: The steering vector to evaluate
            activations: Original activations used to create the vector
            verbose: Whether to print evaluation results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        metrics = {}
        
        for layer_idx, direction in steering_vector.directions.items():
            if layer_idx not in activations:
                continue
                
            h = activations[layer_idx].cpu().float().numpy()
            
            # Project activations onto direction
            projected = np.dot(h, direction)
            
            positive_projections = projected[::2]
            negative_projections = projected[1::2]
            
            # Compute separation metrics
            pos_mean = np.mean(positive_projections)
            neg_mean = np.mean(negative_projections)
            pos_std = np.std(positive_projections)
            neg_std = np.std(negative_projections)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
            cohens_d = (pos_mean - neg_mean) / pooled_std if pooled_std > 0 else 0
            
            # Accuracy: how often positive > negative
            accuracy = np.mean(positive_projections > negative_projections)
            
            metrics[f"layer_{layer_idx}"] = {
                "cohens_d": cohens_d,
                "accuracy": accuracy,
                "pos_mean": pos_mean,
                "neg_mean": neg_mean,
                "separation": pos_mean - neg_mean
            }
        
        # Overall metrics
        overall_accuracy = np.mean([m["accuracy"] for m in metrics.values()])
        overall_cohens_d = np.mean([m["cohens_d"] for m in metrics.values()])
        
        metrics["overall"] = {
            "accuracy": overall_accuracy,
            "cohens_d": overall_cohens_d
        }
        
        if verbose:
            print(f"Steering Vector Evaluation:")
            print(f"  Overall Accuracy: {overall_accuracy:.3f}")
            print(f"  Overall Cohen's d: {overall_cohens_d:.3f}")
        
        return metrics
    
    def save_steering_vector(self, steering_vector: SteeringVector, filepath: str):
        """Save steering vector to a pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(steering_vector, f)
        print(f"Saved steering vector to {filepath}")
    
    def load_steering_vector(self, filepath: str) -> SteeringVector:
        """Load steering vector from a pickle file."""
        with open(filepath, 'rb') as f:
            steering_vector = pickle.load(f)
        print(f"Loaded steering vector from {filepath}")
        return steering_vector
    
    def combine_steering_vectors(
        self,
        vectors: List[SteeringVector],
        weights: Optional[List[float]] = None
    ) -> SteeringVector:
        """
        Combine multiple steering vectors with optional weights.
        
        Args:
            vectors: List of SteeringVector objects
            weights: Optional weights for each vector (default: equal weights)
            
        Returns:
            Combined SteeringVector
        """
        if not vectors:
            raise ValueError("No vectors provided")
        
        if weights is None:
            weights = [1.0 / len(vectors)] * len(vectors)
        
        if len(weights) != len(vectors):
            raise ValueError("Number of weights must match number of vectors")
        
        # Find common layers
        common_layers = set(vectors[0].directions.keys())
        for vector in vectors[1:]:
            common_layers &= set(vector.directions.keys())
        
        if not common_layers:
            raise ValueError("No common layers found across all vectors")
        
        # Combine directions
        combined_directions = {}
        for layer_idx in common_layers:
            combined_direction = np.zeros_like(vectors[0].directions[layer_idx])
            
            for vector, weight in zip(vectors, weights):
                combined_direction += weight * vector.directions[layer_idx]
            
            # Normalize
            combined_direction = combined_direction / np.linalg.norm(combined_direction)
            combined_directions[layer_idx] = combined_direction
        
        combined_vector = SteeringVector(
            model_type=vectors[0].model_type,
            directions=combined_directions,
            metadata={
                "method": "combined",
                "source_vectors": len(vectors),
                "weights": weights,
                "layers": list(common_layers)
            }
        )
        
        return combined_vector


def create_steering_vector(
    activations: Dict[int, torch.Tensor],
    method: str = "pca_diff",
    model_type: str = "unknown",
    whiten: bool = False
) -> SteeringVector:
    """
    Convenience function to create a steering vector.
    
    Args:
        activations: Dictionary mapping layer_idx -> activations tensor
        method: Method to use for generation
        model_type: Model type identifier
        whiten: Whether to apply whitening in PCA
        
    Returns:
        SteeringVector object
    """
    generator = RepengSteeringVectorGenerator(model_type)
    return generator.generate_steering_vectors(activations, method, whiten)


def evaluate_steering_separation(
    activations: Dict[int, torch.Tensor],
    steering_vector: SteeringVector,
    layer_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get positive and negative projections for a specific layer.
    
    Args:
        activations: Original activations
        steering_vector: Steering vector to project onto
        layer_idx: Layer index to analyze
        
    Returns:
        Tuple of (positive_projections, negative_projections)
    """
    if layer_idx not in activations or layer_idx not in steering_vector.directions:
        raise ValueError(f"Layer {layer_idx} not found in activations or steering vector")
    
    h = activations[layer_idx].cpu().float().numpy()
    direction = steering_vector.directions[layer_idx]
    
    projected = np.dot(h, direction)
    positive_projections = projected[::2]
    negative_projections = projected[1::2]
    
    return positive_projections, negative_projections