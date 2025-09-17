"""
Multi-layer activation extraction using NNsight for steering vector generation.

This module provides functionality to:
1. Extract activations from multiple layers simultaneously using NNsight
2. Process batched inputs efficiently
3. Handle last-token extraction similar to repeng methodology
4. Support different layer selection strategies
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import numpy as np
from tqdm import tqdm
import warnings

try:
    import nnsight
except ImportError:
    raise ImportError("NNsight is required. Install it with: pip install nnsight")

from .repeng_dataset_generator import DatasetEntry
from ..utils import get_model_layers, get_layer_by_path
from ..device_utils import ensure_device_compatibility


class RepengActivationExtractor:
    """
    Extract activations from multiple layers using NNsight for steering vector training.
    
    This class implements the repeng methodology of extracting activations from the last
    non-padding token across multiple layers simultaneously.
    """
    
    def __init__(self, model, tokenizer, layer_indices: Optional[List[int]] = None):
        """
        Initialize the activation extractor.
        
        Args:
            model: NNsight LanguageModel instance
            tokenizer: Associated tokenizer
            layer_indices: List of layer indices to extract from (default: all layers)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_paths = get_model_layers(self.model)
        
        # Set layer indices
        if layer_indices is None:
            # Default: use all layers
            self.layer_indices = list(range(len(self.layer_paths)))
        else:
            self.layer_indices = layer_indices
            
        # Filter out vision components for Gemma 3 4B models
        if self._is_gemma_3_4b():
            self.layer_paths = self._filter_vision_components(self.layer_paths)
            # Update layer indices to match filtered layers
            self.layer_indices = [i for i in self.layer_indices if i < len(self.layer_paths)]
            print(f"Filtered vision components. Using {len(self.layer_paths)} layers.")
        
        print(f"Initialized activation extractor for {len(self.layer_indices)} layers")
    
    def _clear_device_cache(self):
        """Clear device cache in a device-agnostic way"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        # CPU doesn't need explicit cache clearing
    
    def _is_gemma_3_4b(self) -> bool:
        """Check if the loaded model is Gemma 3 4B."""
        if hasattr(self.model, 'model_name'):
            model_name_lower = self.model.model_name.lower()
            return "gemma" in model_name_lower and "3" in model_name_lower and "4b" in model_name_lower
        return False
    
    def _filter_vision_components(self, layer_paths: List[str]) -> List[str]:
        """Filter out vision components from layer paths."""
        return [path for path in layer_paths if 'vision_tower' not in path and 'vision_model' not in path]
    
    def extract_last_token_activations(
        self,
        inputs: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations from the last non-padding token for each input.
        
        This follows repeng's methodology of extracting from the last token position
        to capture the model's final state after processing the entire input.
        
        Args:
            inputs: List of input strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping layer_idx -> tensor of shape (num_inputs, hidden_dim)
        """
        all_activations = {layer_idx: [] for layer_idx in self.layer_indices}
        
        # Process inputs in batches
        batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
        
        for batch in tqdm(batches, disable=not show_progress, desc="Extracting activations"):
            batch_activations = self._extract_batch_activations(batch)
            
            # Accumulate activations
            for layer_idx in self.layer_indices:
                all_activations[layer_idx].extend(batch_activations[layer_idx])
        
        # Convert lists to tensors, skipping empties to avoid RuntimeError
        empty_layers = []
        for layer_idx in self.layer_indices:
            if len(all_activations[layer_idx]) == 0:
                empty_layers.append(layer_idx)
                continue
            all_activations[layer_idx] = torch.stack(all_activations[layer_idx])

        # Remove any layers that produced no activations
        for layer_idx in empty_layers:
            all_activations.pop(layer_idx, None)
            print(f"Warning: No activations collected for layer {layer_idx}; skipping this layer.")

        if not all_activations:
            raise RuntimeError("No activations collected for any layer. Verify layer paths and tracing.")

        print(f"Extracted activations for {len(inputs)} inputs across {len(all_activations)} layers")
        return all_activations
    
    def _extract_batch_activations(self, batch: List[str]) -> Dict[int, List[torch.Tensor]]:
        """Extract activations for a single batch with memory optimization."""
        batch_activations = {layer_idx: [] for layer_idx in self.layer_indices}
        
        # Process each input individually to minimize memory usage
        for input_text in batch:
            # Clear device cache before each input (device-agnostic)
            self._clear_device_cache()
                
            # Initialize layer_outputs for this single input
            layer_outputs = {}
            
            # Use forward pass context to extract activations without generation
            encoded_inputs = self.tokenizer(
                [input_text],  # Single input only
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            attention_mask = encoded_inputs['attention_mask']
            
            with self.model.trace([input_text]) as tracer:
                # Store layer outputs
                for layer_idx in self.layer_indices:
                    try:
                        layer = get_layer_by_path(self.model, self.layer_paths[layer_idx])
                        # Save the hidden states from this layer
                        layer_outputs[layer_idx] = layer.output[0].save()
                    except Exception as e:
                        print(f"Warning: Could not access layer {layer_idx} ({self.layer_paths[layer_idx]}): {e}")
                        continue
            
            # Process this single input's activations immediately
            self._process_single_input_activations(layer_outputs, attention_mask, 0, batch_activations)
            
            # Clear layer outputs to free memory
            del layer_outputs
            
        return batch_activations
    
    def _process_single_input_activations(self, layer_outputs, attention_mask, input_idx, batch_activations):
        """Process activations for a single input immediately."""
        
        # Check if we got any layer outputs
        if not layer_outputs:
            raise RuntimeError("Failed to extract any layer outputs. Check layer paths and model compatibility.")
        
        # Find last non-padding token position for this single input
        last_non_padding_idx = self._find_last_non_padding_token(attention_mask[input_idx])
        
        for layer_idx in self.layer_indices:
            if layer_idx not in layer_outputs:
                continue
                
            # Extract activation at last token position
            try:
                activation = layer_outputs[layer_idx][input_idx, last_non_padding_idx, :]
                # Ensure proper device placement
                activation = ensure_device_compatibility(activation, self.model.device)
                batch_activations[layer_idx].append(activation)
            except Exception as e:
                print(f"Warning: Could not extract activation for layer {layer_idx}, sample {input_idx}: {e}")
                continue
    
    def _find_last_non_padding_token(self, attention_mask: torch.Tensor) -> int:
        """Find the index of the last non-padding token."""
        non_padding_indices = attention_mask.nonzero(as_tuple=True)[0]
        if len(non_padding_indices) == 0:
            return 0  # Fallback to first token if all are padding
        return non_padding_indices[-1].item()
    
    def extract_dataset_activations(
        self,
        dataset: List[DatasetEntry],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Tuple[Dict[int, torch.Tensor], List[str]]:
        """
        Extract activations for a dataset in repeng format.
        
        This creates the alternating positive/negative structure used by repeng:
        [positive, negative, positive, negative, ...]
        
        Args:
            dataset: List of DatasetEntry objects
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (activations_dict, input_texts_list)
            activations_dict: Maps layer_idx -> tensor of shape (2*num_entries, hidden_dim)
            input_texts_list: List of input texts in alternating pos/neg order
        """
        # Create alternating positive/negative input list
        alternating_inputs = []
        for entry in dataset:
            alternating_inputs.append(entry.positive)
            alternating_inputs.append(entry.negative)
        
        # Extract activations
        activations = self.extract_last_token_activations(
            alternating_inputs, batch_size, show_progress
        )
        
        print(f"Extracted activations for {len(dataset)} dataset entries (2x examples)")
        return activations, alternating_inputs
    
    def extract_layer_range(
        self,
        inputs: List[str],
        start_layer: int = -5,
        end_layer: int = -18,
        batch_size: int = 32
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations from a specific range of layers (repeng common pattern).
        
        Args:
            inputs: List of input strings
            start_layer: Starting layer (negative indexing supported)
            end_layer: Ending layer (negative indexing supported) 
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping layer_idx -> activations tensor
        """
        # Convert negative indices to positive
        total_layers = len(self.layer_paths)
        if start_layer < 0:
            start_layer = total_layers + start_layer
        if end_layer < 0:
            end_layer = total_layers + end_layer
        
        # Create range (inclusive of start, exclusive of end, step by -1 if descending)
        if start_layer > end_layer:
            layer_range = list(range(start_layer, end_layer - 1, -1))
        else:
            layer_range = list(range(start_layer, end_layer + 1))
        
        # Filter to valid indices
        layer_range = [idx for idx in layer_range if 0 <= idx < total_layers]
        
        # Temporarily set layer indices
        original_indices = self.layer_indices
        self.layer_indices = layer_range
        
        try:
            activations = self.extract_last_token_activations(inputs, batch_size)
        finally:
            # Restore original indices
            self.layer_indices = original_indices
        
        print(f"Extracted activations from layer range {layer_range}")
        return activations
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about available layers."""
        return {
            "total_layers": len(self.layer_paths),
            "selected_layers": self.layer_indices,
            "layer_paths": [self.layer_paths[i] for i in self.layer_indices],
            "model_type": type(self.model).__name__
        }
    
    def save_activations(self, activations: Dict[int, torch.Tensor], filepath: str):
        """Save extracted activations to a file."""
        # Convert to numpy for saving
        np_activations = {
            layer_idx: tensor.cpu().numpy() 
            for layer_idx, tensor in activations.items()
        }
        
        np.savez_compressed(filepath, **{f"layer_{layer_idx}": arr for layer_idx, arr in np_activations.items()})
        print(f"Saved activations to {filepath}")
    
    def load_activations(self, filepath: str) -> Dict[int, torch.Tensor]:
        """Load activations from a file."""
        data = np.load(filepath)
        
        activations = {}
        for key in data.files:
            if key.startswith("layer_"):
                layer_idx = int(key.split("_")[1])
                tensor = torch.from_numpy(data[key])
                # Move to model device
                tensor = ensure_device_compatibility(tensor, self.model.device)
                activations[layer_idx] = tensor
        
        print(f"Loaded activations for {len(activations)} layers from {filepath}")
        return activations


def extract_repeng_activations(
    model,
    tokenizer,
    dataset: List[DatasetEntry],
    layer_range: Optional[Tuple[int, int]] = None,
    batch_size: int = 32
) -> Tuple[Dict[int, torch.Tensor], List[str]]:
    """
    Convenience function to extract activations in repeng format.
    
    Args:
        model: NNsight LanguageModel
        tokenizer: Tokenizer
        dataset: List of DatasetEntry objects
        layer_range: Optional (start, end) layer range (e.g., (-5, -18))
        batch_size: Batch size
        
    Returns:
        Tuple of (activations_dict, input_texts_list)
    """
    extractor = RepengActivationExtractor(model, tokenizer)
    
    if layer_range is not None:
        start, end = layer_range
        # Extract only from specified range
        alternating_inputs = []
        for entry in dataset:
            alternating_inputs.append(entry.positive)
            alternating_inputs.append(entry.negative)
        
        activations = extractor.extract_layer_range(
            alternating_inputs, start, end, batch_size
        )
        return activations, alternating_inputs
    else:
        # Extract from all layers
        return extractor.extract_dataset_activations(dataset, batch_size)