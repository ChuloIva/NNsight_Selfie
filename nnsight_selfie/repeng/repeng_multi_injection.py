"""
Multi-layer token-level injection mechanism using NNsight.

This module provides functionality to:
1. Inject steering vectors across multiple layers simultaneously at specific token positions
2. Support different injection modes (addition, normalized, replacement)
3. Handle batch processing for efficient inference
4. Compare outputs with/without injection for analysis
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import torch
import numpy as np
from tqdm import tqdm
import warnings

try:
    import nnsight
except ImportError:
    raise ImportError("NNsight is required. Install it with: pip install nnsight")

from .repeng_steering_vectors import SteeringVector
from ..utils import get_model_layers, get_layer_by_path
from ..device_utils import ensure_device_compatibility


class RepengMultiLayerInjector:
    """
    Inject steering vectors across multiple layers at token level using NNsight.
    
    This class enables simultaneous injection of different steering directions
    across multiple layers, allowing for more sophisticated control over model behavior.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize the multi-layer injector.
        
        Args:
            model: NNsight LanguageModel instance
            tokenizer: Associated tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_paths = get_model_layers(self.model)
        
        # Filter out vision components for Gemma 3 4B models
        if self._is_gemma_3_4b():
            self.layer_paths = self._filter_vision_components(self.layer_paths)
            print(f"Filtered vision components. Using {len(self.layer_paths)} layers.")
    
    def _is_gemma_3_4b(self) -> bool:
        """Check if the loaded model is Gemma 3 4B."""
        if hasattr(self.model, 'model_name'):
            model_name_lower = self.model.model_name.lower()
            return "gemma" in model_name_lower and "3" in model_name_lower and "4b" in model_name_lower
        return False
    
    def _filter_vision_components(self, layer_paths: List[str]) -> List[str]:
        """Filter out vision components from layer paths."""
        return [path for path in layer_paths if 'vision_tower' not in path and 'vision_model' not in path]
    
    def inject_steering_vector(
        self,
        prompt: str,
        steering_vector: SteeringVector,
        injection_positions: Union[List[int], int],
        injection_strength: float = 1.0,
        injection_mode: str = "addition",
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Inject steering vector across multiple layers at specified token positions.
        
        Args:
            prompt: Input prompt
            steering_vector: SteeringVector containing directions for each layer
            injection_positions: Token position(s) to inject at (list or single int)
            injection_strength: Strength multiplier for the steering vectors
            injection_mode: How to apply injection ("addition", "normalized", "replacement")
            max_new_tokens: Maximum new tokens to generate
            do_sample: Whether to use sampling for generation
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary containing generation results and metadata
        """
        if isinstance(injection_positions, int):
            injection_positions = [injection_positions]
        
        # Tokenize prompt to get sequence length
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
        prompt_length = prompt_tokens['input_ids'].shape[1]
        
        # Validate injection positions
        valid_positions = [pos for pos in injection_positions if 0 <= pos < prompt_length]
        if len(valid_positions) != len(injection_positions):
            warnings.warn(f"Some injection positions are out of range. Using {valid_positions}")
            injection_positions = valid_positions
        
        if not injection_positions:
            raise ValueError("No valid injection positions")
        
        # Generate with steering injection
        output_ids = self._generate_with_injection(
            prompt,
            steering_vector,
            injection_positions,
            injection_strength,
            injection_mode,
            max_new_tokens,
            do_sample,
            temperature,
            top_p
        )
        
        # Decode results
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = self.tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
        
        return {
            "full_text": full_text,
            "generated_text": generated_text,
            "prompt": prompt,
            "output_ids": output_ids,
            "metadata": {
                "injection_positions": injection_positions,
                "injection_strength": injection_strength,
                "injection_mode": injection_mode,
                "steering_layers": list(steering_vector.directions.keys()),
                "generation_params": {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
        }
    
    def _generate_with_injection(
        self,
        prompt: str,
        steering_vector: SteeringVector,
        injection_positions: List[int],
        injection_strength: float,
        injection_mode: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float
    ) -> torch.Tensor:
        """Internal method to perform generation with multi-layer injection."""
        
        # Prepare steering directions as tensors
        steering_directions = {}
        for layer_idx, direction in steering_vector.directions.items():
            direction_tensor = torch.from_numpy(direction).to(self.model.device)
            steering_directions[layer_idx] = direction_tensor
        
        # Generate with intervention
        with self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id
        ) as tracer:
            
            # Apply interventions to all specified layers
            for layer_idx in steering_directions:
                if layer_idx >= len(self.layer_paths):
                    warnings.warn(f"Layer {layer_idx} out of range, skipping")
                    continue
                
                layer = get_layer_by_path(self.model, self.layer_paths[layer_idx])
                original_output = layer.output[0]
                
                # Apply injection at specified positions
                for pos in injection_positions:
                    self._apply_injection_at_position(
                        original_output,
                        steering_directions[layer_idx],
                        pos,
                        injection_strength,
                        injection_mode
                    )
            
            # Get the generated output
            output_ids = self.model.generator.output.save()
        
        return output_ids
    
    def _apply_injection_at_position(
        self,
        layer_output: torch.Tensor,
        direction: torch.Tensor,
        position: int,
        strength: float,
        mode: str
    ):
        """Apply injection at a specific token position."""
        batch_size, seq_len, hidden_dim = layer_output.shape
        
        if position >= seq_len:
            return  # Position out of range
        
        # Ensure direction is properly shaped and on correct device
        if direction.dim() == 1:
            direction = direction.unsqueeze(0).expand(batch_size, -1)
        elif direction.dim() == 2 and direction.shape[0] == 1:
            direction = direction.expand(batch_size, -1)
        
        direction = ensure_device_compatibility(direction, self.model.device)
        
        if mode == "addition":
            layer_output[:, position, :] += strength * direction
            
        elif mode == "normalized":
            # Store original norm
            original_norm = torch.norm(layer_output[:, position, :], dim=-1, keepdim=True)
            
            # Apply intervention
            modified = layer_output[:, position, :] + strength * direction
            
            # Restore original magnitude
            new_norm = torch.norm(modified, dim=-1, keepdim=True)
            layer_output[:, position, :] = modified * (original_norm / new_norm)
            
        elif mode == "replacement":
            # Replace with scaled direction
            layer_output[:, position, :] = strength * direction
            
        else:
            raise ValueError(f"Unknown injection mode: {mode}")
    
    def compare_with_without_injection(
        self,
        prompt: str,
        steering_vector: SteeringVector,
        injection_positions: Union[List[int], int],
        injection_strength: float = 1.0,
        injection_mode: str = "addition",
        max_new_tokens: int = 50,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate text with and without injection for comparison.
        
        Args:
            prompt: Input prompt
            steering_vector: SteeringVector to apply
            injection_positions: Position(s) to inject at
            injection_strength: Injection strength
            injection_mode: Injection mode
            max_new_tokens: Max tokens to generate
            generation_params: Optional generation parameters
            
        Returns:
            Dictionary with both generations and comparison metrics
        """
        if generation_params is None:
            generation_params = {"do_sample": True, "temperature": 0.7, "top_p": 0.9}
        
        # Generate without injection (baseline)
        with self.model.generate(prompt, max_new_tokens=max_new_tokens, **generation_params) as tracer:
            baseline_output = self.model.generator.output.save()
        
        # Generate with injection
        injection_result = self.inject_steering_vector(
            prompt=prompt,
            steering_vector=steering_vector,
            injection_positions=injection_positions,
            injection_strength=injection_strength,
            injection_mode=injection_mode,
            max_new_tokens=max_new_tokens,
            **generation_params
        )
        
        # Decode baseline
        prompt_length = len(self.tokenizer(prompt, return_tensors="pt")['input_ids'][0])
        baseline_full = self.tokenizer.decode(baseline_output[0], skip_special_tokens=True)
        baseline_generated = self.tokenizer.decode(baseline_output[0][prompt_length:], skip_special_tokens=True)
        
        return {
            "baseline": {
                "full_text": baseline_full,
                "generated_text": baseline_generated,
                "output_ids": baseline_output
            },
            "injection": injection_result,
            "comparison": {
                "prompt": prompt,
                "baseline_length": len(baseline_generated.split()),
                "injection_length": len(injection_result["generated_text"].split()),
                "tokens_different": torch.sum(baseline_output != injection_result["output_ids"]).item()
            }
        }
    
    def batch_inject(
        self,
        prompts: List[str],
        steering_vector: SteeringVector,
        injection_positions: Union[List[int], int],
        injection_strength: float = 1.0,
        batch_size: int = 4,
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Apply injection to a batch of prompts.
        
        Args:
            prompts: List of input prompts
            steering_vector: SteeringVector to apply
            injection_positions: Position(s) to inject at
            injection_strength: Injection strength
            batch_size: Processing batch size
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generation results
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
            batch_prompts = prompts[i:i + batch_size]
            
            for prompt in batch_prompts:
                result = self.inject_steering_vector(
                    prompt=prompt,
                    steering_vector=steering_vector,
                    injection_positions=injection_positions,
                    injection_strength=injection_strength,
                    **generation_kwargs
                )
                results.append(result)
        
        return results
    
    def analyze_injection_effects(
        self,
        prompt: str,
        steering_vector: SteeringVector,
        position_range: Tuple[int, int],
        strength_range: Tuple[float, float, int] = (0.1, 2.0, 10),
        max_new_tokens: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze effects of injection at different positions and strengths.
        
        Args:
            prompt: Input prompt
            steering_vector: SteeringVector to analyze
            position_range: (start_pos, end_pos) range of positions to test
            strength_range: (min_strength, max_strength, num_steps) for strength sweep
            max_new_tokens: Max tokens to generate
            
        Returns:
            Dictionary containing analysis results
        """
        start_pos, end_pos = position_range
        min_strength, max_strength, num_steps = strength_range
        
        strengths = np.linspace(min_strength, max_strength, num_steps)
        positions = list(range(start_pos, end_pos + 1))
        
        results = {
            "positions": positions,
            "strengths": strengths.tolist(),
            "generations": {},
            "prompt": prompt
        }
        
        for pos in positions:
            for strength in strengths:
                key = f"pos_{pos}_strength_{strength:.2f}"
                
                try:
                    result = self.inject_steering_vector(
                        prompt=prompt,
                        steering_vector=steering_vector,
                        injection_positions=[pos],
                        injection_strength=strength,
                        max_new_tokens=max_new_tokens,
                        do_sample=False  # Use deterministic generation for analysis
                    )
                    results["generations"][key] = result["generated_text"]
                    
                except Exception as e:
                    results["generations"][key] = f"Error: {str(e)}"
        
        return results


def inject_multi_layer(
    model,
    tokenizer,
    prompt: str,
    steering_vector: SteeringVector,
    injection_positions: Union[List[int], int],
    injection_strength: float = 1.0,
    injection_mode: str = "addition",
    max_new_tokens: int = 50
) -> str:
    """
    Convenience function for multi-layer injection.
    
    Args:
        model: NNsight model
        tokenizer: Tokenizer
        prompt: Input prompt
        steering_vector: Steering vector to apply
        injection_positions: Position(s) to inject at
        injection_strength: Injection strength
        injection_mode: Injection mode
        max_new_tokens: Max new tokens
        
    Returns:
        Generated text with injection applied
    """
    injector = RepengMultiLayerInjector(model, tokenizer)
    result = injector.inject_steering_vector(
        prompt=prompt,
        steering_vector=steering_vector,
        injection_positions=injection_positions,
        injection_strength=injection_strength,
        injection_mode=injection_mode,
        max_new_tokens=max_new_tokens
    )
    return result["generated_text"]