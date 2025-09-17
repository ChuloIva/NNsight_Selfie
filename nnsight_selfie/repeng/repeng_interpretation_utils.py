"""
Utility functions for combining Repeng-style steering with interpretation workflows.

This module provides helper functions that bridge the gap between multi-layer steering
and interpretation capabilities, enabling analysis of what happens when steering vectors
are injected across multiple layers.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import pandas as pd

from ..model_agnostic_selfie import ModelAgnosticSelfie, InterpretationPrompt
from .repeng_multi_injection import RepengMultiLayerInjector
from .repeng_steering_vectors import SteeringVector


@dataclass
class SteeringInterpretationResult:
    """Results from steering + interpretation analysis."""
    prompt: str
    steering_strength: float
    baseline_text: str
    steered_text: str
    baseline_interpretations: Dict[int, str]
    steered_interpretations: Dict[int, str]
    layer_effects: Dict[int, Dict[str, float]]
    

class RepengInterpretationAnalyzer:
    """
    Combines Repeng-style multi-layer steering with interpretation analysis.
    
    This class provides methods to:
    1. Apply steering vectors across multiple layers
    2. Interpret the resulting activations
    3. Compare before/after interpretations
    4. Analyze layer-wise effects
    """
    
    def __init__(self, steering_model, interpretation_model, tokenizer):
        """
        Initialize the analyzer with both steering and interpretation models.
        
        Args:
            steering_model: NNsight LanguageModel for steering operations
            interpretation_model: ModelAgnosticSelfie for interpretation
            tokenizer: Tokenizer shared by both models
        """
        self.steering_model = steering_model
        self.interpretation_model = interpretation_model
        self.tokenizer = tokenizer
        self.injector = RepengMultiLayerInjector(steering_model, tokenizer)
        
    def create_interpretation_prompts(self) -> Dict[str, InterpretationPrompt]:
        """Create a set of useful interpretation prompts."""
        prompts = {
            'emotion': InterpretationPrompt(
                self.tokenizer,
                ["This neural pattern represents the emotion of ", None]
            ),
            'sentiment': InterpretationPrompt(
                self.tokenizer,
                ["The sentiment expressed here is ", None]
            ),
            'affect': InterpretationPrompt(
                self.tokenizer,
                ["The emotional tone captured in this activation is ", None]
            ),
            'concept': InterpretationPrompt.create_concept_prompt(self.tokenizer),
            'valence': InterpretationPrompt(
                self.tokenizer,
                ["This represents a ", None, " emotional state"]
            )
        }
        return prompts
        
    def analyze_steering_effects(
        self, 
        prompt: str,
        steering_vector: SteeringVector,
        strengths: List[float] = [1.0, -1.0, 2.0],
        interpretation_layers: Optional[List[int]] = None,
        interpretation_prompt: Optional[InterpretationPrompt] = None,
        max_new_tokens: int = 15
    ) -> List[SteeringInterpretationResult]:
        """
        Analyze steering effects with interpretation at multiple strengths.
        
        Args:
            prompt: Input prompt to analyze
            steering_vector: Multi-layer steering vector
            strengths: List of steering strengths to test
            interpretation_layers: Layers to interpret (default: vector layers)
            interpretation_prompt: Prompt for interpretation (default: emotion)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of SteeringInterpretationResult objects
        """
        if interpretation_prompt is None:
            interpretation_prompt = self.create_interpretation_prompts()['emotion']
            
        if interpretation_layers is None:
            interpretation_layers = list(steering_vector.directions.keys())[:5]  # First 5
            
        results = []
        tokens = self.tokenizer.tokenize(prompt)
        injection_pos = len(tokens) - 1
        
        for strength in strengths:
            # Generate with steering
            comparison = self.injector.compare_with_without_injection(
                prompt=prompt,
                steering_vector=steering_vector,
                injection_positions=[injection_pos],
                injection_strength=strength,
                injection_mode="addition",
                max_new_tokens=max_new_tokens
            )
            
            baseline_text = comparison['baseline']['generated_text'].strip()
            steered_text = comparison['injection']['generated_text'].strip()
            
            # Get baseline interpretations
            baseline_interps = self._get_interpretations(
                prompt, interpretation_layers, interpretation_prompt
            )
            
            # Analyze steering vector directions
            steered_interps = self._interpret_steering_directions(
                steering_vector, interpretation_layers, interpretation_prompt, strength
            )
            
            # Calculate layer effects
            layer_effects = self._calculate_layer_effects(steering_vector, strength)
            
            result = SteeringInterpretationResult(
                prompt=prompt,
                steering_strength=strength,
                baseline_text=baseline_text,
                steered_text=steered_text,
                baseline_interpretations=baseline_interps,
                steered_interpretations=steered_interps,
                layer_effects=layer_effects
            )
            
            results.append(result)
            
        return results
        
    def _get_interpretations(
        self, 
        prompt: str, 
        layers: List[int], 
        interpretation_prompt: InterpretationPrompt
    ) -> Dict[int, str]:
        """Get baseline interpretations for specified layers."""
        interpretations = {}
        tokens = self.tokenizer.tokenize(prompt)
        last_token_pos = len(tokens) - 1
        
        for layer in layers:
            try:
                result = self.interpretation_model.interpret(
                    original_prompt=prompt,
                    interpretation_prompt=interpretation_prompt,
                    tokens_to_interpret=[(layer, last_token_pos)],
                    max_new_tokens=8
                )
                interpretations[layer] = result['interpretation'][0].strip()
            except Exception as e:
                interpretations[layer] = f"Error: {str(e)[:30]}..."
                
        return interpretations
        
    def _interpret_steering_directions(
        self,
        steering_vector: SteeringVector,
        layers: List[int],
        interpretation_prompt: InterpretationPrompt,
        strength: float
    ) -> Dict[int, str]:
        """Interpret what the steering directions represent."""
        interpretations = {}
        
        for layer in layers:
            if layer not in steering_vector.directions:
                interpretations[layer] = "No steering vector"
                continue
                
            direction = steering_vector.directions[layer] * strength
            
            try:
                interp = self.interpretation_model.interpret_vectors(
                    vectors=[direction],
                    interpretation_prompt=interpretation_prompt,
                    injection_layer=max(0, layer - 3),
                    max_new_tokens=8
                )[0].strip()
                interpretations[layer] = interp
            except Exception as e:
                interpretations[layer] = f"Error: {str(e)[:30]}..."
                
        return interpretations
        
    def _calculate_layer_effects(
        self, 
        steering_vector: SteeringVector, 
        strength: float
    ) -> Dict[int, Dict[str, float]]:
        """Calculate statistical effects of steering at each layer."""
        effects = {}
        
        for layer, direction in steering_vector.directions.items():
            scaled_direction = direction * strength
            
            effects[layer] = {
                'magnitude': float(torch.norm(scaled_direction)),
                'mean': float(scaled_direction.mean()),
                'std': float(scaled_direction.std()),
                'max': float(scaled_direction.max()),
                'min': float(scaled_direction.min()),
                'strength_factor': abs(strength)
            }
            
        return effects
        
    def create_analysis_dataframe(
        self, 
        results: List[SteeringInterpretationResult]
    ) -> pd.DataFrame:
        """Create a pandas DataFrame summarizing the analysis results."""
        rows = []
        
        for result in results:
            # Basic info
            base_row = {
                'prompt': result.prompt,
                'strength': result.steering_strength,
                'baseline_text': result.baseline_text,
                'steered_text': result.steered_text,
                'text_changed': result.baseline_text != result.steered_text
            }
            
            # Add layer-specific information
            for layer in result.layer_effects.keys():
                row = base_row.copy()
                row.update({
                    'layer': layer,
                    'magnitude': result.layer_effects[layer]['magnitude'],
                    'mean': result.layer_effects[layer]['mean'],
                    'baseline_interp': result.baseline_interpretations.get(layer, ''),
                    'steered_interp': result.steered_interpretations.get(layer, '')
                })
                rows.append(row)
                
        return pd.DataFrame(rows)
        
    def compare_interpretations(
        self, 
        results: List[SteeringInterpretationResult]
    ) -> Dict[str, Any]:
        """Compare interpretations across different steering strengths."""
        comparison = {
            'strength_effects': {},
            'layer_consistency': {},
            'interpretation_changes': []
        }
        
        # Analyze effects by strength
        for result in results:
            strength = result.steering_strength
            comparison['strength_effects'][strength] = {
                'text_similarity': self._text_similarity(
                    result.baseline_text, result.steered_text
                ),
                'avg_magnitude': np.mean([
                    effects['magnitude'] 
                    for effects in result.layer_effects.values()
                ])
            }
            
        # Analyze layer consistency
        if results:
            for layer in results[0].layer_effects.keys():
                layer_interpretations = [
                    result.steered_interpretations.get(layer, '')
                    for result in results
                ]
                # Simple consistency check (could be more sophisticated)
                consistency = len(set(layer_interpretations)) / len(layer_interpretations)
                comparison['layer_consistency'][layer] = 1.0 - consistency
                
        return comparison
        
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity measure."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0


def create_interpretation_comparison_plot(
    results: List[SteeringInterpretationResult],
    save_path: Optional[str] = None
):
    """
    Create visualization comparing steering effects across layers and strengths.
    
    Args:
        results: List of analysis results
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/seaborn required for plotting")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Steering + Interpretation Analysis', fontsize=16)
    
    # Convert to DataFrame for easier plotting
    analyzer = RepengInterpretationAnalyzer(None, None, None)  # Just for the method
    df = analyzer.create_analysis_dataframe(results)
    
    if df.empty:
        print("No data to plot")
        return
        
    # Plot 1: Magnitude by layer and strength
    if 'magnitude' in df.columns:
        pivot_mag = df.pivot_table(
            values='magnitude', 
            index='layer', 
            columns='strength', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_mag, ax=axes[0,0], annot=True, fmt='.2f', cmap='viridis')
        axes[0,0].set_title('Steering Magnitude by Layer & Strength')
    
    # Plot 2: Mean activation by layer
    if 'mean' in df.columns:
        for strength in df['strength'].unique():
            strength_data = df[df['strength'] == strength]
            axes[0,1].plot(strength_data['layer'], strength_data['mean'], 
                          'o-', label=f'Strength {strength}')
        axes[0,1].set_title('Mean Activation by Layer')
        axes[0,1].set_xlabel('Layer')
        axes[0,1].set_ylabel('Mean Activation')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Text generation changes
    text_changes = df.groupby('strength')['text_changed'].mean()
    axes[1,0].bar(text_changes.index, text_changes.values)
    axes[1,0].set_title('Text Generation Changes by Strength')
    axes[1,0].set_xlabel('Steering Strength')
    axes[1,0].set_ylabel('Fraction Changed')
    
    # Plot 4: Summary statistics
    axes[1,1].axis('off')
    summary_text = f"""Summary Statistics:
    
Prompts analyzed: {df['prompt'].nunique()}
Strengths tested: {sorted(df['strength'].unique())}
Layers analyzed: {sorted(df['layer'].unique())}
Avg magnitude: {df['magnitude'].mean():.2f}
Max magnitude: {df['magnitude'].max():.2f}
    """
    axes[1,1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        
    plt.show()


# Convenience function for quick analysis
def quick_steering_interpretation_analysis(
    steering_model,
    interpretation_model, 
    tokenizer,
    prompt: str,
    steering_vector: SteeringVector,
    strengths: List[float] = [1.0, -1.0],
    max_layers: int = 5
) -> Dict[str, Any]:
    """
    Quick analysis function for steering + interpretation.
    
    Returns a summary dictionary with key findings.
    """
    analyzer = RepengInterpretationAnalyzer(
        steering_model, interpretation_model, tokenizer
    )
    
    # Get representative layers
    all_layers = list(steering_vector.directions.keys())
    selected_layers = all_layers[:max_layers] if len(all_layers) > max_layers else all_layers
    
    # Run analysis
    results = analyzer.analyze_steering_effects(
        prompt=prompt,
        steering_vector=steering_vector,
        strengths=strengths,
        interpretation_layers=selected_layers
    )
    
    # Create summary
    summary = {
        'prompt': prompt,
        'strengths_tested': strengths,
        'layers_analyzed': selected_layers,
        'results': results,
        'comparison': analyzer.compare_interpretations(results),
        'dataframe': analyzer.create_analysis_dataframe(results)
    }
    
    return summary