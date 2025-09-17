"""
Dataset generator based on repeng methodology for creating steering vectors.

This module provides functionality to:
1. Load repeng's truncated outputs
2. Generate positive/negative persona pairs
3. Create dataset entries with customizable persona prompts
4. Support for different template types
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import dataclasses
from pathlib import Path
import torch
from transformers import PreTrainedTokenizer


@dataclasses.dataclass
class DatasetEntry:
    """
    A single entry in the steering dataset containing positive and negative examples.
    """
    positive: str
    negative: str


class RepengDatasetGenerator:
    """
    Generate datasets for steering vector training using repeng methodology.
    
    This class creates paired positive/negative examples by:
    1. Loading truncated conversation snippets from repeng data
    2. Applying persona templates to create contrasting examples
    3. Supporting token-level truncation for multiple training examples
    """
    
    def __init__(self, repeng_data_path: Optional[str] = None):
        """
        Initialize the dataset generator.
        
        Args:
            repeng_data_path: Path to repeng data directory (optional, will look in third_party)
        """
        self.repeng_data_path = self._find_repeng_data_path(repeng_data_path)
        self.truncated_outputs = self._load_truncated_outputs()
        
    def _find_repeng_data_path(self, provided_path: Optional[str]) -> Path:
        """Find the repeng data directory."""
        if provided_path:
            return Path(provided_path)
        
        # Look in third_party directory
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # Go up to project root
        repeng_data = project_root / "third_party" / "repeng" / "repeng" / "notebooks" / "data"
        
        if repeng_data.exists():
            return repeng_data
        else:
            raise FileNotFoundError(f"Could not find repeng data directory at {repeng_data}")
    
    def _load_truncated_outputs(self) -> List[str]:
        """Load the truncated outputs from repeng data."""
        truncated_file = self.repeng_data_path / "all_truncated_outputs.json"
        
        if not truncated_file.exists():
            raise FileNotFoundError(f"Could not find {truncated_file}")
        
        with open(truncated_file, 'r') as f:
            outputs = json.load(f)
        
        # Filter out empty strings and very short outputs
        filtered_outputs = [output for output in outputs if output.strip() and len(output.strip()) > 2]
        
        print(f"Loaded {len(filtered_outputs)} truncated outputs from repeng data")
        return filtered_outputs
    
    def create_truncated_dataset(
        self,
        tokenizer: PreTrainedTokenizer,
        template: str = "Act as if you're extremely {persona}. {suffix}",
        positive_personas: List[str] = None,
        negative_personas: List[str] = None,
        max_suffixes: Optional[int] = 500,
        min_tokens: int = 1,
        max_tokens: Optional[int] = None
    ) -> List[DatasetEntry]:
        """
        Create a truncated dataset similar to repeng methodology.
        
        Args:
            tokenizer: Tokenizer to use for text processing
            template: Template string with {persona} and {suffix} placeholders
            positive_personas: List of positive personas (default: ["happy"])
            negative_personas: List of negative personas (default: ["sad"])
            max_suffixes: Maximum number of suffix variations to use
            min_tokens: Minimum number of tokens per truncated suffix
            max_tokens: Maximum number of tokens per truncated suffix
            
        Returns:
            List of DatasetEntry objects
        """
        if positive_personas is None:
            positive_personas = ["happy"]
        if negative_personas is None:
            negative_personas = ["sad"]
            
        if len(positive_personas) != len(negative_personas):
            raise ValueError("Number of positive and negative personas must match")
        
        # Generate truncated suffixes
        truncated_suffixes = self._generate_truncated_suffixes(
            tokenizer, max_suffixes, min_tokens, max_tokens
        )
        
        # Create dataset entries
        dataset = []
        for suffix in truncated_suffixes:
            for pos_persona, neg_persona in zip(positive_personas, negative_personas):
                positive_text = template.format(persona=pos_persona, suffix=suffix)
                negative_text = template.format(persona=neg_persona, suffix=suffix)
                
                dataset.append(DatasetEntry(
                    positive=positive_text,
                    negative=negative_text
                ))
        
        print(f"Generated dataset with {len(dataset)} entries")
        return dataset
    
    def _generate_truncated_suffixes(
        self,
        tokenizer: PreTrainedTokenizer,
        max_suffixes: Optional[int],
        min_tokens: int,
        max_tokens: Optional[int]
    ) -> List[str]:
        """
        Generate truncated suffixes from the loaded outputs.
        
        This follows repeng's methodology of creating multiple training examples
        by truncating each suffix at different token lengths.
        """
        truncated_suffixes = []
        outputs_to_use = self.truncated_outputs[:max_suffixes] if max_suffixes else self.truncated_outputs
        
        for output in outputs_to_use:
            # Tokenize the output
            tokens = tokenizer.tokenize(output)
            
            if len(tokens) < min_tokens:
                continue
            
            # Determine max length for this output
            end_idx = len(tokens)
            if max_tokens is not None:
                end_idx = min(end_idx, max_tokens)
            
            # Create truncated versions: from min_tokens to end_idx
            for i in range(min_tokens, end_idx):
                truncated_tokens = tokens[:i]
                truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
                
                # Only add if it's not empty after conversion
                if truncated_text.strip():
                    truncated_suffixes.append(truncated_text.strip())
        
        print(f"Generated {len(truncated_suffixes)} truncated suffixes")
        return truncated_suffixes
    
    def create_custom_dataset(
        self,
        positive_examples: List[str],
        negative_examples: List[str]
    ) -> List[DatasetEntry]:
        """
        Create a dataset from custom positive/negative example lists.
        
        Args:
            positive_examples: List of positive example texts
            negative_examples: List of negative example texts (must match length)
            
        Returns:
            List of DatasetEntry objects
        """
        if len(positive_examples) != len(negative_examples):
            raise ValueError("Number of positive and negative examples must match")
        
        dataset = []
        for pos, neg in zip(positive_examples, negative_examples):
            dataset.append(DatasetEntry(positive=pos, negative=neg))
        
        return dataset
    
    def get_persona_templates(self) -> Dict[str, str]:
        """
        Get common persona templates for different types of steering.
        
        Returns:
            Dictionary of template names to template strings
        """
        return {
            "emotion": "Act as if you're extremely {persona}. {suffix}",
            "personality": "Behave as if you have a {persona} personality. {suffix}",
            "role": "You are a {persona} person. {suffix}",
            "attitude": "Respond with a {persona} attitude. {suffix}",
            "simple": "{persona}: {suffix}",
            "instruction": "Please be {persona} in your response. {suffix}",
        }
    
    def get_common_persona_pairs(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Get common positive/negative persona pairs.
        
        Returns:
            Dictionary mapping category to (positive_personas, negative_personas) tuples
        """
        return {
            "emotion": (
                ["happy", "joyful", "cheerful", "optimistic"],
                ["sad", "depressed", "pessimistic", "gloomy"]
            ),
            "honesty": (
                ["honest", "truthful", "sincere"],
                ["dishonest", "deceptive", "lying"]
            ),
            "aggression": (
                ["peaceful", "calm", "gentle"],
                ["aggressive", "hostile", "violent"]
            ),
            "confidence": (
                ["confident", "self-assured", "bold"],
                ["insecure", "uncertain", "timid"]
            ),
            "helpfulness": (
                ["helpful", "supportive", "cooperative"],
                ["unhelpful", "dismissive", "uncooperative"]
            ),
            "formality": (
                ["formal", "professional", "polite"],
                ["casual", "informal", "relaxed"]
            ),
        }
    
    def save_dataset(self, dataset: List[DatasetEntry], filepath: str):
        """Save dataset to a JSON file."""
        data = [{"positive": entry.positive, "negative": entry.negative} for entry in dataset]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved dataset with {len(dataset)} entries to {filepath}")
    
    def load_dataset(self, filepath: str) -> List[DatasetEntry]:
        """Load dataset from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        dataset = [DatasetEntry(positive=item["positive"], negative=item["negative"]) for item in data]
        print(f"Loaded dataset with {len(dataset)} entries from {filepath}")
        return dataset


def create_quick_dataset(
    tokenizer: PreTrainedTokenizer,
    persona_category: str = "emotion",
    template_type: str = "emotion",
    max_suffixes: int = 500,
    repeng_data_path: Optional[str] = None
) -> List[DatasetEntry]:
    """
    Quick function to create a standard dataset.
    
    Args:
        tokenizer: Tokenizer to use
        persona_category: Category of personas ("emotion", "honesty", etc.)
        template_type: Type of template to use
        max_suffixes: Maximum number of suffixes to use
        repeng_data_path: Optional path to repeng data
        
    Returns:
        List of DatasetEntry objects
    """
    generator = RepengDatasetGenerator(repeng_data_path)
    
    # Get personas and template
    persona_pairs = generator.get_common_persona_pairs()
    templates = generator.get_persona_templates()
    
    if persona_category not in persona_pairs:
        raise ValueError(f"Unknown persona category: {persona_category}")
    if template_type not in templates:
        raise ValueError(f"Unknown template type: {template_type}")
    
    positive_personas, negative_personas = persona_pairs[persona_category]
    template = templates[template_type]
    
    return generator.create_truncated_dataset(
        tokenizer=tokenizer,
        template=template,
        positive_personas=positive_personas,
        negative_personas=negative_personas,
        max_suffixes=max_suffixes
    )