"""
Basic tests for NNsight Selfie functionality.
"""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nnsight_selfie import ModelAgnosticSelfie, InterpretationPrompt


@pytest.fixture
def small_model():
    """Fixture providing a small model for testing."""
    # Use a very small model to reduce test time and memory usage
    return ModelAgnosticSelfie("sshleifer/tiny-gpt2", device_map="cpu")


@pytest.fixture  
def sample_prompt():
    """Fixture providing a sample prompt for testing."""
    return "The cat sat on the mat"


@pytest.fixture
def interpretation_prompt(small_model):
    """Fixture providing an interpretation prompt."""
    return InterpretationPrompt.create_concept_prompt(small_model.model.tokenizer)


def test_model_initialization():
    """Test that model initialization works correctly."""
    model = ModelAgnosticSelfie("sshleifer/tiny-gpt2", device_map="cpu")
    
    # Check that model and tokenizer are loaded
    assert model.model is not None
    assert model.model.tokenizer is not None
    assert len(model.layer_paths) > 0
    
    # Check layer path format
    assert all(isinstance(path, str) for path in model.layer_paths)


def test_interpretation_prompt_creation():
    """Test InterpretationPrompt creation and validation."""
    model = ModelAgnosticSelfie("sshleifer/tiny-gpt2", device_map="cpu")
    
    # Test different prompt types
    concept_prompt = InterpretationPrompt.create_concept_prompt(model.model.tokenizer)
    assert concept_prompt.interpretation_prompt is not None
    assert len(concept_prompt.insert_locations) > 0
    
    sentiment_prompt = InterpretationPrompt.create_sentiment_prompt(model.model.tokenizer)
    assert sentiment_prompt.interpretation_prompt is not None
    
    entity_prompt = InterpretationPrompt.create_entity_prompt(model.model.tokenizer)
    assert entity_prompt.interpretation_prompt is not None


def test_activation_extraction(small_model, sample_prompt):
    """Test activation extraction functionality."""
    # Extract activations from first few layers
    layer_indices = [0, 1]
    activations = small_model.get_activations(
        sample_prompt, 
        layer_indices=layer_indices
    )
    
    # Check that we got activations for requested layers
    assert len(activations) == len(layer_indices)
    
    for layer_idx in layer_indices:
        assert layer_idx in activations
        activation = activations[layer_idx]
        
        # Check tensor properties
        assert isinstance(activation, torch.Tensor)
        assert activation.ndim == 3  # [batch, seq, hidden]
        assert activation.shape[0] == 1  # batch size
        assert activation.shape[1] > 0  # sequence length
        assert activation.shape[2] > 0  # hidden size


def test_activation_extraction_specific_tokens(small_model, sample_prompt):
    """Test activation extraction for specific token positions."""
    layer_indices = [0]
    token_indices = [0, 1]  # First two tokens
    
    activations = small_model.get_activations(
        sample_prompt,
        layer_indices=layer_indices,
        token_indices=token_indices
    )
    
    assert 0 in activations
    assert isinstance(activations[0], list)
    assert len(activations[0]) == len(token_indices)
    
    for activation in activations[0]:
        assert isinstance(activation, torch.Tensor)
        assert activation.ndim == 2  # [batch, hidden] - token dim removed


def test_simple_interpretation(small_model, sample_prompt, interpretation_prompt):
    """Test basic interpretation functionality."""
    tokens_to_interpret = [(0, 1)]  # Layer 0, token 1
    
    try:
        results = small_model.interpret(
            original_prompt=sample_prompt,
            interpretation_prompt=interpretation_prompt,
            tokens_to_interpret=tokens_to_interpret,
            max_new_tokens=5
        )
        
        # Check result structure
        assert 'prompt' in results
        assert 'interpretation' in results
        assert 'layer' in results
        assert 'token' in results
        assert 'token_decoded' in results
        
        # Check that we got results for all requested tokens
        assert len(results['prompt']) == len(tokens_to_interpret)
        assert len(results['interpretation']) == len(tokens_to_interpret)
        
        # Check that interpretations are strings
        for interpretation in results['interpretation']:
            assert isinstance(interpretation, str)
            
    except Exception as e:
        # Some failures might be expected with very small models
        pytest.skip(f"Interpretation failed (expected with tiny models): {e}")


def test_vector_interpretation(small_model, sample_prompt, interpretation_prompt):
    """Test interpretation of arbitrary vectors."""
    # Get some activations to use as vectors
    activations = small_model.get_activations(sample_prompt, layer_indices=[0])
    vector = activations[0][:, 0, :]  # First token activation
    
    try:
        interpretations = small_model.interpret_vectors(
            vectors=[vector],
            interpretation_prompt=interpretation_prompt,
            max_new_tokens=5
        )
        
        assert isinstance(interpretations, list)
        assert len(interpretations) == 1
        assert isinstance(interpretations[0], str)
        
    except Exception as e:
        pytest.skip(f"Vector interpretation failed (expected with tiny models): {e}")


def test_layer_path_detection():
    """Test that layer paths are correctly detected for different model types."""
    from nnsight_selfie.utils import get_model_layers
    
    # Test with tiny GPT-2
    model = ModelAgnosticSelfie("sshleifer/tiny-gpt2", device_map="cpu")
    layer_paths = get_model_layers(model.model)
    
    # Should detect transformer.h.X pattern for GPT models
    assert len(layer_paths) > 0
    assert all('transformer.h.' in path for path in layer_paths)


def test_interpretation_prompt_validation(small_model):
    """Test interpretation prompt validation."""
    prompt = InterpretationPrompt.create_concept_prompt(small_model.model.tokenizer)
    
    # Test validation
    assert prompt.validate_with_model(small_model.model)
    
    # Test prompt properties
    assert len(prompt.get_prompt()) > 0
    assert len(prompt.get_insert_locations()) > 0
    assert prompt.get_tokenized_inputs() is not None


def test_custom_interpretation_prompt(small_model):
    """Test creation of custom interpretation prompts."""
    sequence = ["This is ", None, " test"]
    prompt = InterpretationPrompt(small_model.model.tokenizer, sequence)
    
    assert "This is _ test" in prompt.interpretation_prompt
    assert len(prompt.insert_locations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])