# NNsight Selfie

A model-agnostic implementation of the [selfie library](https://github.com/tonychenxyz/selfie) using [NNsight](https://github.com/ndif-team/nnsight) for neural network interpretation.

## Overview

This library extends the original [selfie approach](https://arxiv.org/abs/2403.10949) to work with any transformer model architecture through NNsight's universal interface. While the original selfie was designed specifically for LLaMA models, our implementation provides the same interpretation capabilities across GPT, BERT, T5, and other transformer architectures.

## Key Features

- **Model Agnostic**: Works with GPT, LLaMA, BERT, T5, and other transformer models
- **Cross-Platform**: Automatic device detection (MPS, CUDA, CPU)
- **Activation Extraction**: Extract activations from any layer and token position
- **Activation Injection**: Inject activations during generation to steer model behavior
- **Vector Arithmetic**: Perform concept arithmetic like "King - Man + Woman = Queen"
- **Flexible Interpretation**: Create custom interpretation prompts with placeholders
- **Batch Processing**: Efficiently process multiple interpretations
- **Analysis Tools**: Utilities for activation analysis, similarity computation, and vector projections

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd NNsight_selfie

# Install dependencies
pip install torch transformers nnsight tqdm pandas numpy pytest

# Install in development mode
pip install -e .
```

## Quick Start

```python
from nnsight_selfie import ModelAgnosticSelfie, InterpretationPrompt

# Initialize with any HuggingFace model (automatically detects MPS/CUDA/CPU)
selfie = ModelAgnosticSelfie("openai-community/gpt2")

# Create interpretation prompt
interpretation_prompt = InterpretationPrompt.create_concept_prompt(selfie.model.tokenizer)

# Interpret specific tokens
results = selfie.interpret(
    original_prompt="The Eiffel Tower is in Paris",
    interpretation_prompt=interpretation_prompt,
    tokens_to_interpret=[(6, 2)],  # Layer 6, token 2
    max_new_tokens=10
)

print(f"Token interpretation: {results['interpretation'][0]}")
```

## Device Compatibility

NNsight Selfie automatically detects and uses the optimal device for your system:

```python
from nnsight_selfie import get_optimal_device

# Automatic device detection: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
device = get_optimal_device()
selfie = ModelAgnosticSelfie("google/gemma-2-2b-it")  # Uses optimal device automatically

# Or specify manually
selfie = ModelAgnosticSelfie("google/gemma-3-4b-it", device="cuda")  # Force specific device
```

## Core Components

### ModelAgnosticSelfie

The main class that wraps any transformer model and provides interpretation capabilities:

- `get_activations()`: Extract activations from specified layers/tokens
- `inject_activation()`: Generate text with injected activations  
- `interpret()`: High-level interpretation of model tokens
- `interpret_vectors()`: Interpret arbitrary activation vectors

### InterpretationPrompt

Helper class for creating interpretation prompts with placeholders:

- `create_concept_prompt()`: "This represents the concept of _"
- `create_sentiment_prompt()`: "This expresses the sentiment of _"  
- `create_entity_prompt()`: "This refers to the entity _"
- Custom prompts with multiple placeholders supported

## Examples

### Basic Token Interpretation

```python
selfie = ModelAgnosticSelfie("google/gemma-2-2b-it")
prompt = InterpretationPrompt.create_concept_prompt(selfie.model.tokenizer)

results = selfie.interpret(
    original_prompt="I love sunny weather",
    interpretation_prompt=prompt,
    tokens_to_interpret=[(4, 2), (4, 3)],  # "sunny", "weather"
    max_new_tokens=15
)

for i, interpretation in enumerate(results['interpretation']):
    token = results['token_decoded'][i]
    print(f"'{token}' represents: {interpretation}")
```

### Vector Arithmetic

```python
# Extract activations for different concepts
king_acts = selfie.get_activations("The king ruled wisely", layer_indices=[6])
man_acts = selfie.get_activations("The man walked home", layer_indices=[6])
woman_acts = selfie.get_activations("The woman read quietly", layer_indices=[6])

# Perform king - man + woman
result_vector = king_acts[6][:,1,:] - man_acts[6][:,1,:] + woman_acts[6][:,1,:]

# Interpret the result
interpretation = selfie.interpret_vectors(
    vectors=[result_vector],
    interpretation_prompt=prompt,
    max_new_tokens=10
)[0]

print(f"King - Man + Woman = {interpretation}")  # Expected: Queen-like concept
```

*See [`examples/Vector_Arithmetic_Lab.ipynb`](examples/Vector_Arithmetic_Lab.ipynb) for comprehensive vector arithmetic experiments and analysis.*

### Multi-Model Analysis

```python
models = ["google/gemma-2-2b-it", "google/gemma-3-4b-it", "meta-llama/Llama-3.2-1B-Instruct"]

for model_name in models:
    selfie = ModelAgnosticSelfie(model_name)
    print(f"\n{model_name}: {len(selfie.layer_paths)} layers detected")
    
    # Quick interpretation test
    results = selfie.interpret(
        original_prompt="Hello world",
        interpretation_prompt=prompt,
        tokens_to_interpret=[(2, 0)],
        max_new_tokens=8
    )
    print(f"Interpretation: {results['interpretation'][0]}")
```

## Supported Model Architectures

The library automatically detects layer structures for:

- **GPT-style**: GPT-2, GPT-Neo, DialoGPT (layers at `transformer.h.*`)
- **LLaMA-style**: LLaMA, Alpaca, Vicuna (layers at `model.layers.*`) 
- **BERT-style**: BERT, RoBERTa, DeBERTa (layers at `bert.encoder.layer.*`)
- **T5/BART**: Encoder-decoder models (layers at `encoder.layer.*`, `decoder.layers.*`)

For unsupported architectures, the library attempts automatic layer detection.

## Advanced Features

### Batch Processing

```python
from nnsight_selfie.utils import batch_process_interpretations

prompts = [
    "London is the capital of England",
    "Tokyo is a bustling city", 
    "Paris is known for art"
]

tokens_per_prompt = [[(4, 0)], [(4, 0)], [(4, 0)]]  # First token in each

results = batch_process_interpretations(
    selfie, prompts, interpretation_prompt, 
    tokens_per_prompt, batch_size=2
)
```

### Activation Analysis

```python
from nnsight_selfie.utils import compute_activation_similarity

# Compare activations between similar concepts
dog_acts = selfie.get_activations("The dog barked", layer_indices=[6])
cat_acts = selfie.get_activations("The cat meowed", layer_indices=[6])

similarity = compute_activation_similarity(
    dog_acts[6][:,1,:], cat_acts[6][:,1,:], 
    metric='cosine'
)
print(f"Dog-Cat similarity: {similarity:.3f}")
```

### Custom Intervention Modes

```python
# Different intervention strategies
results_normalized = selfie.interpret(
    prompt, interpretation_prompt, [(6, 2)],
    replacing_mode='normalized',  # Blend with original
    overlay_strength=0.7
)

results_additive = selfie.interpret(
    prompt, interpretation_prompt, [(6, 2)], 
    replacing_mode='addition',    # Add to original
    overlay_strength=0.5  
)
```

## Comparison with Original Selfie

| Feature | Original Selfie | NNsight Selfie |
|---------|----------------|----------------|
| Model Support | LLaMA only | Any transformer |
| Implementation | Custom forward wrappers | NNsight tracing |
| Layer Access | Hardcoded paths | Dynamic detection |
| Intervention | Manual injection | Context managers |
| Batching | Manual | Built-in |
| Remote Execution | No | Yes (via NNsight) |

## API Reference

See the docstrings in the source code for detailed API documentation:

- [`ModelAgnosticSelfie`](nnsight_selfie/model_agnostic_selfie.py)
- [`InterpretationPrompt`](nnsight_selfie/interpretation_prompt.py)  
- [`Utils`](nnsight_selfie/utils.py)

## Examples and Tests

**🔬 Vector Arithmetic Lab**: [`examples/Vector_Arithmetic_Lab.ipynb`](examples/Vector_Arithmetic_Lab.ipynb) - Interactive notebook demonstrating:
- Classic vector arithmetic (King - Man + Woman = Queen)
- Multi-string concept extraction for robust results
- Vector projections and directional analysis
- Emotion and abstract concept arithmetic
- Layer comparison and multi-layer aggregation

Other examples:
- [`examples/basic_usage.py`](examples/basic_usage.py): Basic functionality examples
- [`examples/advanced_usage.py`](examples/advanced_usage.py): Advanced analysis workflows
- [`tests/test_basic.py`](tests/test_basic.py): Unit tests

Run tests:
```bash
pytest tests/ -v
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers  
- NNsight
- tqdm, pandas, numpy (for utilities)
- pytest (for testing)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original [selfie](https://github.com/tonychenxyz/selfie) library for inspiration
- [NNsight](https://github.com/ndif-team/nnsight) for the universal intervention framework
- HuggingFace Transformers for model implementations

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure NNsight is installed: `pip install nnsight`
2. **CUDA/Memory Issues**: Use `device="cpu"` for testing, or smaller models like `google/gemma-2-2b-it`
3. **Model Not Found**: Ensure model name is correct and you have internet access
4. **Layer Detection Failed**: Check model architecture or provide custom layer paths
5. **Device Issues**: Library supports MPS (Apple Silicon), CUDA (NVIDIA), and CPU with automatic fallback

### Getting Help

- Check the examples for usage patterns
- Look at test files for expected behavior
- Open an issue for bugs or feature requests