"""
Model configuration module for NNsight Selfie.

Provides flexible configuration for loading models either locally with quantization
or remotely via NDIF.
"""

import os
import torch
from typing import Optional

try:
    import nnsight
    from nnsight import LanguageModel, CONFIG
except ImportError:
    raise ImportError("NNsight is required. Install it with: pip install nnsight")

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


def detect_colab():
    """Detect if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def configure_model_loading(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
    remote: bool = True,
    n_examples: int = 2,
    quantization: bool = False,  # Changed default to False for no quantization
    clear_output: bool = True
) -> LanguageModel:
    """
    Configure and load a language model with flexible options.
    
    Args:
        model_name: HuggingFace model identifier
        remote: Whether to use NDIF hosted model (True) or load locally (False)
        n_examples: Number of examples for future effect tests
        quantization: Whether to use quantization when loading locally
        clear_output: Whether to clear output after loading
        
    Returns:
        Configured LanguageModel instance
    """
    is_colab = detect_colab()
    
    # Model configuration
    print(f"Model configuration:")
    print(f"  Remote (NDIF): {remote}")
    print(f"  Model: {model_name}")
    print(f"  N_examples: {n_examples}")
    print(f"  Quantization (local only): {quantization}")
    
    if remote:
        # Configure for remote NDIF usage
        if is_colab:
            try:
                from google.colab import userdata
                NDIF_API = userdata.get('NDIF_API')
                HF_TOKEN = userdata.get('HF_TOKEN')
                
                CONFIG.set_default_api_key(NDIF_API)
                
                # Login to HuggingFace
                import subprocess
                subprocess.run(['huggingface-cli', 'login', '--token', HF_TOKEN], check=True)
            except Exception as e:
                print(f"Warning: Failed to configure Colab secrets: {e}")
                print("Please ensure NDIF_API and HF_TOKEN are set in Colab secrets")
        else:
            # Use environment variables for non-Colab environments
            if "NDIF_API" in os.environ:
                nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API"]
            else:
                print("Warning: NDIF_API environment variable not set")
        
        # Load remote model
        print("Loading remote model via NDIF...")
        llm = LanguageModel(model_name)
    else:
        # Configure for local loading with optional quantization
        if quantization and BitsAndBytesConfig is not None:
            print("Loading local model with 8-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
            llm = LanguageModel(
                model_name, 
                device_map="auto", 
                quantization_config=bnb_config, 
                torch_dtype=torch.bfloat16
            )
        else:
            if quantization and BitsAndBytesConfig is None:
                print("Warning: BitsAndBytesConfig not available. Loading without quantization.")
            print("Loading local model...")
            llm = LanguageModel(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    
    llm.eval()
    
    if clear_output:
        try:
            from IPython.display import clear_output
            clear_output()
        except ImportError:
            pass  # Not in Jupyter/Colab
    
    print(f"Model loaded successfully: {llm}")
    return llm


class ModelConfig:
    """
    Configuration class for model loading with sensible defaults.
    """
    
    def __init__(
        self,
        remote: bool = True,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B", 
        n_examples: int = 2,
        quantization: bool = False  # Changed default to False for no quantization
    ):
        self.remote = remote
        self.model_name = model_name
        self.n_examples = n_examples
        self.quantization = quantization
        self.is_colab = detect_colab()
        
    def load_model(self, clear_output: bool = True) -> LanguageModel:
        """Load model with current configuration."""
        return configure_model_loading(
            model_name=self.model_name,
            remote=self.remote,
            n_examples=self.n_examples,
            quantization=self.quantization,
            clear_output=clear_output
        )
    
    def __repr__(self):
        return (f"ModelConfig(remote={self.remote}, model_name='{self.model_name}', "
                f"n_examples={self.n_examples}, quantization={self.quantization})")