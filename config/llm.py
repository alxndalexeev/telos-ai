"""
LLM configuration for Telos AI.

This module contains settings and configuration for large language model
providers and specific models used by Telos. It centralizes all LLM-related
configuration to make it easier to switch models or providers.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Enum of supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure_openai"
    LOCAL = "local"

class ModelCapability(Enum):
    """Capabilities that different models may have."""
    CHAT = "chat"
    COMPLETION = "completion"
    CODE = "code_generation"
    REASONING = "complex_reasoning"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    FUNCTION_CALLING = "function_calling"

# Default API configuration
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT = 30  # seconds

# Model configurations
MODELS = {
    # OpenAI Models - Latest Models
    "gpt-4.1": {
        "provider": ModelProvider.OPENAI,
        "capabilities": [
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
        ],
        "max_tokens": 32768,
        "token_limit": 1047576,  # ~1M tokens
        "cost_per_1k_prompt": 2.00,
        "cost_per_1k_completion": 8.00,
    },
    "gpt-4.1-mini": {
        "provider": ModelProvider.OPENAI,
        "capabilities": [
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
        ],
        "max_tokens": 32768,
        "token_limit": 1047576,  # ~1M tokens
        "cost_per_1k_prompt": 1.00,
        "cost_per_1k_completion": 4.00,
    },
    "gpt-4.1-nano": {
        "provider": ModelProvider.OPENAI,
        "capabilities": [
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
        ],
        "max_tokens": 32768,
        "token_limit": 1047576,  # ~1M tokens
        "cost_per_1k_prompt": 0.50,
        "cost_per_1k_completion": 2.00,
    },
    "gpt-4o": {
        "provider": ModelProvider.OPENAI,
        "capabilities": [
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
        ],
        "max_tokens": 16384,
        "token_limit": 128000,
        "cost_per_1k_prompt": 0.005,
        "cost_per_1k_completion": 0.015,
    },
    "gpt-4o-mini": {
        "provider": ModelProvider.OPENAI,
        "capabilities": [
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
        ],
        "max_tokens": 16384,
        "token_limit": 128000,
        "cost_per_1k_prompt": 0.0015,
        "cost_per_1k_completion": 0.006,
    },
    "gpt-4-turbo-preview": {
        "provider": ModelProvider.OPENAI,
        "capabilities": [
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.FUNCTION_CALLING,
        ],
        "max_tokens": 4096,
        "token_limit": 128000,
        "cost_per_1k_prompt": 0.01,
        "cost_per_1k_completion": 0.03,
    },
    "gpt-4": {
        "provider": ModelProvider.OPENAI,
        "capabilities": [
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.FUNCTION_CALLING,
        ],
        "max_tokens": 4096,
        "token_limit": 8192,
        "cost_per_1k_prompt": 0.03,
        "cost_per_1k_completion": 0.06,
    },
    "gpt-4-vision-preview": {
        "provider": ModelProvider.OPENAI,
        "capabilities": [
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING,
        ],
        "max_tokens": 4096,
        "token_limit": 128000,
        "cost_per_1k_prompt": 0.01,
        "cost_per_1k_completion": 0.03,
    },
    "gpt-3.5-turbo": {
        "provider": ModelProvider.OPENAI,
        "capabilities": [
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.FUNCTION_CALLING,
        ],
        "max_tokens": 4096,
        "token_limit": 16385,
        "cost_per_1k_prompt": 0.0015,
        "cost_per_1k_completion": 0.002,
    },
    "text-embedding-3-small": {
        "provider": ModelProvider.OPENAI,
        "capabilities": [ModelCapability.EMBEDDINGS],
        "dimensions": 1536,
        "cost_per_1k_prompt": 0.00002,
    },
    "text-embedding-3-large": {
        "provider": ModelProvider.OPENAI,
        "capabilities": [ModelCapability.EMBEDDINGS],
        "dimensions": 3072,
        "cost_per_1k_prompt": 0.00013,
    },
    
    # Anthropic Models
    "claude-3-opus": {
        "provider": ModelProvider.ANTHROPIC,
        "capabilities": [
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.VISION,
        ],
        "max_tokens": 4096,
        "token_limit": 200000,
        "cost_per_1k_prompt": 0.015,
        "cost_per_1k_completion": 0.075,
    },
    "claude-3-sonnet": {
        "provider": ModelProvider.ANTHROPIC,
        "capabilities": [
            ModelCapability.CHAT,
            ModelCapability.CODE,
            ModelCapability.REASONING,
            ModelCapability.VISION,
        ],
        "max_tokens": 4096,
        "token_limit": 200000,
        "cost_per_1k_prompt": 0.003,
        "cost_per_1k_completion": 0.015,
    },
    "claude-3-haiku": {
        "provider": ModelProvider.ANTHROPIC,
        "capabilities": [
            ModelCapability.CHAT,
            ModelCapability.VISION,
        ],
        "max_tokens": 4096,
        "token_limit": 200000,
        "cost_per_1k_prompt": 0.00025,
        "cost_per_1k_completion": 0.00125,
    },
}

# Default models to use for specific tasks
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_REASONING_MODEL = "gpt-4.1"
DEFAULT_CODE_MODEL = "gpt-4.1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_VISION_MODEL = "gpt-4.1"

# Provider API keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

def get_model_for_capability(capability: ModelCapability) -> str:
    """
    Get the default model for a specific capability.
    
    Args:
        capability: The capability required
        
    Returns:
        Model name string
    """
    capability_map = {
        ModelCapability.CHAT: DEFAULT_CHAT_MODEL,
        ModelCapability.COMPLETION: DEFAULT_CHAT_MODEL,
        ModelCapability.CODE: DEFAULT_CODE_MODEL,
        ModelCapability.REASONING: DEFAULT_REASONING_MODEL,
        ModelCapability.VISION: DEFAULT_VISION_MODEL,
        ModelCapability.EMBEDDINGS: DEFAULT_EMBEDDING_MODEL,
        ModelCapability.FUNCTION_CALLING: DEFAULT_REASONING_MODEL,
    }
    
    return capability_map.get(capability, DEFAULT_CHAT_MODEL)

def get_available_models() -> List[str]:
    """
    Get a list of available models based on configured API keys.
    
    Returns:
        List of available model names
    """
    available_models = []
    
    # Check which providers we have API keys for
    available_providers = set()
    if OPENAI_API_KEY:
        available_providers.add(ModelProvider.OPENAI)
    if ANTHROPIC_API_KEY:
        available_providers.add(ModelProvider.ANTHROPIC)
    if GOOGLE_API_KEY:
        available_providers.add(ModelProvider.GOOGLE)
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        available_providers.add(ModelProvider.AZURE)
    
    # Add models from available providers
    for model_name, model_config in MODELS.items():
        if model_config["provider"] in available_providers:
            available_models.append(model_name)
    
    return available_models

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model configuration
        
    Raises:
        ValueError: If model is not found
    """
    if model_name in MODELS:
        return MODELS[model_name]
    
    raise ValueError(f"Model '{model_name}' not found in configuration")

def validate_model_availability(model_name: str) -> bool:
    """
    Check if a specific model is available with current API keys.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model is available, False otherwise
    """
    if model_name not in MODELS:
        return False
    
    model_provider = MODELS[model_name]["provider"]
    
    # Check if we have the API key for this provider
    if model_provider == ModelProvider.OPENAI and OPENAI_API_KEY:
        return True
    if model_provider == ModelProvider.ANTHROPIC and ANTHROPIC_API_KEY:
        return True
    if model_provider == ModelProvider.GOOGLE and GOOGLE_API_KEY:
        return True
    if (model_provider == ModelProvider.AZURE and 
        AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT):
        return True
    if model_provider == ModelProvider.LOCAL:
        return True
    
    return False

def initialize_llm_config() -> None:
    """
    Initialize and validate LLM configuration.
    Logs warnings for missing API keys and sets fallback models if needed.
    """
    logger.info("Initializing LLM configuration")
    
    # Check which providers are available
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not found. OpenAI models will not be available.")
    
    if not ANTHROPIC_API_KEY:
        logger.warning("Anthropic API key not found. Claude models will not be available.")
    
    if not GOOGLE_API_KEY:
        logger.warning("Google API key not found. Google models will not be available.")
    
    if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT):
        logger.warning("Azure OpenAI configuration incomplete. Azure models will not be available.")
    
    # Log available models
    available = get_available_models()
    logger.info(f"Available models: {', '.join(available)}")
    
    # Check if default models are available
    if DEFAULT_CHAT_MODEL not in available:
        logger.warning(f"Default chat model {DEFAULT_CHAT_MODEL} not available.")
    
    if DEFAULT_REASONING_MODEL not in available:
        logger.warning(f"Default reasoning model {DEFAULT_REASONING_MODEL} not available.")
    
    if DEFAULT_CODE_MODEL not in available:
        logger.warning(f"Default code model {DEFAULT_CODE_MODEL} not available.") 