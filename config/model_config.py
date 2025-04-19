"""
Model configuration for Telos AI.

This module provides configuration settings for various AI models,
including LLMs, embedding models, vision models, and audio models.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple

class ModelProvider(Enum):
    """Enumeration of supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LLAMA = "llama"
    MISTRAL = "mistral"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"
    COHERE = "cohere"
    TOGETHER = "together"
    CUSTOM = "custom"
    LOCAL = "local"

class ModelType(Enum):
    """Enumeration of model types."""
    TEXT = "text"
    CHAT = "chat"
    EMBEDDING = "embedding"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    CODE = "code"
    FUNCTION = "function"

class ModelSize(Enum):
    """Enumeration of model sizes."""
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XL = "xl"
    XXL = "xxl"

class ModelContext(Enum):
    """Enumeration of model context sizes."""
    XS = "xs"  # 1-2k
    S = "s"    # 4-8k
    M = "m"    # 16-32k
    L = "l"    # 64-128k
    XL = "xl"  # 128k+
    XXL = "xxl" # 1M+

# Global model settings
GLOBAL_MODEL_CONFIG = {
    "default_provider": ModelProvider.OPENAI,
    "default_model": "gpt-4.1",
    "token_usage_tracking": True,
    "cost_tracking": True,
    "failover_enabled": True,
    "use_cache": True,
    "cache_ttl_minutes": 60,
    "request_timeout_seconds": 60,
    "max_retries": 3,
    "retry_backoff_factor": 2,
    "log_requests": True,
    "log_responses": True,
    "max_parallel_requests": 10,
    "temperature_default": 0.7,
    "top_p_default": 1.0,
    "frequency_penalty_default": 0.0,
    "presence_penalty_default": 0.0,
    "system_message_default": "You are Telos, a helpful AI assistant.",
    "seed_default": None,
    "concurrent_request_limit": 50,
    "rate_limit_rpm": 100,
    "enable_monitoring": True,
    "performance_tracking": True,
    "output_filtering": True,
    "streaming_default": True,
    "debug_mode": False,
    "api_keys_file": "./config/api_keys.json",
    "models_cache_dir": "./data/models",
    "allow_model_switching": True,
    "preferred_output_format": "markdown",
}

# OpenAI model configurations
OPENAI_MODELS = {
    "gpt-4.1": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.XXL,
        "context_length": 1047576,
        "context_size": ModelContext.XXL,
        "input_cost_per_1k": 2.00,
        "output_cost_per_1k": 8.00,
        "token_encoding": "cl100k_base",
        "supports_functions": True,
        "supports_vision": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 20,
        "supports_parallel": True,
        "recommended_roles": ["general", "creative", "code", "reasoning"],
        "description": "OpenAI's latest flagship model with 1M token context window, optimized for coding and reasoning",
        "max_function_calls": 100,
    },
    "gpt-4.1-mini": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.XL,
        "context_length": 1047576,
        "context_size": ModelContext.XXL,
        "input_cost_per_1k": 1.00,
        "output_cost_per_1k": 4.00,
        "token_encoding": "cl100k_base",
        "supports_functions": True,
        "supports_vision": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 20,
        "supports_parallel": True,
        "recommended_roles": ["general", "creative", "code", "reasoning"],
        "description": "Smaller version of GPT-4.1 with lower cost while maintaining excellent capabilities",
        "max_function_calls": 100,
    },
    "gpt-4.1-nano": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.LARGE,
        "context_length": 1047576,
        "context_size": ModelContext.XXL,
        "input_cost_per_1k": 0.50,
        "output_cost_per_1k": 2.00,
        "token_encoding": "cl100k_base",
        "supports_functions": True,
        "supports_vision": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 20,
        "supports_parallel": True,
        "recommended_roles": ["general", "creative", "code"],
        "description": "Smallest and most efficient version of GPT-4.1, ideal for cost-effective tasks",
        "max_function_calls": 100,
    },
    "gpt-4o": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.XL,
        "context_length": 128000,
        "context_size": ModelContext.L,
        "input_cost_per_1k": 0.005,
        "output_cost_per_1k": 0.015,
        "token_encoding": "cl100k_base",
        "supports_functions": True,
        "supports_vision": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 20,
        "supports_parallel": True,
        "recommended_roles": ["general", "creative", "code", "reasoning"],
        "description": "OpenAI's most advanced model, combining capabilities of GPT-4 and vision.",
        "max_function_calls": 100,
    },
    "gpt-4-turbo": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.XL,
        "context_length": 128000,
        "context_size": ModelContext.L,
        "input_cost_per_1k": 0.01,
        "output_cost_per_1k": 0.03,
        "token_encoding": "cl100k_base",
        "supports_functions": True,
        "supports_vision": False,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 20,
        "supports_parallel": True,
        "recommended_roles": ["general", "creative", "code", "reasoning"],
        "description": "OpenAI's GPT-4 Turbo model with improved capabilities at lower cost.",
        "max_function_calls": 100,
    },
    "gpt-4-vision-preview": {
        "model_type": ModelType.MULTIMODAL,
        "model_size": ModelSize.XL,
        "context_length": 128000,
        "context_size": ModelContext.L,
        "input_cost_per_1k": 0.01,
        "output_cost_per_1k": 0.03,
        "token_encoding": "cl100k_base",
        "supports_functions": True,
        "supports_vision": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 10,
        "supports_parallel": True,
        "recommended_roles": ["vision", "image_analysis", "multimodal"],
        "description": "GPT-4 with vision capabilities for analyzing images.",
        "vision_pricing": {
            "low_res_cost_per_image": 0.002,
            "high_res_cost_per_image": 0.004,
        },
        "max_function_calls": 100,
    },
    "gpt-3.5-turbo": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.LARGE,
        "context_length": 16385,
        "context_size": ModelContext.S,
        "input_cost_per_1k": 0.0005,
        "output_cost_per_1k": 0.0015,
        "token_encoding": "cl100k_base",
        "supports_functions": True,
        "supports_vision": False,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 50,
        "supports_parallel": True,
        "recommended_roles": ["general", "simple_tasks", "high_throughput"],
        "description": "OpenAI's faster and cheaper model with good capabilities for many tasks.",
        "max_function_calls": 100,
    },
    "text-embedding-3-small": {
        "model_type": ModelType.EMBEDDING,
        "model_size": ModelSize.SMALL,
        "context_length": 8191,
        "context_size": ModelContext.S,
        "cost_per_1k": 0.00002,
        "token_encoding": "cl100k_base",
        "dimensions": 1536,
        "max_batch_size": 2048,
        "supports_parallel": True,
        "description": "OpenAI's smaller embedding model with great quality/price ratio.",
    },
    "text-embedding-3-large": {
        "model_type": ModelType.EMBEDDING,
        "model_size": ModelSize.LARGE,
        "context_length": 8191,
        "context_size": ModelContext.S,
        "cost_per_1k": 0.00013,
        "token_encoding": "cl100k_base",
        "dimensions": 3072,
        "max_batch_size": 2048,
        "supports_parallel": True,
        "description": "OpenAI's larger embedding model with highest quality.",
    },
    "whisper-1": {
        "model_type": ModelType.AUDIO,
        "model_size": ModelSize.MEDIUM,
        "max_duration_seconds": 600,
        "cost_per_minute": 0.006,
        "supports_translation": True,
        "supported_formats": ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
        "max_file_size_mb": 25,
        "languages": "multilingual",
        "description": "OpenAI's Whisper model for audio transcription and translation.",
    },
    "tts-1": {
        "model_type": ModelType.AUDIO,
        "model_size": ModelSize.SMALL,
        "cost_per_1k": 0.015,
        "max_input_characters": 4096,
        "supported_voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        "supported_formats": ["mp3", "opus", "aac", "flac"],
        "description": "OpenAI's TTS (text-to-speech) model.",
    },
    "tts-1-hd": {
        "model_type": ModelType.AUDIO,
        "model_size": ModelSize.MEDIUM,
        "cost_per_1k": 0.030,
        "max_input_characters": 4096,
        "supported_voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        "supported_formats": ["mp3", "opus", "aac", "flac"],
        "description": "OpenAI's high-definition TTS (text-to-speech) model.",
    },
    "dall-e-3": {
        "model_type": ModelType.VISION,
        "model_size": ModelSize.LARGE,
        "cost_per_image": {
            "standard": 0.040,
            "hd": 0.080,
        },
        "sizes": ["1024x1024", "1024x1792", "1792x1024"],
        "quality": ["standard", "hd"],
        "styles": ["vivid", "natural"],
        "description": "OpenAI's DALL-E 3 model for image generation.",
    },
}

# Anthropic model configurations
ANTHROPIC_MODELS = {
    "claude-3-opus": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.XXL,
        "context_length": 200000,
        "context_size": ModelContext.XL,
        "input_cost_per_1k": 0.015,
        "output_cost_per_1k": 0.075,
        "token_encoding": "claude",
        "supports_functions": True,
        "supports_vision": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 5,
        "supports_parallel": True,
        "recommended_roles": ["general", "reasoning", "creative", "security"],
        "description": "Anthropic's most powerful model with exceptional reasoning and writing capabilities.",
    },
    "claude-3-sonnet": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.XL,
        "context_length": 200000,
        "context_size": ModelContext.XL,
        "input_cost_per_1k": 0.003,
        "output_cost_per_1k": 0.015,
        "token_encoding": "claude",
        "supports_functions": True,
        "supports_vision": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 10,
        "supports_parallel": True,
        "recommended_roles": ["general", "reasoning", "creative"],
        "description": "Anthropic's balanced model with good capabilities and cost balance.",
    },
    "claude-3-haiku": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.MEDIUM,
        "context_length": 200000,
        "context_size": ModelContext.XL,
        "input_cost_per_1k": 0.00025,
        "output_cost_per_1k": 0.00125,
        "token_encoding": "claude",
        "supports_functions": True,
        "supports_vision": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 50,
        "supports_parallel": True,
        "recommended_roles": ["simple_tasks", "high_throughput", "classification"],
        "description": "Anthropic's fastest and most cost-effective model, good for simple tasks.",
    },
}

# Google model configurations
GOOGLE_MODELS = {
    "gemini-1.5-pro": {
        "model_type": ModelType.MULTIMODAL,
        "model_size": ModelSize.XL,
        "context_length": 1000000,
        "context_size": ModelContext.XXL,
        "input_cost_per_1k": 0.00035,
        "output_cost_per_1k": 0.00140,
        "token_encoding": "gemini",
        "supports_functions": True,
        "supports_vision": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 10,
        "supports_parallel": True,
        "recommended_roles": ["general", "multimodal", "long_context"],
        "description": "Google's most capable model with massive context window and multimodal abilities.",
    },
    "gemini-1.5-flash": {
        "model_type": ModelType.MULTIMODAL,
        "model_size": ModelSize.LARGE,
        "context_length": 1000000,
        "context_size": ModelContext.XXL,
        "input_cost_per_1k": 0.00010,
        "output_cost_per_1k": 0.00035,
        "token_encoding": "gemini",
        "supports_functions": True,
        "supports_vision": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 20,
        "supports_parallel": True,
        "recommended_roles": ["general", "high_throughput", "multimodal"],
        "description": "Google's faster and more cost-effective model with a large context window.",
    },
    "gemini-1.0-pro": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.LARGE,
        "context_length": 32768,
        "context_size": ModelContext.M,
        "input_cost_per_1k": 0.00025,
        "output_cost_per_1k": 0.00125,
        "token_encoding": "gemini",
        "supports_functions": True,
        "supports_vision": True,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 20,
        "supports_parallel": True,
        "recommended_roles": ["general", "reasoning", "code"],
        "description": "Google's balanced model with good performance across a variety of tasks.",
    },
}

# Mistral model configurations
MISTRAL_MODELS = {
    "mistral-large": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.LARGE,
        "context_length": 32768,
        "context_size": ModelContext.M,
        "input_cost_per_1k": 0.002,
        "output_cost_per_1k": 0.006,
        "token_encoding": "mistral",
        "supports_functions": True,
        "supports_vision": False,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 20,
        "supports_parallel": True,
        "recommended_roles": ["general", "code", "reasoning"],
        "description": "Mistral's most capable model with excellent reasoning and coding abilities.",
    },
    "mistral-small": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.MEDIUM,
        "context_length": 32768,
        "context_size": ModelContext.M,
        "input_cost_per_1k": 0.0002,
        "output_cost_per_1k": 0.0006,
        "token_encoding": "mistral",
        "supports_functions": True,
        "supports_vision": False,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "max_batch_size": 50,
        "supports_parallel": True,
        "recommended_roles": ["general", "high_throughput"],
        "description": "Mistral's balanced model with good performance and cost efficiency.",
    },
}

# Llama model configurations
LLAMA_MODELS = {
    "llama-3-70b-instruct": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.XL,
        "context_length": 8192,
        "context_size": ModelContext.S,
        "local_path": None,  # Set if running locally
        "quantization": None,  # Options: None, 4, 8, 16 (bits)
        "supports_functions": False,
        "supports_vision": False,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "description": "Meta's largest Llama 3 model with instruction tuning.",
        "recommended_vram_gb": 80,
        "recommended_roles": ["general", "creative", "reasoning"],
        "local_config": {
            "use_gpu": True,
            "num_gpu": "auto",
            "max_tokens": 4096,
            "load_in_8bit": False,
            "load_in_4bit": True,
            "rope_scaling": {"type": "dynamic", "factor": 2.0},
            "flash_attn": True,
        },
    },
    "llama-3-8b-instruct": {
        "model_type": ModelType.CHAT,
        "model_size": ModelSize.MEDIUM,
        "context_length": 8192,
        "context_size": ModelContext.S,
        "local_path": None,  # Set if running locally
        "quantization": None,  # Options: None, 4, 8, 16 (bits)
        "supports_functions": False,
        "supports_vision": False,
        "supports_streaming": True,
        "default_temperature": 0.7,
        "description": "Meta's smaller Llama 3 model that can run on consumer hardware.",
        "recommended_vram_gb": 10,
        "recommended_roles": ["general", "local_deployment"],
        "local_config": {
            "use_gpu": True,
            "num_gpu": 1,
            "max_tokens": 4096,
            "load_in_8bit": False,
            "load_in_4bit": True,
            "flash_attn": True,
        },
    },
}

# Provider configurations
PROVIDER_CONFIGS = {
    ModelProvider.OPENAI: {
        "api_base": "https://api.openai.com/v1",
        "api_type": "openai",
        "models": OPENAI_MODELS,
        "default_model": "gpt-4o",
        "organization_id": "",
        "request_timeout": 60,
        "streaming_supported": True,
        "functions_supported": True,
        "vision_supported": True,
        "max_retries": 3,
        "default_headers": {},
    },
    
    ModelProvider.ANTHROPIC: {
        "api_base": "https://api.anthropic.com/v1",
        "models": ANTHROPIC_MODELS,
        "default_model": "claude-3-sonnet",
        "request_timeout": 60,
        "streaming_supported": True,
        "functions_supported": True,
        "vision_supported": True,
        "max_retries": 3,
        "default_headers": {
            "anthropic-version": "2023-06-01",
        },
    },
    
    ModelProvider.GOOGLE: {
        "api_base": "https://generativelanguage.googleapis.com/v1beta",
        "models": GOOGLE_MODELS,
        "default_model": "gemini-1.5-pro",
        "request_timeout": 60,
        "streaming_supported": True,
        "functions_supported": True,
        "vision_supported": True,
        "max_retries": 3,
    },
    
    ModelProvider.MISTRAL: {
        "api_base": "https://api.mistral.ai/v1",
        "models": MISTRAL_MODELS,
        "default_model": "mistral-large",
        "request_timeout": 60,
        "streaming_supported": True,
        "functions_supported": True,
        "vision_supported": False,
        "max_retries": 3,
    },
    
    ModelProvider.LLAMA: {
        "models": LLAMA_MODELS,
        "default_model": "llama-3-8b-instruct",
        "local_only": True,
        "streaming_supported": True,
        "functions_supported": False,
        "vision_supported": False,
        "server_port": 8000,
        "server_host": "localhost",
    },
    
    ModelProvider.TOGETHER: {
        "api_base": "https://api.together.xyz/v1",
        "default_model": None,  # Set based on preferences
        "request_timeout": 60,
        "streaming_supported": True,
        "functions_supported": False,
        "vision_supported": False,
        "max_retries": 3,
    },
    
    ModelProvider.HUGGINGFACE: {
        "api_base": "https://api-inference.huggingface.co/models",
        "default_model": None,  # Set based on preferences
        "request_timeout": 120,
        "streaming_supported": False,
        "functions_supported": False,
        "vision_supported": True,
        "max_retries": 3,
    },
    
    ModelProvider.REPLICATE: {
        "api_base": "https://api.replicate.com/v1",
        "default_model": None,  # Set based on preferences
        "request_timeout": 120,
        "streaming_supported": True,
        "functions_supported": False,
        "vision_supported": True,
        "max_retries": 3,
    },
    
    ModelProvider.COHERE: {
        "api_base": "https://api.cohere.ai/v1",
        "default_model": "command-r-plus",
        "request_timeout": 60,
        "streaming_supported": True,
        "functions_supported": True,
        "vision_supported": False,
        "max_retries": 3,
        "default_headers": {
            "Content-Type": "application/json",
        },
    },
}

# Model roles for specific tasks
MODEL_ROLES = {
    "general": {
        "primary": ["gpt-4o", "claude-3-sonnet", "gemini-1.5-pro"],
        "backup": ["gpt-3.5-turbo", "claude-3-haiku", "mistral-small"],
        "description": "Models suitable for general-purpose tasks.",
    },
    "reasoning": {
        "primary": ["gpt-4o", "claude-3-opus", "mistral-large"],
        "backup": ["claude-3-sonnet", "gemini-1.5-pro"],
        "description": "Models with strong reasoning capabilities.",
    },
    "creative": {
        "primary": ["claude-3-opus", "gpt-4o"],
        "backup": ["claude-3-sonnet", "gemini-1.5-pro"],
        "description": "Models good at creative writing and generation.",
    },
    "code": {
        "primary": ["gpt-4o", "claude-3-opus", "mistral-large"],
        "backup": ["gpt-3.5-turbo", "claude-3-sonnet"],
        "description": "Models specialized for code generation and analysis.",
    },
    "vision": {
        "primary": ["gpt-4o", "gpt-4-vision-preview", "claude-3-opus"],
        "backup": ["claude-3-sonnet", "gemini-1.5-pro"],
        "description": "Models that can process and analyze images.",
    },
    "long_context": {
        "primary": ["claude-3-opus", "claude-3-sonnet", "gemini-1.5-pro"],
        "backup": ["gpt-4o"],
        "description": "Models with large context windows for processing long documents.",
    },
    "high_throughput": {
        "primary": ["gpt-3.5-turbo", "claude-3-haiku", "gemini-1.5-flash"],
        "backup": ["mistral-small"],
        "description": "Faster, cost-effective models for high-volume processing.",
    },
    "local_deployment": {
        "primary": ["llama-3-8b-instruct"],
        "backup": ["llama-3-70b-instruct"],
        "description": "Models that can be run locally without API calls.",
    },
}

# Prompt templates for different providers and models
PROMPT_TEMPLATES = {
    "openai_chat": {
        "system": {"role": "system", "content": "{system_message}"},
        "user": {"role": "user", "content": "{message}"},
        "assistant": {"role": "assistant", "content": "{message}"},
    },
    "anthropic_chat": {
        "system": {"role": "system", "content": "{system_message}"},
        "user": {"role": "user", "content": "{message}"},
        "assistant": {"role": "assistant", "content": "{message}"},
    },
    "google_chat": {
        "system": {"role": "system", "content": "{system_message}"},
        "user": {"role": "user", "content": "{message}"},
        "assistant": {"role": "model", "content": "{message}"},
    },
    "mistral_chat": {
        "system": {"role": "system", "content": "{system_message}"},
        "user": {"role": "user", "content": "{message}"},
        "assistant": {"role": "assistant", "content": "{message}"},
    },
    "llama_chat": {
        "system": "<|system|>\n{system_message}</s>",
        "user": "<|user|>\n{message}</s>",
        "assistant": "<|assistant|>\n{message}</s>",
    },
}

# Function to get model configuration by name
def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model by name.
    
    Args:
        model_name: The name of the model
        
    Returns:
        Dictionary with model configuration
    """
    for provider_config in PROVIDER_CONFIGS.values():
        models = provider_config.get("models", {})
        if model_name in models:
            return models[model_name].copy()
    
    return {}

# Function to get provider configuration
def get_provider_config(provider: ModelProvider) -> Dict[str, Any]:
    """
    Get configuration for a specific provider.
    
    Args:
        provider: The model provider
        
    Returns:
        Dictionary with provider configuration
    """
    if provider in PROVIDER_CONFIGS:
        return PROVIDER_CONFIGS[provider].copy()
    
    return {}

# Function to get default model for a provider
def get_default_model(provider: ModelProvider) -> str:
    """
    Get the default model for a specific provider.
    
    Args:
        provider: The model provider
        
    Returns:
        Name of the default model
    """
    provider_config = get_provider_config(provider)
    return provider_config.get("default_model", "")

# Function to get models for a specific role
def get_models_for_role(role: str) -> List[str]:
    """
    Get models recommended for a specific role.
    
    Args:
        role: The role name
        
    Returns:
        List of model names
    """
    if role in MODEL_ROLES:
        role_config = MODEL_ROLES[role]
        return role_config.get("primary", []) + role_config.get("backup", [])
    
    return []

# Function to get primary models for a specific role
def get_primary_models_for_role(role: str) -> List[str]:
    """
    Get primary models recommended for a specific role.
    
    Args:
        role: The role name
        
    Returns:
        List of model names
    """
    if role in MODEL_ROLES:
        return MODEL_ROLES[role].get("primary", [])
    
    return []

# Function to get provider for a model
def get_provider_for_model(model_name: str) -> Optional[ModelProvider]:
    """
    Get the provider for a specific model.
    
    Args:
        model_name: The name of the model
        
    Returns:
        Provider enum value or None if not found
    """
    for provider, provider_config in PROVIDER_CONFIGS.items():
        models = provider_config.get("models", {})
        if model_name in models:
            return provider
    
    return None

# Function to calculate cost for a request
def calculate_request_cost(model_name: str, input_tokens: int, output_tokens: int = 0) -> float:
    """
    Calculate the cost of a model request.
    
    Args:
        model_name: The name of the model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Estimated cost in USD
    """
    model_config = get_model_config(model_name)
    
    if not model_config:
        return 0.0
    
    model_type = model_config.get("model_type")
    
    if model_type == ModelType.EMBEDDING:
        return (input_tokens / 1000) * model_config.get("cost_per_1k", 0)
    
    input_cost = (input_tokens / 1000) * model_config.get("input_cost_per_1k", 0)
    output_cost = (output_tokens / 1000) * model_config.get("output_cost_per_1k", 0)
    
    return input_cost + output_cost

# Function to find models with specific capabilities
def find_models_with_capability(capability: str) -> List[str]:
    """
    Find models with a specific capability.
    
    Args:
        capability: The capability to filter by (e.g., "vision", "functions")
        
    Returns:
        List of model names
    """
    supported_models = []
    
    capability_key = f"supports_{capability}"
    
    for provider_config in PROVIDER_CONFIGS.values():
        models = provider_config.get("models", {})
        
        for model_name, model_config in models.items():
            if model_config.get(capability_key, False):
                supported_models.append(model_name)
    
    return supported_models

# Function to find models by context size
def find_models_by_context_size(min_tokens: int) -> List[str]:
    """
    Find models with at least the specified context size.
    
    Args:
        min_tokens: Minimum number of context tokens required
        
    Returns:
        List of model names
    """
    supported_models = []
    
    for provider_config in PROVIDER_CONFIGS.values():
        models = provider_config.get("models", {})
        
        for model_name, model_config in models.items():
            context_length = model_config.get("context_length", 0)
            if context_length >= min_tokens:
                supported_models.append(model_name)
    
    return supported_models

# Function to get prompt format for a model
def get_prompt_format_for_model(model_name: str) -> Dict[str, Any]:
    """
    Get the prompt format to use for a specific model.
    
    Args:
        model_name: The name of the model
        
    Returns:
        Dictionary with prompt format templates
    """
    provider = get_provider_for_model(model_name)
    
    if provider == ModelProvider.OPENAI:
        return PROMPT_TEMPLATES["openai_chat"]
    elif provider == ModelProvider.ANTHROPIC:
        return PROMPT_TEMPLATES["anthropic_chat"]
    elif provider == ModelProvider.GOOGLE:
        return PROMPT_TEMPLATES["google_chat"]
    elif provider == ModelProvider.MISTRAL:
        return PROMPT_TEMPLATES["mistral_chat"]
    elif provider == ModelProvider.LLAMA:
        return PROMPT_TEMPLATES["llama_chat"]
    
    # Default to OpenAI format if unknown
    return PROMPT_TEMPLATES["openai_chat"]

# Function to validate model availability
def is_model_available(model_name: str, check_api: bool = False) -> bool:
    """
    Check if a model is available in the configuration.
    
    Args:
        model_name: The name of the model
        check_api: Whether to check API connectivity
        
    Returns:
        True if available, False otherwise
    """
    # Check if model exists in configuration
    model_config = get_model_config(model_name)
    
    if not model_config:
        return False
    
    # If not checking API, just confirm it's in config
    if not check_api:
        return True
    
    # Check API connectivity would be implemented here
    # This is a placeholder that would need to be implemented
    return True

# Function to get global model configuration
def get_global_config() -> Dict[str, Any]:
    """
    Get global model configuration.
    
    Returns:
        Dictionary with global configuration
    """
    return GLOBAL_MODEL_CONFIG.copy()

# Function to update model configuration
def update_model_config(model_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration for a specific model.
    
    This function does not persist changes to disk. It only updates the in-memory configuration.
    
    Args:
        model_name: The name of the model to update
        updates: Dictionary with configuration updates
        
    Returns:
        Updated configuration dictionary
    """
    provider = get_provider_for_model(model_name)
    
    if not provider:
        return {}
    
    provider_config = PROVIDER_CONFIGS[provider]
    models = provider_config.get("models", {})
    
    if model_name not in models:
        return {}
    
    model_config = models[model_name].copy()
    model_config.update(updates)
    models[model_name] = model_config
    
    return model_config

# Function to validate model configuration
def validate_model_config(model_name: str, config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration for a specific model.
    
    Args:
        model_name: The name of the model
        config: The configuration to validate
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Basic validation
    if not isinstance(config, dict):
        errors.append("Configuration must be a dictionary")
        return errors
    
    # Required fields for all models
    required_fields = ["model_type", "model_size"]
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Type-specific validation
    model_type = config.get("model_type")
    
    if model_type == ModelType.CHAT:
        if "context_length" not in config:
            errors.append("Chat models require a 'context_length' field")
    
    elif model_type == ModelType.EMBEDDING:
        if "dimensions" not in config:
            errors.append("Embedding models require a 'dimensions' field")
    
    return errors

# Function to find the most cost-effective model for a task
def find_cost_effective_model(task_type: str, min_quality: float = 0.7) -> str:
    """
    Find the most cost-effective model for a specific task type.
    
    Args:
        task_type: The type of task
        min_quality: Minimum quality threshold (0-1)
        
    Returns:
        Model name
    """
    # This is a placeholder implementation
    # In a real implementation, this would consider performance benchmarks
    # and cost data to make an intelligent recommendation
    
    if task_type == "general":
        if min_quality > 0.8:
            return "claude-3-sonnet"
        else:
            return "gpt-3.5-turbo"
    elif task_type == "code":
        if min_quality > 0.8:
            return "gpt-4o"
        else:
            return "gpt-3.5-turbo"
    elif task_type == "vision":
        return "gpt-4o"
    
    # Default fallback
    return "gpt-3.5-turbo" 