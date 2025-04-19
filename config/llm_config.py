"""
LLM configuration for Telos AI.

This module provides configuration settings for different language models,
including model selection, inference parameters, and provider settings.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

# LLM providers
class LLMProvider(Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    MISTRAL = "mistral"
    COHERE = "cohere"
    TOGETHER = "together"

# Model sizes
class ModelSize(Enum):
    """Enumeration of model sizes."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XL = "xl"
    XXL = "xxl"

# Task types that LLMs can be used for
class LLMTaskType(Enum):
    """Enumeration of task types that LLMs can be used for."""
    CHAT = "chat"
    COMPLETION = "completion"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    CODE_GENERATION = "code_generation"
    EMBEDDING = "embedding"
    PLANNING = "planning"
    RESEARCH = "research"

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: LLMProvider
    model_id: str
    size: ModelSize
    max_tokens: int
    supported_tasks: List[LLMTaskType]
    context_length: int
    cost_per_1k_tokens: float
    input_cost_per_1k_tokens: Optional[float] = None
    output_cost_per_1k_tokens: Optional[float] = None
    avg_tokens_per_second: Optional[float] = None
    supports_functions: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False
    local_path: Optional[str] = None
    api_base: Optional[str] = None
    required_vram_gb: Optional[int] = None

# Default models by provider
DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-4.1",
    LLMProvider.ANTHROPIC: "claude-3-opus-20240229",
    LLMProvider.GOOGLE: "gemini-1.5-pro",
    LLMProvider.LOCAL: "llama-3-70b-instruct.Q5_K_M",
    LLMProvider.HUGGINGFACE: "meta-llama/Llama-3-70b-chat-hf",
    LLMProvider.MISTRAL: "mistral-large-latest",
    LLMProvider.COHERE: "command-r-plus",
    LLMProvider.TOGETHER: "meta-llama/Llama-3-70b-chat-hf",
}

# Model configurations
MODELS = {
    # OpenAI Models
    "gpt-4.1": ModelConfig(
        provider=LLMProvider.OPENAI,
        model_id="gpt-4.1",
        size=ModelSize.XXL,
        max_tokens=4096,
        supported_tasks=[task for task in LLMTaskType],
        context_length=128000,
        cost_per_1k_tokens=0.0,  # Combined rate
        input_cost_per_1k_tokens=0.005,
        output_cost_per_1k_tokens=0.015,
        avg_tokens_per_second=40,
        supports_functions=True,
        supports_vision=True,
        supports_json_mode=True,
    ),
    "gpt-4-turbo": ModelConfig(
        provider=LLMProvider.OPENAI,
        model_id="gpt-4-turbo",
        size=ModelSize.XXL,
        max_tokens=4096,
        supported_tasks=[task for task in LLMTaskType],
        context_length=128000,
        cost_per_1k_tokens=0.0,  # Combined rate
        input_cost_per_1k_tokens=0.01,
        output_cost_per_1k_tokens=0.03,
        avg_tokens_per_second=35,
        supports_functions=True,
        supports_vision=True,
        supports_json_mode=True,
    ),
    "gpt-3.5-turbo": ModelConfig(
        provider=LLMProvider.OPENAI,
        model_id="gpt-3.5-turbo",
        size=ModelSize.LARGE,
        max_tokens=4096,
        supported_tasks=[task for task in LLMTaskType],
        context_length=16385,
        cost_per_1k_tokens=0.0,  # Combined rate
        input_cost_per_1k_tokens=0.0005,
        output_cost_per_1k_tokens=0.0015,
        avg_tokens_per_second=45,
        supports_functions=True,
        supports_vision=False,
        supports_json_mode=True,
    ),
    "text-embedding-3-large": ModelConfig(
        provider=LLMProvider.OPENAI,
        model_id="text-embedding-3-large",
        size=ModelSize.LARGE,
        max_tokens=8191,
        supported_tasks=[LLMTaskType.EMBEDDING],
        context_length=8191,
        cost_per_1k_tokens=0.00013,
        avg_tokens_per_second=60,
        supports_functions=False,
        supports_vision=False,
        supports_json_mode=False,
    ),

    # Anthropic Models
    "claude-3-opus-20240229": ModelConfig(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-3-opus-20240229",
        size=ModelSize.XXL,
        max_tokens=4096,
        supported_tasks=[task for task in LLMTaskType if task != LLMTaskType.EMBEDDING],
        context_length=200000,
        cost_per_1k_tokens=0.0,  # Combined rate
        input_cost_per_1k_tokens=0.015,
        output_cost_per_1k_tokens=0.075,
        avg_tokens_per_second=30,
        supports_functions=True,
        supports_vision=True,
        supports_json_mode=True,
    ),
    "claude-3-sonnet-20240229": ModelConfig(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-3-sonnet-20240229",
        size=ModelSize.XL,
        max_tokens=4096,
        supported_tasks=[task for task in LLMTaskType if task != LLMTaskType.EMBEDDING],
        context_length=200000,
        cost_per_1k_tokens=0.0,  # Combined rate
        input_cost_per_1k_tokens=0.003,
        output_cost_per_1k_tokens=0.015,
        avg_tokens_per_second=35,
        supports_functions=True,
        supports_vision=True,
        supports_json_mode=True,
    ),
    "claude-3-haiku-20240307": ModelConfig(
        provider=LLMProvider.ANTHROPIC,
        model_id="claude-3-haiku-20240307",
        size=ModelSize.LARGE,
        max_tokens=4096,
        supported_tasks=[task for task in LLMTaskType if task != LLMTaskType.EMBEDDING],
        context_length=200000,
        cost_per_1k_tokens=0.0,  # Combined rate
        input_cost_per_1k_tokens=0.00025,
        output_cost_per_1k_tokens=0.00125,
        avg_tokens_per_second=40,
        supports_functions=True,
        supports_vision=True,
        supports_json_mode=True,
    ),

    # Local Models
    "llama-3-70b-instruct.Q5_K_M": ModelConfig(
        provider=LLMProvider.LOCAL,
        model_id="llama-3-70b-instruct.Q5_K_M",
        size=ModelSize.XXL,
        max_tokens=4096,
        supported_tasks=[
            LLMTaskType.CHAT, 
            LLMTaskType.COMPLETION, 
            LLMTaskType.REASONING,
            LLMTaskType.SUMMARIZATION,
            LLMTaskType.CODE_GENERATION,
        ],
        context_length=8192,
        cost_per_1k_tokens=0.0,
        avg_tokens_per_second=12,
        required_vram_gb=30,
        local_path="/models/llama-3-70b-instruct.Q5_K_M.gguf",
    ),
    "mistral-7b-instruct.Q5_K_M": ModelConfig(
        provider=LLMProvider.LOCAL,
        model_id="mistral-7b-instruct.Q5_K_M",
        size=ModelSize.MEDIUM,
        max_tokens=4096,
        supported_tasks=[
            LLMTaskType.CHAT, 
            LLMTaskType.COMPLETION, 
            LLMTaskType.SUMMARIZATION,
        ],
        context_length=8192,
        cost_per_1k_tokens=0.0,
        avg_tokens_per_second=25,
        required_vram_gb=6,
        local_path="/models/mistral-7b-instruct.Q5_K_M.gguf",
    ),
}

# Default inference parameters
DEFAULT_INFERENCE_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 1024,
    "stop_sequences": [],
    "timeout": 120,  # seconds
    "stream": True,
}

# Task-specific inference parameters
TASK_INFERENCE_PARAMS = {
    LLMTaskType.CHAT: {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1024,
    },
    LLMTaskType.COMPLETION: {
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 2048,
    },
    LLMTaskType.REASONING: {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 4096,
    },
    LLMTaskType.SUMMARIZATION: {
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 1024,
    },
    LLMTaskType.CLASSIFICATION: {
        "temperature": 0.1,
        "top_p": 0.95,
        "max_tokens": 256,
    },
    LLMTaskType.CODE_GENERATION: {
        "temperature": 0.2,
        "top_p": 0.95,
        "frequency_penalty": 0.2,
        "max_tokens": 4096,
    },
    LLMTaskType.EMBEDDING: {
        "temperature": 0.0,
        "max_tokens": 0,
    },
    LLMTaskType.PLANNING: {
        "temperature": 0.4,
        "top_p": 0.95,
        "max_tokens": 4096,
    },
    LLMTaskType.RESEARCH: {
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 4096,
    },
}

# API provider configurations
PROVIDER_CONFIGS = {
    LLMProvider.OPENAI: {
        "api_type": "openai",
        "api_version": "2023-05-15",
        "max_parallel_requests": 20,
        "rate_limit_rpm": 10000,
        "max_retries": 5,
        "retry_delay": 2,  # seconds
    },
    LLMProvider.ANTHROPIC: {
        "api_type": "anthropic",
        "api_version": "2023-06-01",
        "max_parallel_requests": 15,
        "rate_limit_rpm": 5000,
        "max_retries": 5,
        "retry_delay": 3,  # seconds
    },
    LLMProvider.GOOGLE: {
        "api_type": "google",
        "max_parallel_requests": 15,
        "rate_limit_rpm": 3000,
        "max_retries": 3,
        "retry_delay": 2,  # seconds
    },
    LLMProvider.LOCAL: {
        "api_type": "local",
        "max_parallel_requests": 1,
        "rate_limit_rpm": 60,
        "max_retries": 1,
        "retry_delay": 1,  # seconds
    },
    LLMProvider.HUGGINGFACE: {
        "api_type": "huggingface",
        "max_parallel_requests": 5,
        "rate_limit_rpm": 300,
        "max_retries": 3,
        "retry_delay": 2,  # seconds
    },
    LLMProvider.MISTRAL: {
        "api_type": "mistral",
        "max_parallel_requests": 10,
        "rate_limit_rpm": 600,
        "max_retries": 3,
        "retry_delay": 2,  # seconds
    },
    LLMProvider.COHERE: {
        "api_type": "cohere",
        "max_parallel_requests": 10,
        "rate_limit_rpm": 600,
        "max_retries": 3,
        "retry_delay": 2,  # seconds
    },
    LLMProvider.TOGETHER: {
        "api_type": "together",
        "max_parallel_requests": 5,
        "rate_limit_rpm": 300,
        "max_retries": 3,
        "retry_delay": 2,  # seconds
    },
}

# Fallback configuration
FALLBACK_CONFIG = {
    "enabled": True,
    "max_attempts": 3,
    "fallback_models": {
        "gpt-4.1": ["gpt-4-turbo", "claude-3-opus-20240229"],
        "claude-3-opus-20240229": ["gpt-4.1", "claude-3-sonnet-20240229"],
        "gpt-3.5-turbo": ["claude-3-haiku-20240307", "mistral-7b-instruct.Q5_K_M"],
    }
}

# Caching configuration
CACHE_CONFIG = {
    "enabled": True,
    "cache_type": "redis",  # Options: "memory", "redis", "disk"
    "expiration_time": 86400,  # 24 hours in seconds
    "max_cache_size": 10 * 1024 * 1024 * 1024,  # 10 GB
    "include_system_prompt_in_key": True,
}

# Prompt template configuration
PROMPT_TEMPLATES = {
    "system_message": "You are Telos, an AI assistant. {custom_instructions}",
    "chat": "{message}",
    "reasoning": "Step-by-step reasoning for: {query}",
    "summarization": "Summarize the following text: {text}",
    "classification": "Classify the following text into one of these categories ({categories}): {text}",
    "code_generation": "Generate code to solve the following problem: {problem}",
}

def get_default_model(provider: LLMProvider = None) -> str:
    """
    Get the default model for a provider.
    
    Args:
        provider: The provider to get the default model for
        
    Returns:
        Model ID string
    """
    if provider is None:
        return DEFAULT_MODELS[LLMProvider.OPENAI]
    
    if provider in DEFAULT_MODELS:
        return DEFAULT_MODELS[provider]
    
    # If provider not found, return OpenAI default
    return DEFAULT_MODELS[LLMProvider.OPENAI]

def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """
    Get configuration for a specific model.
    
    Args:
        model_id: The model ID to get configuration for
        
    Returns:
        ModelConfig object if found, None otherwise
    """
    if model_id in MODELS:
        return MODELS[model_id]
    return None

def get_inference_params(task_type: LLMTaskType = None, custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get inference parameters for a specific task type.
    
    Args:
        task_type: The task type to get parameters for
        custom_params: Custom parameters to override defaults
        
    Returns:
        Dictionary with inference parameters
    """
    # Start with default parameters
    params = DEFAULT_INFERENCE_PARAMS.copy()
    
    # Override with task-specific parameters if available
    if task_type is not None and task_type in TASK_INFERENCE_PARAMS:
        params.update(TASK_INFERENCE_PARAMS[task_type])
    
    # Override with custom parameters if provided
    if custom_params:
        params.update(custom_params)
    
    return params

def get_provider_config(provider: LLMProvider) -> Dict[str, Any]:
    """
    Get configuration for a specific provider.
    
    Args:
        provider: The provider to get configuration for
        
    Returns:
        Dictionary with provider configuration
    """
    if provider in PROVIDER_CONFIGS:
        return PROVIDER_CONFIGS[provider].copy()
    
    # If provider not found, return empty config
    return {}

def estimate_cost(
    model_id: str, 
    input_tokens: int, 
    output_tokens: int = None
) -> Optional[float]:
    """
    Estimate the cost of an LLM call.
    
    Args:
        model_id: The model ID
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens (if None, equals input_tokens)
        
    Returns:
        Estimated cost in USD, None if model not found
    """
    model_config = get_model_config(model_id)
    if not model_config:
        return None
    
    if output_tokens is None:
        output_tokens = input_tokens
    
    # If the model has separate input/output costs
    if model_config.input_cost_per_1k_tokens is not None and model_config.output_cost_per_1k_tokens is not None:
        input_cost = (input_tokens / 1000) * model_config.input_cost_per_1k_tokens
        output_cost = (output_tokens / 1000) * model_config.output_cost_per_1k_tokens
        return input_cost + output_cost
    
    # Otherwise use the combined cost
    return ((input_tokens + output_tokens) / 1000) * model_config.cost_per_1k_tokens

def get_models_for_task(task_type: LLMTaskType) -> List[str]:
    """
    Get a list of models that support a specific task type.
    
    Args:
        task_type: The task type to find models for
        
    Returns:
        List of model IDs
    """
    supported_models = []
    
    for model_id, config in MODELS.items():
        if task_type in config.supported_tasks:
            supported_models.append(model_id)
    
    return supported_models

def get_fallback_models(model_id: str) -> List[str]:
    """
    Get fallback models for a specific model.
    
    Args:
        model_id: The model ID to get fallbacks for
        
    Returns:
        List of fallback model IDs
    """
    if not FALLBACK_CONFIG["enabled"]:
        return []
    
    if model_id in FALLBACK_CONFIG["fallback_models"]:
        return FALLBACK_CONFIG["fallback_models"][model_id]
    
    # If no specific fallbacks defined, return models of the same size
    model_config = get_model_config(model_id)
    if not model_config:
        return []
    
    fallbacks = []
    for other_id, other_config in MODELS.items():
        if other_id != model_id and other_config.size == model_config.size:
            fallbacks.append(other_id)
    
    return fallbacks

def get_prompt_template(template_name: str) -> Optional[str]:
    """
    Get a prompt template.
    
    Args:
        template_name: The template name
        
    Returns:
        Template string if found, None otherwise
    """
    if template_name in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[template_name]
    return None

def validate_model_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a model configuration dictionary.
    
    Args:
        config: The model configuration to validate
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check required fields
    required_fields = ["provider", "model_id", "size", "max_tokens", "supported_tasks", "context_length"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate types
    if "provider" in config and not isinstance(config["provider"], LLMProvider):
        errors.append("provider must be a LLMProvider enum value")
    
    if "size" in config and not isinstance(config["size"], ModelSize):
        errors.append("size must be a ModelSize enum value")
    
    if "max_tokens" in config and not isinstance(config["max_tokens"], int):
        errors.append("max_tokens must be an integer")
    
    if "supported_tasks" in config:
        if not isinstance(config["supported_tasks"], list):
            errors.append("supported_tasks must be a list")
        else:
            for task in config["supported_tasks"]:
                if not isinstance(task, LLMTaskType):
                    errors.append(f"Task {task} must be a LLMTaskType enum value")
    
    # Validate ranges
    if "max_tokens" in config and config["max_tokens"] <= 0:
        errors.append("max_tokens must be positive")
    
    if "context_length" in config and config["context_length"] <= 0:
        errors.append("context_length must be positive")
    
    return errors 