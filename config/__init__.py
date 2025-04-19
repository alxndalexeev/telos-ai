"""
Configuration module for Telos AI.

This module centralizes configuration from various submodules.
"""

# Import all configuration from submodules
from config.api_keys import *
from config.vector_db import *
from config.llm import *
from config.system import *

# Re-export everything
__all__ = []

# API keys configuration
__all__ += [
    'OPENAI_API_KEY',
    'TAVILY_API_KEY',
    'PINECONE_API_KEY',
    'PINECONE_ENVIRONMENT',
]

# Vector DB configuration
__all__ += [
    'PINECONE_INDEX_NAME',
    'PINECONE_NAMESPACE',
    'EMBEDDING_MODEL',
    'EMBEDDING_DIMENSION',
    'USE_VECTOR_DB',
    'VECTOR_DB_FULL_REFRESH_DAYS',
]

# LLM configuration
__all__ += [
    'PLANNER_LLM_MODEL',
    'EXECUTOR_LLM_MODEL',
    'REVIEWER_LLM_MODEL',
    'MAX_OUTPUT_TOKENS',
    'TEMPERATURE',
]

# System configuration
__all__ += [
    'MEMORY_DIR',
    'HEARTBEAT_INTERVAL',
    'DEBUG_MODE',
    'LOG_LEVEL',
]
