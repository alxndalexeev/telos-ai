"""
Memory configuration for Telos AI.

This module provides configuration settings for the memory subsystem,
including different memory types, storage options, and retrieval mechanisms.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

class MemoryType(Enum):
    """Enumeration of supported memory types."""
    SHORT_TERM = "short_term"
    CONVERSATION = "conversation"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    ENTITY = "entity"
    META = "meta"

class StorageType(Enum):
    """Enumeration of supported storage types."""
    IN_MEMORY = "in_memory"
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    REDIS = "redis"
    JSON = "json"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    MILVUS = "milvus"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    CUSTOM = "custom"

class IndexMethod(Enum):
    """Enumeration of supported index methods."""
    EXACT = "exact"
    HNSW = "hnsw"
    IVF = "ivf"
    PQ = "pq"
    LSH = "lsh"
    ANNOY = "annoy"
    FAISS = "faiss"
    NONE = "none"

class DistanceMetric(Enum):
    """Enumeration of supported distance metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"
    JACCARD = "jaccard"
    HAMMING = "hamming"

class Embedder(Enum):
    """Enumeration of supported embedding models."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    CUSTOM = "custom"

# Global memory settings
GLOBAL_MEMORY_CONFIG = {
    "enabled": True,
    "default_memory_type": MemoryType.CONVERSATION,
    "default_storage_type": StorageType.SQLITE,
    "storage_path": "./data/memory",
    "vector_dimensions": 1536,  # Default for OpenAI embeddings
    "auto_prune": True,
    "max_memory_entries": 100000,
    "ttl_days": 90,  # Default time-to-live for memories in days
    "backup_enabled": True,
    "backup_interval_hours": 24,
    "backup_max_copies": 5,
    "encryption_enabled": False,
    "encryption_key_path": None,
    "enable_search_filtering": True,
    "enable_deduplication": True,
    "deduplication_threshold": 0.92,  # Similarity threshold for deduplication
    "consolidation_enabled": True,  # Periodically consolidate similar memories
    "enable_memory_reranking": True,
    "log_memory_operations": True,
    "context_window_size": 10,  # Number of entries to include in context window
}

# Short-term memory configuration
SHORT_TERM_MEMORY_CONFIG = {
    "enabled": True,
    "storage_type": StorageType.IN_MEMORY,
    "max_items": 1000,
    "ttl_minutes": 30,
    "recency_bias": 0.8,  # Higher values favor more recent memories
    "auto_archive": True,  # Automatically archive to long-term memory
    "archive_threshold_minutes": 15,
    "priority_weighting": True,  # Weight items by importance
}

# Conversation memory configuration
CONVERSATION_MEMORY_CONFIG = {
    "enabled": True,
    "storage_type": StorageType.SQLITE,
    "max_messages": 100,
    "include_system_messages": True,
    "include_function_calls": True,
    "include_timestamps": True,
    "include_metadata": True,
    "summarization": {
        "enabled": True,
        "max_summary_items": 50,
        "summary_trigger_length": 20,  # Summarize after this many new messages
        "summary_refresh_interval": 10,  # Refresh summary every X new messages
        "include_summary_in_context": True,
    },
    "token_tracking": True,
    "persistent": True,
    "auto_save": True,
    "save_interval_seconds": 60,
}

# Long-term memory configuration
LONG_TERM_MEMORY_CONFIG = {
    "enabled": True,
    "storage_type": StorageType.SQLITE,
    "embedder": Embedder.OPENAI,
    "embedding_model": "text-embedding-3-small",
    "embedding_dimensions": 1536,
    "embedding_batch_size": 100,
    "distance_metric": DistanceMetric.COSINE,
    "top_k": 5,  # Default number of results to return
    "score_threshold": 0.75,  # Minimum similarity score to include
    "pre_filter_with_keywords": True,
    "max_tokens_per_doc": 4000,
    "support_hybrid_search": True,  # Combine vector and keyword search
    "persist_embeddings": True,
    "reindex_interval_hours": 24,
    "scheduled_maintenance": True,
    "compression_enabled": False,
    "enable_cache": True,
    "cache_ttl_minutes": 30,
}

# Episodic memory configuration
EPISODIC_MEMORY_CONFIG = {
    "enabled": True,
    "storage_type": StorageType.SQLITE,
    "max_episodes": 1000,
    "episode_expiry_days": 90,
    "emotions_tracking": True,
    "context_tracking": True,
    "timestamp_precision": "second",
    "location_tracking": False,
    "categorical_tagging": True,
    "importance_scoring": True,
    "recall_recency_bias": 0.7,
    "recall_importance_bias": 0.3,
    "auto_link_entities": True,
    "minimum_episode_importance": 0.2,  # Filter out unimportant episodes
}

# Semantic memory configuration
SEMANTIC_MEMORY_CONFIG = {
    "enabled": True,
    "storage_type": StorageType.CHROMA,
    "concept_extraction": True,
    "concept_linking": True,
    "concept_hierarchy": True,
    "max_concepts": 10000,
    "unknown_concept_handling": "create",  # Options: "create", "ignore", "prompt"
    "concept_similarity_threshold": 0.85,
    "automated_concept_refinement": True,
    "concept_verification": True,
    "concept_decay_rate": 0.01,  # Rate at which unused concepts decay in importance
    "external_knowledge_integration": False,
}

# Procedural memory configuration
PROCEDURAL_MEMORY_CONFIG = {
    "enabled": True,
    "storage_type": StorageType.SQLITE,
    "max_procedures": 1000,
    "version_control": True,
    "execution_tracking": True,
    "success_rate_tracking": True,
    "auto_optimize": True,
    "optimization_threshold": 0.7,  # Success rate below which to optimize
    "track_dependencies": True,
    "allow_external_execution": False,
    "execution_timeout_seconds": 30,
    "sandboxed_execution": True,
}

# Working memory configuration
WORKING_MEMORY_CONFIG = {
    "enabled": True,
    "storage_type": StorageType.IN_MEMORY,
    "capacity": 7,  # Classic "7 plus or minus 2" working memory model
    "ttl_minutes": 5,
    "prioritization": True,
    "chunking": True,
    "max_chunk_size": 4,
    "auto_refresh": True,
    "refresh_interval_seconds": 30,
    "attention_weighting": True,
}

# Entity memory configuration
ENTITY_MEMORY_CONFIG = {
    "enabled": True,
    "storage_type": StorageType.SQLITE,
    "max_entities": 10000,
    "entity_types": ["person", "organization", "location", "concept", "object", "event"],
    "attribute_tracking": True,
    "relationship_tracking": True,
    "auto_entity_detection": True,
    "entity_merging": True,
    "entity_merging_threshold": 0.9,
    "entity_importance_scoring": True,
    "entity_decay_rate": 0.005,  # Rate at which unused entities decay in importance
}

# Meta memory configuration
META_MEMORY_CONFIG = {
    "enabled": True,
    "storage_type": StorageType.SQLITE,
    "track_memory_usage": True,
    "track_memory_performance": True,
    "track_memory_accuracy": True,
    "optimization_enabled": True,
    "optimization_interval_hours": 24,
    "self_reflection_enabled": True,
    "reflection_interval_hours": 24,
    "performance_metrics_ttl_days": 90,
}

# Storage backend configurations
STORAGE_CONFIGS = {
    StorageType.IN_MEMORY: {
        "persistent": False,
    },
    
    StorageType.SQLITE: {
        "database_path": "./data/memory/telos_memory.db",
        "journal_mode": "WAL",
        "synchronous": "NORMAL",
        "temp_store": "MEMORY",
        "cache_size": 10000,  # Pages in cache
        "mmap_size": 0,  # 0 for auto
        "busy_timeout": 5000,  # ms
        "backup_enabled": True,
    },
    
    StorageType.POSTGRES: {
        "host": "localhost",
        "port": 5432,
        "database": "telos_memory",
        "user": "telos",
        "password": "",
        "ssl_mode": "prefer",
        "min_connections": 1,
        "max_connections": 10,
        "connection_timeout": 30,
        "use_prepared_statements": True,
    },
    
    StorageType.REDIS: {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": "",
        "ssl": False,
        "encoding": "utf-8",
        "socket_timeout": 5,
        "socket_connect_timeout": 5,
        "connection_pool_size": 10,
        "use_sentinel": False,
        "sentinel_hosts": [],
        "sentinel_master": "mymaster",
    },
    
    StorageType.JSON: {
        "directory_path": "./data/memory/json",
        "pretty_print": True,
        "compression": False,
        "index_file": "memory_index.json",
        "auto_backup": True,
    },
    
    StorageType.CHROMA: {
        "persist_directory": "./data/memory/chroma",
        "collection_name": "telos_memory",
        "distance_metric": DistanceMetric.COSINE,
        "anonymized_telemetry": False,
        "allow_reset": True,
    },
    
    StorageType.PINECONE: {
        "api_key": "",
        "environment": "us-west1-gcp",
        "index_name": "telos-memory",
        "namespace": "default",
        "dimension": 1536,
        "metric": DistanceMetric.COSINE,
        "pod_type": "p1",
        "replicas": 1,
        "shards": 1,
        "metadata_filtering": True,
    },
    
    StorageType.MILVUS: {
        "host": "localhost",
        "port": 19530,
        "collection_name": "telos_memory",
        "dimension": 1536,
        "index_type": IndexMethod.HNSW,
        "metric_type": DistanceMetric.COSINE,
        "auto_id": True,
        "timeout": 60,  # seconds
        "consistency_level": "Strong",
    },
    
    StorageType.QDRANT: {
        "url": "http://localhost:6333",
        "api_key": "",
        "collection_name": "telos_memory",
        "vector_size": 1536,
        "distance": DistanceMetric.COSINE,
        "shard_number": 1,
        "replication_factor": 1,
        "write_consistency_factor": 1,
        "on_disk_payload": True,
    },
    
    StorageType.WEAVIATE: {
        "url": "http://localhost:8080",
        "api_key": "",
        "class_name": "TelosMemory",
        "batch_size": 100,
        "batch_timeout_retries": 3,
        "batch_timeout_seconds": 60,
        "vector_index_type": IndexMethod.HNSW,
        "vector_index_config": {
            "maxConnections": 64,
            "efConstruction": 128,
        },
    },
}

# Embedding model configurations
EMBEDDING_CONFIGS = {
    Embedder.OPENAI: {
        "default_model": "text-embedding-3-small",
        "models": {
            "text-embedding-3-small": {
                "dimensions": 1536,
                "max_input_tokens": 8191,
                "token_cost_per_1k": 0.00002,
                "batch_size": 100,
            },
            "text-embedding-3-large": {
                "dimensions": 3072,
                "max_input_tokens": 8191,
                "token_cost_per_1k": 0.00013,
                "batch_size": 100,
            },
            "text-embedding-ada-002": {
                "dimensions": 1536,
                "max_input_tokens": 8191,
                "token_cost_per_1k": 0.0001,
                "batch_size": 100,
            },
        },
        "api_type": "open_ai",
        "api_version": "2023-05-15",
        "api_base": "https://api.openai.com/v1",
    },
    
    Embedder.HUGGINGFACE: {
        "default_model": "sentence-transformers/all-mpnet-base-v2",
        "models": {
            "sentence-transformers/all-mpnet-base-v2": {
                "dimensions": 768,
                "max_input_tokens": 512,
                "local_model_path": None,
                "use_gpu": True,
                "device": "cuda:0",
                "batch_size": 32,
                "normalize_embeddings": True,
            },
            "sentence-transformers/all-MiniLM-L6-v2": {
                "dimensions": 384,
                "max_input_tokens": 256,
                "local_model_path": None,
                "use_gpu": True,
                "device": "cuda:0",
                "batch_size": 64,
                "normalize_embeddings": True,
            },
        },
        "use_auth_token": "",
        "quantization": "none",  # Options: "none", "int8", "int4"
        "trust_remote_code": False,
    },
    
    Embedder.COHERE: {
        "default_model": "embed-english-v3.0",
        "models": {
            "embed-english-v3.0": {
                "dimensions": 1024,
                "max_input_tokens": 512,
                "token_cost_per_1k": 0.0001,
                "batch_size": 96,
                "input_type": "search_document",  # Options: "search_document", "search_query", "classification", "clustering"
            },
            "embed-multilingual-v3.0": {
                "dimensions": 1024,
                "max_input_tokens": 512,
                "token_cost_per_1k": 0.0001,
                "batch_size": 96,
                "input_type": "search_document",
            },
        },
        "api_key": "",
        "timeout": 60,
    },
    
    Embedder.SENTENCE_TRANSFORMERS: {
        "default_model": "all-mpnet-base-v2",
        "models": {
            "all-mpnet-base-v2": {
                "dimensions": 768,
                "max_input_tokens": 384,
                "local_model_path": None,
                "use_gpu": True,
                "batch_size": 32,
                "normalize_embeddings": True,
            },
            "all-MiniLM-L6-v2": {
                "dimensions": 384,
                "max_input_tokens": 256,
                "local_model_path": None,
                "use_gpu": True,
                "batch_size": 64,
                "normalize_embeddings": True,
            },
        },
        "cache_folder": "./data/models/sentence_transformers",
        "show_progress_bar": False,
    },
}

# Memory registry
MEMORY_CONFIGS = {
    MemoryType.SHORT_TERM: SHORT_TERM_MEMORY_CONFIG,
    MemoryType.CONVERSATION: CONVERSATION_MEMORY_CONFIG,
    MemoryType.LONG_TERM: LONG_TERM_MEMORY_CONFIG,
    MemoryType.EPISODIC: EPISODIC_MEMORY_CONFIG,
    MemoryType.SEMANTIC: SEMANTIC_MEMORY_CONFIG,
    MemoryType.PROCEDURAL: PROCEDURAL_MEMORY_CONFIG,
    MemoryType.WORKING: WORKING_MEMORY_CONFIG,
    MemoryType.ENTITY: ENTITY_MEMORY_CONFIG,
    MemoryType.META: META_MEMORY_CONFIG,
}

def get_memory_config(memory_type: MemoryType) -> Dict[str, Any]:
    """
    Get configuration for a specific memory type.
    
    Args:
        memory_type: The memory type to get configuration for
        
    Returns:
        Dictionary with memory configuration
    """
    if memory_type in MEMORY_CONFIGS:
        return MEMORY_CONFIGS[memory_type].copy()
    
    # If memory type not found, return empty config
    return {}

def get_storage_config(storage_type: StorageType) -> Dict[str, Any]:
    """
    Get configuration for a specific storage type.
    
    Args:
        storage_type: The storage type to get configuration for
        
    Returns:
        Dictionary with storage configuration
    """
    if storage_type in STORAGE_CONFIGS:
        return STORAGE_CONFIGS[storage_type].copy()
    
    # If storage type not found, return empty config
    return {}

def get_embedding_config(embedder: Embedder) -> Dict[str, Any]:
    """
    Get configuration for a specific embedding provider.
    
    Args:
        embedder: The embedding provider to get configuration for
        
    Returns:
        Dictionary with embedding configuration
    """
    if embedder in EMBEDDING_CONFIGS:
        return EMBEDDING_CONFIGS[embedder].copy()
    
    # If embedder not found, return empty config
    return {}

def get_embedder_model_config(embedder: Embedder, model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific embedding model.
    
    Args:
        embedder: The embedding provider
        model_name: The name of the embedding model
        
    Returns:
        Dictionary with embedding model configuration
    """
    embedder_config = get_embedding_config(embedder)
    models = embedder_config.get("models", {})
    
    if model_name in models:
        return models[model_name].copy()
    
    # If model not found, return empty config
    return {}

def get_default_embedder_model(embedder: Embedder) -> str:
    """
    Get the default model for a specific embedding provider.
    
    Args:
        embedder: The embedding provider
        
    Returns:
        Name of the default model
    """
    embedder_config = get_embedding_config(embedder)
    return embedder_config.get("default_model", "")

def is_memory_enabled(memory_type: MemoryType) -> bool:
    """
    Check if a specific memory type is enabled.
    
    Args:
        memory_type: The memory type to check
        
    Returns:
        True if enabled, False otherwise
    """
    memory_config = get_memory_config(memory_type)
    return memory_config.get("enabled", False)

def get_global_config() -> Dict[str, Any]:
    """
    Get global memory configuration.
    
    Returns:
        Dictionary with global memory configuration
    """
    return GLOBAL_MEMORY_CONFIG.copy()

def get_enabled_memory_types() -> List[MemoryType]:
    """
    Get a list of all enabled memory types.
    
    Returns:
        List of enabled memory types
    """
    return [
        memory_type
        for memory_type in MemoryType
        if is_memory_enabled(memory_type)
    ]

def calculate_embedding_cost(embedder: Embedder, model_name: str, token_count: int) -> float:
    """
    Calculate the cost of generating embeddings.
    
    Args:
        embedder: The embedding provider
        model_name: The name of the embedding model
        token_count: Number of tokens to embed
        
    Returns:
        Estimated cost in USD
    """
    model_config = get_embedder_model_config(embedder, model_name)
    
    if not model_config:
        return 0.0
    
    token_cost_per_1k = model_config.get("token_cost_per_1k", 0)
    return (token_count / 1000) * token_cost_per_1k

def validate_memory_config(memory_type: MemoryType, config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration for a specific memory type.
    
    Args:
        memory_type: The memory type
        config: The configuration to validate
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Basic validation
    if not isinstance(config, dict):
        errors.append("Configuration must be a dictionary")
        return errors
    
    # Required fields for all memory types
    required_fields = ["enabled", "storage_type"]
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Memory-specific validation
    if memory_type == MemoryType.LONG_TERM:
        if "embedder" not in config:
            errors.append("Long-term memory requires an 'embedder' field")
        
        if "embedding_model" not in config:
            errors.append("Long-term memory requires an 'embedding_model' field")
    
    # Check storage type is valid
    if "storage_type" in config:
        try:
            storage_type = StorageType(config["storage_type"]) if isinstance(config["storage_type"], str) else config["storage_type"]
            
            # Ensure storage configuration exists
            if storage_type not in STORAGE_CONFIGS:
                errors.append(f"No configuration found for storage type: {storage_type}")
        except (ValueError, TypeError):
            errors.append(f"Invalid storage type: {config['storage_type']}")
    
    return errors

def validate_storage_config(storage_type: StorageType, config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration for a specific storage type.
    
    Args:
        storage_type: The storage type
        config: The configuration to validate
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Basic validation
    if not isinstance(config, dict):
        errors.append("Configuration must be a dictionary")
        return errors
    
    # Storage-specific validation
    if storage_type == StorageType.SQLITE:
        if "database_path" in config and not isinstance(config["database_path"], str):
            errors.append("database_path must be a string")
    
    elif storage_type == StorageType.POSTGRES:
        required_fields = ["host", "port", "database", "user"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field for PostgreSQL: {field}")
        
        if "port" in config and not isinstance(config["port"], int):
            errors.append("port must be an integer")
    
    elif storage_type == StorageType.REDIS:
        required_fields = ["host", "port"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field for Redis: {field}")
        
        if "port" in config and not isinstance(config["port"], int):
            errors.append("port must be an integer")
    
    elif storage_type in [StorageType.CHROMA, StorageType.PINECONE, StorageType.MILVUS, StorageType.QDRANT, StorageType.WEAVIATE]:
        if storage_type == StorageType.PINECONE and "api_key" not in config:
            errors.append("Pinecone requires an 'api_key' field")
        
        if storage_type == StorageType.WEAVIATE and "url" not in config:
            errors.append("Weaviate requires a 'url' field")
    
    return errors

def update_memory_config(memory_type: MemoryType, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration for a specific memory type.
    
    This function does not persist changes to disk. It only updates the in-memory configuration.
    
    Args:
        memory_type: The memory type to update
        updates: Dictionary with configuration updates
        
    Returns:
        Updated configuration dictionary
    """
    if memory_type not in MEMORY_CONFIGS:
        return {}
    
    config = MEMORY_CONFIGS[memory_type].copy()
    config.update(updates)
    MEMORY_CONFIGS[memory_type] = config
    
    return config

def update_storage_config(storage_type: StorageType, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration for a specific storage type.
    
    This function does not persist changes to disk. It only updates the in-memory configuration.
    
    Args:
        storage_type: The storage type to update
        updates: Dictionary with configuration updates
        
    Returns:
        Updated configuration dictionary
    """
    if storage_type not in STORAGE_CONFIGS:
        return {}
    
    config = STORAGE_CONFIGS[storage_type].copy()
    config.update(updates)
    STORAGE_CONFIGS[storage_type] = config
    
    return config

def update_embedding_config(embedder: Embedder, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration for a specific embedding provider.
    
    This function does not persist changes to disk. It only updates the in-memory configuration.
    
    Args:
        embedder: The embedding provider to update
        updates: Dictionary with configuration updates
        
    Returns:
        Updated configuration dictionary
    """
    if embedder not in EMBEDDING_CONFIGS:
        return {}
    
    config = EMBEDDING_CONFIGS[embedder].copy()
    
    # Handle special case for updating models
    if "models" in updates:
        models = config.get("models", {}).copy()
        models.update(updates.pop("models"))
        config["models"] = models
    
    config.update(updates)
    EMBEDDING_CONFIGS[embedder] = config
    
    return config

def get_storage_for_memory_type(memory_type: MemoryType) -> StorageType:
    """
    Get the storage type configured for a specific memory type.
    
    Args:
        memory_type: The memory type
        
    Returns:
        The configured storage type
    """
    memory_config = get_memory_config(memory_type)
    storage_type_value = memory_config.get("storage_type")
    
    if isinstance(storage_type_value, StorageType):
        return storage_type_value
    
    try:
        return StorageType(storage_type_value)
    except (ValueError, TypeError):
        # Return default storage type if invalid
        return StorageType(GLOBAL_MEMORY_CONFIG["default_storage_type"]) 