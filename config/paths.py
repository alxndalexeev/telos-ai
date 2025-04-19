"""
Path configuration for Telos AI.

This module centralizes path definitions for various components 
and resources used by the Telos system.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base directories
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
TEMP_DIR = BASE_DIR / "temp"

# Memory-related paths
MEMORY_DIR = DATA_DIR / "memory"
LONG_TERM_MEMORY_DIR = MEMORY_DIR / "long_term"
SHORT_TERM_MEMORY_DIR = MEMORY_DIR / "short_term"
EPISODIC_MEMORY_FILE = MEMORY_DIR / "episodic.json"
SEMANTIC_MEMORY_DIR = MEMORY_DIR / "semantic"
INDEX_DIR = MEMORY_DIR / "index"

# Task-related paths
TASKS_DIR = DATA_DIR / "tasks"
TASK_ARCHIVE_DIR = TASKS_DIR / "archive"
TASK_TEMPLATES_DIR = TASKS_DIR / "templates"

# Knowledge base paths
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
DOCUMENTS_DIR = KNOWLEDGE_BASE_DIR / "documents"
EMBEDDINGS_DIR = KNOWLEDGE_BASE_DIR / "embeddings"

# User-specific paths
USER_DATA_DIR = DATA_DIR / "user"
USER_PREFERENCES_FILE = USER_DATA_DIR / "preferences.json"
USER_HISTORY_FILE = USER_DATA_DIR / "history.json"

# Asset paths
ASSETS_DIR = BASE_DIR / "assets"
TEMPLATES_DIR = ASSETS_DIR / "templates"

# Create required directories at import time
def ensure_directories_exist():
    """Create all required directories if they don't exist."""
    for directory in [
        DATA_DIR,
        MODELS_DIR,
        LOGS_DIR,
        CACHE_DIR,
        TEMP_DIR,
        MEMORY_DIR,
        LONG_TERM_MEMORY_DIR,
        SHORT_TERM_MEMORY_DIR,
        SEMANTIC_MEMORY_DIR,
        INDEX_DIR,
        TASKS_DIR,
        TASK_ARCHIVE_DIR,
        TASK_TEMPLATES_DIR,
        KNOWLEDGE_BASE_DIR,
        DOCUMENTS_DIR,
        EMBEDDINGS_DIR,
        USER_DATA_DIR,
        ASSETS_DIR,
        TEMPLATES_DIR,
    ]:
        directory.mkdir(exist_ok=True, parents=True)

# Create directories on module import
ensure_directories_exist()

def get_memory_path(memory_type: str, name: str = None) -> Path:
    """
    Get a path for a specific memory type and optional name.
    
    Args:
        memory_type: Type of memory (e.g., 'long_term', 'short_term', 'semantic')
        name: Optional name/identifier for the memory resource
        
    Returns:
        Path object for the requested memory resource
    """
    if memory_type == "long_term":
        base_path = LONG_TERM_MEMORY_DIR
    elif memory_type == "short_term":
        base_path = SHORT_TERM_MEMORY_DIR
    elif memory_type == "semantic":
        base_path = SEMANTIC_MEMORY_DIR
    elif memory_type == "episodic":
        return EPISODIC_MEMORY_FILE
    else:
        base_path = MEMORY_DIR
    
    return base_path / name if name else base_path

def get_task_path(task_id: str, archived: bool = False) -> Path:
    """
    Get the path for a specific task.
    
    Args:
        task_id: Unique identifier for the task
        archived: Whether the task is archived
        
    Returns:
        Path object for the requested task
    """
    base_dir = TASK_ARCHIVE_DIR if archived else TASKS_DIR
    return base_dir / f"{task_id}.json"

def get_knowledge_base_path(document_name: str = None) -> Path:
    """
    Get the path for a knowledge base document.
    
    Args:
        document_name: Optional name of the document
        
    Returns:
        Path object for the requested document
    """
    return DOCUMENTS_DIR / document_name if document_name else DOCUMENTS_DIR

def get_path_config() -> Dict[str, Any]:
    """
    Get the current path configuration as a dictionary.
    
    Returns:
        Dictionary with path configuration details
    """
    return {
        "base_dir": str(BASE_DIR),
        "config_dir": str(CONFIG_DIR),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "logs_dir": str(LOGS_DIR),
        "cache_dir": str(CACHE_DIR),
        "temp_dir": str(TEMP_DIR),
        "memory_dir": str(MEMORY_DIR),
        "tasks_dir": str(TASKS_DIR),
        "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR),
        "user_data_dir": str(USER_DATA_DIR),
        "assets_dir": str(ASSETS_DIR),
    } 