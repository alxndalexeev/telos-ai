"""
Vector database index management module.

This module handles operations related to the Pinecone vector index:
- Index initialization
- Index connection
- Index configuration
"""

import os
import logging
import time
from typing import Optional, Any

import pinecone

# Import configurations
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

logger = logging.getLogger(__name__)

# Cache the Pinecone index connection to avoid reconnecting unnecessarily
_pinecone_index = None
_last_init_time = 0
INDEX_CONNECTION_TIMEOUT = 3600  # 1 hour in seconds

def initialize_pinecone() -> bool:
    """Initialize Pinecone connection."""
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            logger.warning("PINECONE_API_KEY not set. Vector database functionality will be disabled.")
            return False

        environment = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
        pinecone.init(api_key=api_key, environment=environment)
        logger.info(f"Pinecone initialized with environment: {environment}")
        return True
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        return False

def get_pinecone_index() -> Optional[Any]:
    """Get the Pinecone index, initializing if necessary."""
    global _pinecone_index, _last_init_time
    
    # Check if we need to initialize or re-initialize (after timeout)
    current_time = time.time()
    if (_pinecone_index is None or 
        current_time - _last_init_time > INDEX_CONNECTION_TIMEOUT):
        
        # Initialize Pinecone
        if not initialize_pinecone():
            return None
        
        try:
            index_name = os.getenv("PINECONE_INDEX_NAME", config.PINECONE_INDEX_NAME)
            _pinecone_index = pinecone.Index(index_name)
            _last_init_time = current_time
            logger.info(f"Connected to Pinecone index: {index_name}")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone index: {e}")
            _pinecone_index = None
    
    return _pinecone_index

def create_index_if_not_exists() -> bool:
    """Create the Pinecone index if it doesn't exist."""
    try:
        # Initialize Pinecone
        if not initialize_pinecone():
            return False
            
        index_name = os.getenv("PINECONE_INDEX_NAME", config.PINECONE_INDEX_NAME)
        dimension = config.EMBEDDING_DIMENSION
        metric = "cosine"
        
        # Check if index exists
        if index_name not in pinecone.list_indexes():
            logger.info(f"Creating new Pinecone index: {index_name}")
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric
            )
            logger.info(f"Created Pinecone index: {index_name}")
            return True
        else:
            logger.info(f"Pinecone index {index_name} already exists")
            return True
    except Exception as e:
        logger.error(f"Error creating Pinecone index: {e}")
        return False 