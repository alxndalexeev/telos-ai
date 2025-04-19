"""
Vector database refresh module.

This module handles vector database refresh scheduling and operations.
"""

import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Import configurations
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Import local modules
from core.vector_db.index_manager import get_pinecone_index
from core import context_gatherer  # Still need this for full refresh operations

logger = logging.getLogger(__name__)

# Constants for refresh scheduling
FULL_REFRESH_DAYS = 1  # How often to do a full refresh of the vector DB (days)
LAST_FULL_REFRESH_FILE = os.path.join(str(config.MEMORY_DIR), "last_full_refresh.txt")

def check_refresh_needed() -> bool:
    """
    Check if a full vector DB refresh is needed based on timestamp.
    
    Returns:
        True if a full refresh is needed, False otherwise
    """
    try:
        if os.path.exists(LAST_FULL_REFRESH_FILE):
            with open(LAST_FULL_REFRESH_FILE, 'r') as f:
                last_refresh_str = f.read().strip()
                last_refresh = datetime.fromisoformat(last_refresh_str)
                days_since_refresh = (datetime.now() - last_refresh).days
                if days_since_refresh >= FULL_REFRESH_DAYS:
                    logger.info(f"Full vector DB refresh needed ({days_since_refresh} days since last refresh)")
                    return True
                return False
        else:
            logger.info("First-time full vector DB refresh needed")
            return True
    except Exception as e:
        logger.warning(f"Error checking vector DB refresh state: {e}. Will assume incremental update.")
        return False

def record_refresh_timestamp() -> None:
    """Record the timestamp of the last full vector DB refresh."""
    try:
        # Ensure the memory directory exists
        os.makedirs(os.path.dirname(LAST_FULL_REFRESH_FILE), exist_ok=True)
        
        with open(LAST_FULL_REFRESH_FILE, 'w') as f:
            f.write(datetime.now().isoformat())
        logger.info("Recorded full vector DB refresh timestamp")
    except Exception as e:
        logger.warning(f"Failed to record vector DB refresh timestamp: {e}")

def perform_incremental_update() -> bool:
    """
    Perform an incremental update of the vector database.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Call the existing method in context_gatherer
        # This will eventually be refactored into this module
        context_gatherer.self_update_vector_db(force_refresh=False)
        logger.info("Completed incremental vector DB update")
        return True
    except Exception as e:
        logger.error(f"Error during incremental vector DB update: {e}")
        return False

def perform_full_refresh() -> bool:
    """
    Perform a full refresh of the vector database.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Call the existing method in context_gatherer
        # This will eventually be refactored into this module
        context_gatherer.self_update_vector_db(force_refresh=True)
        logger.info("Completed full vector DB refresh")
        return True
    except Exception as e:
        logger.error(f"Error during full vector DB refresh: {e}")
        return False

def update_vector_db(task_name: str, knowledge_chunks: List[Dict]) -> None:
    """
    Update the vector database with new knowledge and check if full refresh is needed.
    
    Args:
        task_name: Name of the task
        knowledge_chunks: Knowledge chunks to store
    """
    # Import here to avoid circular imports
    from core.vector_db.knowledge_store import store_knowledge_in_vector_db
    
    # First store the new knowledge from this task execution
    store_success = store_knowledge_in_vector_db(knowledge_chunks, task_name)
    
    # Check if we need a full refresh
    force_refresh = check_refresh_needed()
    
    # Update the vector DB (full refresh or incremental)
    try:
        if force_refresh:
            # Do full refresh
            success = perform_full_refresh()
            
            # Record last full refresh time if we did one successfully
            if success:
                record_refresh_timestamp()
        else:
            # Do incremental update
            perform_incremental_update()
    except Exception as e:
        logger.error(f"Error during vector DB update: {e}")

def get_vector_db_stats() -> Dict[str, Any]:
    """
    Get statistics about the vector database.
    
    Returns:
        Dictionary with vector DB statistics
    """
    try:
        index = get_pinecone_index()
        if not index:
            logger.warning("Could not get Pinecone index, unable to get stats")
            return {"error": "Could not connect to vector database"}
            
        namespace = os.getenv("PINECONE_NAMESPACE", config.PINECONE_NAMESPACE)
        stats = index.describe_index_stats()
        
        # Extract namespace stats if available
        namespace_stats = {}
        if 'namespaces' in stats and namespace in stats['namespaces']:
            namespace_stats = stats['namespaces'][namespace]
        
        # Get last refresh time
        last_refresh = None
        try:
            if os.path.exists(LAST_FULL_REFRESH_FILE):
                with open(LAST_FULL_REFRESH_FILE, 'r') as f:
                    last_refresh_str = f.read().strip()
                    last_refresh = datetime.fromisoformat(last_refresh_str).isoformat()
        except Exception:
            pass
            
        return {
            "total_vector_count": stats.get('total_vector_count', 0),
            "namespace_vector_count": namespace_stats.get('vector_count', 0),
            "dimension": stats.get('dimension', 0),
            "index_fullness": stats.get('index_fullness', 0),
            "last_full_refresh": last_refresh,
            "namespaces": list(stats.get('namespaces', {}).keys())
        }
    except Exception as e:
        logger.error(f"Error getting vector DB stats: {e}")
        return {"error": str(e)} 