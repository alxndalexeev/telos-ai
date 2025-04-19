"""
Search module for Telos AI.

This module handles online search functionality using the Tavily API.
"""

import os
import json
import logging
from typing import Dict, Optional, Any, List

# Import configurations
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

def perform_search(task: Dict[str, Any]) -> Optional[str]:
    """
    Perform online search for task-related information.
    
    Args:
        task: Task information
        
    Returns:
        Search results as a string or None if search failed
    """
    task_name = task.get('task', 'N/A')
    search_query = f"best practices for {task_name} {task.get('details', '')}"
    
    try:
        from tools import tavily_search
        search_func = getattr(tavily_search, 'run_search', tavily_search)
        if callable(search_func):
            search_results = search_func(search_query)
            if isinstance(search_results, dict):
                return json.dumps(search_results)
            return search_results
    except Exception as e:
        logger.warning(f"Online search failed: {e}")
    
    return None

def search_documentation(query: str, doc_type: str = "general") -> List[Dict[str, Any]]:
    """
    Search documentation based on a query.
    
    Args:
        query: Search query
        doc_type: Type of documentation to search (e.g., "api", "tutorial", "general")
        
    Returns:
        List of relevant documentation snippets
    """
    try:
        # This is a placeholder that would be replaced with actual documentation search
        # In a real implementation, this might search a vector database of documentation
        logger.info(f"Searching {doc_type} documentation for: {query}")
        
        # Placeholder results
        return [{
            "title": "Documentation search not implemented yet",
            "content": f"Query was: {query}, doc_type: {doc_type}",
            "source": "placeholder"
        }]
    except Exception as e:
        logger.error(f"Error searching documentation: {e}")
        return []

def search_codebase(query: str, file_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search the codebase for code examples.
    
    Args:
        query: Search query
        file_pattern: Optional file pattern to restrict search (e.g., "*.py")
        
    Returns:
        List of relevant code snippets
    """
    try:
        # This is a placeholder that would be replaced with actual codebase search
        # In a real implementation, this might use grep, ripgrep, or a vector DB
        logger.info(f"Searching codebase for: {query}")
        
        # Placeholder results
        return [{
            "file": "placeholder.py",
            "content": f"# Code search for: {query}",
            "line_number": 1
        }]
    except Exception as e:
        logger.error(f"Error searching codebase: {e}")
        return [] 