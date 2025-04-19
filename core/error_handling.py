"""
Error handling module for Telos AI.

This module provides centralized error handling mechanisms for various parts
of the Telos system, including error recovery, logging, and adjustments based
on error frequencies.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, TypeVar

# Import configurations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import system as system_config

logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar('T')

def adjust_error_handling(consecutive_errors: int) -> int:
    """
    Adjust system parameters based on error frequency.
    
    Args:
        consecutive_errors: Number of consecutive errors encountered
        
    Returns:
        Updated consecutive error count
    """
    if consecutive_errors >= 5:
        logger.critical("Too many consecutive errors, adjusting settings to compensate")
        # Increase heartbeat interval to give more time between attempts
        system_config.HEARTBEAT_INTERVAL = min(
            system_config.HEARTBEAT_INTERVAL * 1.5, 
            system_config.MAX_HEARTBEAT_INTERVAL
        )
        # Reset consecutive error count but not to zero to maintain awareness
        return 2
    return consecutive_errors

def recover_heartbeat(consecutive_errors: int) -> None:
    """
    Gradually restore heartbeat interval if it was increased due to errors.
    
    Args:
        consecutive_errors: Current count of consecutive errors
    """
    if consecutive_errors == 0 and system_config.HEARTBEAT_INTERVAL > system_config.MIN_HEARTBEAT_INTERVAL:
        system_config.HEARTBEAT_INTERVAL = max(
            system_config.HEARTBEAT_INTERVAL * 0.9, 
            system_config.MIN_HEARTBEAT_INTERVAL
        )
        logger.info(f"Gradually restoring heartbeat interval to {system_config.HEARTBEAT_INTERVAL}s")

def log_exception(exception: Exception, context: Dict[str, Any] = None) -> None:
    """
    Log an exception with additional context.
    
    Args:
        exception: The exception to log
        context: Additional context information for the exception
    """
    if context is None:
        context = {}
        
    error_id = int(time.time())
    
    # Get the stack trace
    stack_trace = traceback.format_exc()
    
    logger.error(
        f"Error ID: {error_id} - {str(exception)}",
        extra={
            "error_id": error_id,
            "error_type": type(exception).__name__,
            "context": context,
        },
        exc_info=True
    )
    
    # Save detailed error information to file for later analysis
    try:
        error_dir = os.path.join(system_config.MEMORY_DIR, "errors")
        os.makedirs(error_dir, exist_ok=True)
        
        with open(os.path.join(error_dir, f"error_{error_id}.log"), "w") as f:
            f.write(f"Error Type: {type(exception).__name__}\n")
            f.write(f"Error Message: {str(exception)}\n")
            f.write(f"Context: {context}\n\n")
            f.write(f"Stack Trace:\n{stack_trace}\n")
    except Exception as e:
        logger.warning(f"Failed to save detailed error information: {e}")

def safe_execute(func: Callable[..., T], *args, **kwargs) -> Optional[T]:
    """
    Execute a function safely, catching and logging any exceptions.
    
    Args:
        func: The function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function or None if an exception occurred
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log_exception(e, {
            "function": func.__name__,
            "args": args,
            "kwargs": {k: v for k, v in kwargs.items() if not k.startswith("_")}
        })
        return None

def get_error_history(limit: int = 10) -> Dict[str, Any]:
    """
    Get information about recent errors.
    
    Args:
        limit: Maximum number of errors to retrieve
        
    Returns:
        Dictionary with error history information
    """
    error_dir = os.path.join(system_config.MEMORY_DIR, "errors")
    if not os.path.exists(error_dir):
        return {"errors": [], "count": 0}
        
    error_files = sorted(
        [f for f in os.listdir(error_dir) if f.startswith("error_")],
        reverse=True
    )[:limit]
    
    errors = []
    for error_file in error_files:
        try:
            with open(os.path.join(error_dir, error_file), "r") as f:
                error_content = f.read()
                
            error_info = {
                "id": error_file.replace("error_", "").replace(".log", ""),
                "timestamp": error_file.replace("error_", "").replace(".log", ""),
                "content": error_content[:500] + "..." if len(error_content) > 500 else error_content
            }
            errors.append(error_info)
        except Exception as e:
            logger.warning(f"Failed to read error file {error_file}: {e}")
    
    return {
        "errors": errors,
        "count": len(errors)
    }

def check_memory_corruption() -> bool:
    """
    Check for signs of memory corruption in critical files.
    
    Returns:
        True if corruption detected, False otherwise
    """
    critical_files = [
        os.path.join(system_config.MEMORY_DIR, "tasks.json"),
        os.path.join(system_config.MEMORY_DIR, "heartbeat.pid"),
        os.path.join(system_config.MEMORY_DIR, "last_heartbeat.txt")
    ]
    
    for file_path in critical_files:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, "r") as f:
                content = f.read()
                
            # Check for common corruption signs
            if file_path.endswith(".json"):
                import json
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    logger.error(f"Detected JSON corruption in {file_path}")
                    return True
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            return True
    
    return False
