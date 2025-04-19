"""
Task status module for Telos AI.

This module handles task progress and status tracking.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union

# Import configurations
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

logger = logging.getLogger(__name__)

# Path to the task progress file
TASK_PROGRESS_DIR = os.path.join(config.MEMORY_DIR, "task_progress")

def get_task_progress(task_id: str) -> Dict[str, Any]:
    """
    Get the progress of a task.
    
    Args:
        task_id: ID of the task
        
    Returns:
        Dictionary with task progress information
    """
    try:
        # Ensure task progress directory exists
        os.makedirs(TASK_PROGRESS_DIR, exist_ok=True)
        
        progress_file = os.path.join(TASK_PROGRESS_DIR, f"{task_id}.json")
        
        if not os.path.exists(progress_file):
            return {
                'task_id': task_id,
                'progress': 0,
                'status': 'not_started',
                'steps': []
            }
            
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            
        return progress
    except Exception as e:
        logger.error(f"Error getting task progress for {task_id}: {e}")
        return {
            'task_id': task_id,
            'progress': 0,
            'status': 'error',
            'error': str(e),
            'steps': []
        }

def save_task_progress(
    task_id: str,
    progress: float,
    status_message: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Save the progress of a task.
    
    Args:
        task_id: ID of the task
        progress: Progress percentage (0-100)
        status_message: Status message
        metadata: Optional additional metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure task progress directory exists
        os.makedirs(TASK_PROGRESS_DIR, exist_ok=True)
        
        progress_file = os.path.join(TASK_PROGRESS_DIR, f"{task_id}.json")
        
        # Get existing progress if it exists
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    task_progress = json.load(f)
            except Exception:
                task_progress = {
                    'task_id': task_id,
                    'progress': 0,
                    'steps': []
                }
        else:
            task_progress = {
                'task_id': task_id,
                'progress': 0,
                'steps': []
            }
        
        # Add new step
        step = {
            'timestamp': time.time(),
            'progress': progress,
            'message': status_message
        }
        
        if metadata:
            step['metadata'] = metadata
            
        task_progress['steps'].append(step)
        task_progress['progress'] = progress
        task_progress['last_update'] = time.time()
        task_progress['last_message'] = status_message
        
        # Save progress
        with open(progress_file, 'w') as f:
            json.dump(task_progress, f, indent=2)
            
        logger.debug(f"Saved progress for task {task_id}: {progress}% - {status_message}")
        return True
    except Exception as e:
        logger.error(f"Error saving task progress for {task_id}: {e}")
        return False

def clear_task_progress(task_id: str) -> bool:
    """
    Clear the progress of a task.
    
    Args:
        task_id: ID of the task
        
    Returns:
        True if successful, False otherwise
    """
    try:
        progress_file = os.path.join(TASK_PROGRESS_DIR, f"{task_id}.json")
        
        if os.path.exists(progress_file):
            os.remove(progress_file)
            logger.info(f"Cleared progress for task {task_id}")
            
        return True
    except Exception as e:
        logger.error(f"Error clearing task progress for {task_id}: {e}")
        return False

def is_task_complete(task_id: str) -> bool:
    """
    Check if a task is complete.
    
    Args:
        task_id: ID of the task
        
    Returns:
        True if the task is complete, False otherwise
    """
    try:
        # Import here to avoid circular imports
        from core.tasks.manager import get_task
        
        task = get_task(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found")
            return False
            
        return task.get('status') == 'completed'
    except Exception as e:
        logger.error(f"Error checking if task {task_id} is complete: {e}")
        return False

def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of a task, combining task info and progress.
    
    Args:
        task_id: ID of the task
        
    Returns:
        Dictionary with task status information
    """
    try:
        # Import here to avoid circular imports
        from core.tasks.manager import get_task
        
        task = get_task(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found")
            return {
                'task_id': task_id,
                'status': 'not_found',
                'error': 'Task not found'
            }
            
        progress = get_task_progress(task_id)
        
        # Combine task and progress information
        status = {
            'task_id': task_id,
            'task_name': task.get('task', 'Unknown'),
            'status': task.get('status', 'pending'),
            'progress': progress.get('progress', 0),
            'created_at': task.get('created_at'),
            'updated_at': task.get('updated_at'),
            'last_progress_update': progress.get('last_update'),
            'last_status_message': progress.get('last_message')
        }
        
        # Add optional fields if they exist
        if 'deadline' in task:
            status['deadline'] = task['deadline']
        if 'tags' in task:
            status['tags'] = task['tags']
        if 'priority' in task:
            status['priority'] = task['priority']
            
        return status
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {e}")
        return {
            'task_id': task_id,
            'status': 'error',
            'error': str(e)
        } 