"""
Task manager module for Telos AI.

This module handles task queue management, task selection, and task state.
"""

import os
import json
import time
import logging
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Import configurations
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

logger = logging.getLogger(__name__)

# Path to the tasks file
TASKS_FILE = os.path.join(config.MEMORY_DIR, "tasks.json")

def get_task(task_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get a task from the queue.
    
    Args:
        task_id: Optional ID of the specific task to get. If not provided,
                returns the highest priority task that hasn't been completed.
    
    Returns:
        Task dictionary or None if no tasks are available
    """
    try:
        tasks = _load_tasks()
        
        # If a specific task ID is provided, return that task
        if task_id:
            for task in tasks:
                if task.get('id') == task_id:
                    return task
            logger.warning(f"Task with ID {task_id} not found")
            return None
            
        # Otherwise, filter for tasks that are not completed or failed
        available_tasks = [
            t for t in tasks 
            if t.get('status', 'pending') not in ['completed', 'failed']
        ]
        
        if not available_tasks:
            logger.info("No available tasks in the queue")
            return None
            
        # Sort by priority (higher is more important)
        available_tasks.sort(key=lambda t: t.get('priority', config.DEFAULT_TASK_PRIORITY), reverse=True)
        
        # Return the highest priority task
        selected_task = available_tasks[0]
        logger.info(f"Selected task: {selected_task.get('task')} (ID: {selected_task.get('id')})")
        return selected_task
    except Exception as e:
        logger.error(f"Error getting task: {e}")
        return None

def add_task(
    task_name: str,
    details: str,
    priority: int = None,
    deadline: Optional[str] = None,
    tags: List[str] = None,
    metadata: Dict[str, Any] = None
) -> Optional[str]:
    """
    Add a new task to the queue.
    
    Args:
        task_name: Name of the task
        details: Details/description of the task
        priority: Priority level (1-10, higher = more important)
        deadline: Optional deadline string (ISO format)
        tags: Optional list of tags for categorization
        metadata: Optional additional metadata
    
    Returns:
        ID of the new task if successful, None otherwise
    """
    try:
        # Load current tasks
        tasks = _load_tasks()
        
        # Generate a unique ID
        task_id = str(uuid.uuid4())
        
        # Create new task
        new_task = {
            'id': task_id,
            'task': task_name,
            'details': details,
            'priority': priority or config.DEFAULT_TASK_PRIORITY,
            'status': 'pending',
            'created_at': time.time(),
            'updated_at': time.time()
        }
        
        # Add optional fields
        if deadline:
            new_task['deadline'] = deadline
        if tags:
            new_task['tags'] = tags
        if metadata:
            new_task['metadata'] = metadata
            
        # Add to tasks list
        tasks.append(new_task)
        
        # Save updated tasks
        if _save_tasks(tasks):
            logger.info(f"Added new task: {task_name} (ID: {task_id})")
            return task_id
        return None
    except Exception as e:
        logger.error(f"Error adding task: {e}")
        return None

def update_task(
    task_id: str,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    notes: Optional[str] = None,
    priority: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Update a task in the queue.
    
    Args:
        task_id: ID of the task to update
        status: Optional new status
        progress: Optional progress (0-100)
        notes: Optional notes to add
        priority: Optional new priority
        metadata: Optional metadata to update
    
    Returns:
        True if successful, False otherwise
    """
    try:
        tasks = _load_tasks()
        
        # Find the task with the given ID
        for task in tasks:
            if task.get('id') == task_id:
                # Update the fields
                if status:
                    task['status'] = status
                if progress is not None:
                    task['progress'] = float(progress)
                if notes:
                    task['notes'] = notes
                if priority is not None:
                    task['priority'] = int(priority)
                if metadata:
                    if 'metadata' not in task:
                        task['metadata'] = {}
                    task['metadata'].update(metadata)
                
                # Update the timestamp
                task['updated_at'] = time.time()
                
                # Save the updated tasks
                if _save_tasks(tasks):
                    logger.info(f"Updated task {task_id} (status: {status})")
                    return True
                return False
                
        logger.warning(f"Task with ID {task_id} not found")
        return False
    except Exception as e:
        logger.error(f"Error updating task: {e}")
        return False

def get_all_tasks(
    status: Optional[str] = None,
    tag: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get all tasks, optionally filtered by status or tag.
    
    Args:
        status: Optional status to filter by
        tag: Optional tag to filter by
    
    Returns:
        List of tasks matching the filters
    """
    try:
        tasks = _load_tasks()
        
        # Apply filters if specified
        if status:
            tasks = [t for t in tasks if t.get('status') == status]
        if tag:
            tasks = [t for t in tasks if tag in t.get('tags', [])]
            
        return tasks
    except Exception as e:
        logger.error(f"Error getting all tasks: {e}")
        return []

def delete_task(task_id: str) -> bool:
    """
    Delete a task from the queue.
    
    Args:
        task_id: ID of the task to delete
    
    Returns:
        True if successful, False otherwise
    """
    try:
        tasks = _load_tasks()
        
        # Find the task with the given ID
        initial_count = len(tasks)
        tasks = [t for t in tasks if t.get('id') != task_id]
        
        # If no task was removed, return False
        if len(tasks) == initial_count:
            logger.warning(f"Task with ID {task_id} not found")
            return False
            
        # Save the updated tasks
        if _save_tasks(tasks):
            logger.info(f"Deleted task {task_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting task: {e}")
        return False

def validate_task(task: Any) -> bool:
    """
    Validate that a task object has the required fields.
    
    Args:
        task: Task to validate
        
    Returns:
        True if the task is valid, False otherwise
    """
    if not isinstance(task, dict):
        logger.error(f"Invalid task: not a dictionary - {task}")
        return False
        
    # Required fields
    required_fields = ["task", "details"]
    for field in required_fields:
        if field not in task:
            logger.error(f"Invalid task: missing required field '{field}' - {task}")
            return False
            
    # Check if task has a valid string value
    if not isinstance(task.get("task"), str) or not task.get("task"):
        logger.error(f"Invalid task: 'task' field must be a non-empty string - {task}")
        return False
        
    # Details could be empty but must be a string
    if not isinstance(task.get("details"), str):
        logger.error(f"Invalid task: 'details' field must be a string - {task}")
        return False
        
    return True

def add_tasks(new_tasks: List[Dict[str, Any]]) -> None:
    """
    Add new tasks to the end of the tasks list.
    
    Args:
        new_tasks: List of tasks to add
    """
    if not isinstance(new_tasks, list):
        logger.error(f"Invalid format for new tasks: {new_tasks}. Expected a list.")
        return
    if not new_tasks:
        logger.info("No new tasks provided to add.")
        return
        
    # Validate tasks before adding
    valid_tasks = []
    for task in new_tasks:
        if validate_task(task):
            # Ensure each task has an ID and timestamps
            if 'id' not in task:
                task['id'] = str(uuid.uuid4())
            if 'created_at' not in task:
                task['created_at'] = time.time()
            if 'updated_at' not in task:
                task['updated_at'] = time.time()
            if 'status' not in task:
                task['status'] = 'pending'
                
            valid_tasks.append(task)
        # Invalid tasks are logged in validate_task
    
    if not valid_tasks:
        logger.warning("No valid tasks found in the provided list.")
        return
        
    # Load existing tasks
    tasks = _load_tasks()
        
    # Append new tasks
    tasks.extend(valid_tasks)
    logger.info(f"Adding {len(valid_tasks)} new tasks to the queue.")

    # Save the updated tasks
    if _save_tasks(tasks):
        logger.info(f"Tasks file updated. Total tasks: {len(tasks)}")
    else:
        logger.error("Failed to save updated tasks")

def add_architecture_task() -> Optional[str]:
    """
    Add a task to analyze and improve the system architecture.
    
    Returns:
        ID of the created task if successful, None otherwise
    """
    task = {
        "task": "architecture",
        "details": "Analyze the current architecture and implement improvements to enhance modularity, extensibility, and maintainability.",
        "priority": 5,  # High priority
        "created_at": time.time(),
        "tags": ["architecture", "improvement"]
    }
    
    task_id = str(uuid.uuid4())
    task['id'] = task_id
    
    add_tasks([task])
    logger.info(f"Added architecture improvement task: {task}")
    return task_id

def add_testing_task(module_path: str = None, module_type: str = "all") -> Optional[str]:
    """
    Add a task to create or run tests.
    
    Args:
        module_path: Optional specific module to test
        module_type: Type of tests ("unit", "integration", or "all")
        
    Returns:
        ID of the created task if successful, None otherwise
    """
    task_id = str(uuid.uuid4())
    
    if module_path:
        task = {
            "id": task_id,
            "task": "testing",
            "details": f"Generate and run {module_type} tests for {module_path}",
            "priority": 4,  # Medium-high priority
            "created_at": time.time(),
            "metadata": {
                "module": module_path,
                "test_type": module_type
            },
            "tags": ["testing", module_type]
        }
    else:
        task = {
            "id": task_id,
            "task": "testing",
            "details": f"Analyze test coverage and generate tests for untested modules",
            "priority": 4,  # Medium-high priority
            "created_at": time.time(),
            "metadata": {
                "test_type": module_type
            },
            "tags": ["testing", "coverage"]
        }
    
    add_tasks([task])
    logger.info(f"Added testing task: {task}")
    return task_id

def _load_tasks() -> List[Dict[str, Any]]:
    """
    Load tasks from the tasks file.
    
    Returns:
        List of tasks or empty list if file doesn't exist or is invalid
    """
    if not os.path.exists(TASKS_FILE):
        # Ensure directory exists
        os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)
        # Create an empty tasks file
        try:
            with open(TASKS_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
            logger.debug("Created empty tasks file.")
        except Exception as e:
            logger.error(f"Error creating tasks file: {e}")
        return []
        
    try:
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return []
                
            tasks = json.loads(content)
            if not isinstance(tasks, list):
                logger.warning(f"Tasks file content is not a list. Returning empty list.")
                return []
                
            return tasks
    except Exception as e:
        logger.error(f"Error loading tasks: {e}")
        return []

def _save_tasks(tasks: List[Dict[str, Any]]) -> bool:
    """
    Save tasks to the tasks file.
    
    Args:
        tasks: List of tasks to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)
        
        with open(TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving tasks: {e}")
        return False 