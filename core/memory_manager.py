import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import config
from telos_logging.logger import log_thought # Import from new module structure

# Get the logger for this module
logger = logging.getLogger(__name__)

def ensure_memory_dir() -> None:
    """Ensure the memory directory and core log files exist."""
    try:
        os.makedirs(config.MEMORY_DIR, exist_ok=True)
        # Ensure log files exist (optional, could be handled by logger solely)
        for log_file in [config.ACTION_LOG, config.THOUGHTS_LOG]:
             if not os.path.exists(log_file):
                 try:
                     with open(log_file, 'w', encoding='utf-8') as f:
                         # Basic header based on filename
                         header_name = os.path.basename(log_file).split('.')[0].replace('_', ' ').title()
                         f.write(f"# {header_name} Log\n\n")
                 except OSError as e:
                     logger.warning(f"Could not create log file {log_file}: {e}")
    except OSError as e:
        logger.error(f"Error creating memory directory {config.MEMORY_DIR}: {e}")
        raise # Reraise the exception to halt if we can't create memory

# --- Core Context Files ---
CORE_CONTEXT_FILES = ["who_i_am.md", "my_goal.md"]

def get_core_context() -> Dict[str, str]:
    """Load core, immutable context files (identity, high-level goals)."""
    context = {}
    for fname in CORE_CONTEXT_FILES:
        fpath = os.path.join(config.MEMORY_DIR, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                context[fname] = f.read()
        except Exception as e:
            logger.warning(f"Could not read core context file {fname}: {e}")
    return context

def get_dynamic_context(previous_memory: Dict[str, str] = None) -> Dict[str, str]:
    """Retrieve dynamic context (logs, notes, etc.).
    For now, loads all .md files in memory except core files. Optionally merges previous_memory for task continuity."""
    context = {}
    try:
        for fname in os.listdir(config.MEMORY_DIR):
            if fname.endswith('.md') and fname not in CORE_CONTEXT_FILES:
                fpath = os.path.join(config.MEMORY_DIR, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        context[fname] = f.read()
                except Exception as e:
                    logger.warning(f"Could not read dynamic context file {fname}: {e}")
    except FileNotFoundError:
        logger.warning(f"Memory directory not found: {config.MEMORY_DIR}. Cannot load dynamic context.")
    except Exception as e:
        logger.error(f"Error reading dynamic context from memory directory: {e}")
    # Optionally merge previous_memory for continuity
    if previous_memory:
        context.update(previous_memory)
    return context

def get_context(previous_memory: Dict[str, str] = None) -> Dict[str, str]:
    """Assemble full context: core (immutable) + dynamic (mutable, e.g. working memory)."""
    context = get_core_context()
    context.update(get_dynamic_context(previous_memory=previous_memory))
    return context

# Define a type alias for Task for clarity
Task = Dict[str, Any]

def get_task() -> Task:
    """Get the next task from the tasks file, returning a 'self-improvement' task if none exist or file is invalid."""
    default_task: Task = {
        'task': 'self-improvement',
        'details': 'No specific tasks. Engaging in self-improvement cycle.'
    }
    
    tasks_file = config.TASKS_FILE
    
    # Create parent directory if needed
    os.makedirs(os.path.dirname(tasks_file), exist_ok=True)
    
    try:
        if not os.path.exists(tasks_file):
            # Create an empty tasks file for future use
            try:
                with open(tasks_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                logger.debug("Created empty tasks file.")
            except Exception as e:
                logger.error(f"Error creating tasks file: {e}")
            return default_task
            
        with open(tasks_file, 'r', encoding='utf-8') as f:
            # Handle empty file case before JSON decoding
            content = f.read().strip()
            if not content:
                logger.debug("Tasks file is empty. Returning default task.")
                # Initialize with an empty array
                with open(tasks_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                return default_task
                
            tasks = json.loads(content)

        if isinstance(tasks, list) and tasks:
            logger.info(f"Retrieved task: {tasks[0]}")
            return tasks[0] # Return the first task
        else:
            logger.debug("Tasks file contains an empty list. Returning default task.")
            return default_task
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding tasks JSON from {tasks_file}: {e}. Returning default task.")
        # Reset to an empty array since the content is invalid
        try:
            with open(tasks_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            logger.debug("Reset tasks file to an empty array.")
        except Exception as write_error:
            logger.error(f"Failed to reset tasks file: {write_error}")
        return default_task
    except Exception as e:
        logger.error(f"Error reading tasks file {tasks_file}: {e}. Returning default task.")
        return default_task

def update_task() -> None:
    """Remove the completed task (the first one) from the tasks list."""
    tasks: List[Task] = []
    task_file_path = config.TASKS_FILE
    try:
        # Read existing tasks first
        if os.path.exists(task_file_path):
            with open(task_file_path, 'r', encoding='utf-8') as f:
                try:
                    content = f.read()
                    if content:
                        tasks = json.loads(content)
                    if not isinstance(tasks, list):
                        logger.warning(f"Tasks file {task_file_path} content is not a list. Overwriting with empty list.")
                        tasks = []
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from {task_file_path} during update: {e}. Task list reset.")
                    tasks = [] # Reset tasks if file is corrupt
        else:
            logger.info(f"No tasks file found at {task_file_path}. Cannot remove task.")
            return # Nothing to do if the file doesn't exist

        # Remove the first task if the list is not empty
        if tasks:
            completed_task = tasks.pop(0)
            logger.info(f"Removed completed task: {completed_task}")
            # Write the remaining tasks back to the file
            with open(task_file_path, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, indent=4)
                logger.info(f"Tasks file updated. Remaining tasks: {len(tasks)}")
        else:
            logger.info("No tasks in the list to remove.")

    except Exception as e:
        logger.error(f"Error updating tasks file {task_file_path}: {e}")

def validate_task(task: Any) -> bool:
    """Validate that a task object has the required fields."""
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

def add_tasks(new_tasks: List[Task]) -> None:
    """Add new tasks to the end of the tasks list."""
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
            valid_tasks.append(task)
        # Invalid tasks are logged in validate_task
    
    if not valid_tasks:
        logger.warning("No valid tasks found in the provided list.")
        return
        
    tasks: List[Task] = []
    task_file_path = config.TASKS_FILE
    try:
        # Read existing tasks
        if os.path.exists(task_file_path):
            with open(task_file_path, 'r', encoding='utf-8') as f:
                try:
                    content = f.read()
                    if content:
                        tasks = json.loads(content)
                    if not isinstance(tasks, list):
                        logger.warning(f"Tasks file {task_file_path} content is not a list. Resetting before adding.")
                        tasks = []
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from {task_file_path} before adding: {e}. Task list reset.")
                    tasks = []

        # Append new tasks
        tasks.extend(valid_tasks)
        logger.info(f"Adding {len(valid_tasks)} new tasks to the queue.")

        # Write the updated list back
        with open(task_file_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=4)
            logger.info(f"Tasks file updated. Total tasks: {len(tasks)}")

    except Exception as e:
        logger.error(f"Error adding tasks to file {task_file_path}: {e}")

def add_architecture_task():
    """Add a task to analyze and improve the system architecture."""
    task = {
        "task": "architecture",
        "details": "Analyze the current architecture and implement improvements to enhance modularity, extensibility, and maintainability.",
        "priority": 5,  # High priority
        "created": datetime.now().isoformat()
    }
    
    add_tasks([task])
    logger.info(f"Added architecture improvement task: {task}")
    return task

def add_testing_task(module_path: str = None, module_type: str = "all"):
    """
    Add a task to create or run tests.
    
    Args:
        module_path: Optional specific module to test
        module_type: Type of tests ("unit", "integration", or "all")
    """
    if module_path:
        task = {
            "task": "testing",
            "details": f"Generate and run {module_type} tests for {module_path}",
            "priority": 4,  # Medium-high priority
            "created": datetime.now().isoformat(),
            "module": module_path,
            "test_type": module_type
        }
    else:
        task = {
            "task": "testing",
            "details": f"Analyze test coverage and generate tests for untested modules",
            "priority": 4,  # Medium-high priority
            "created": datetime.now().isoformat(),
            "test_type": module_type
        }
    
    add_tasks([task])
    logger.info(f"Added testing task: {task}")
    return task

def get_task_progress() -> Optional[Dict[str, Any]]:
    """
    Get the current task progress state if exists.
    Returns None if there's no ongoing task.
    """
    progress_file = config.TASK_PROGRESS_FILE
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    
    if not os.path.exists(progress_file):
        # Create an empty file for future use
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            logger.debug("Created empty task progress file.")
        except Exception as e:
            logger.error(f"Error creating task progress file: {e}")
        return None
        
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                logger.debug("Task progress file is empty. No task in progress.")
                return None
                
            progress = json.loads(content)
            if not isinstance(progress, dict):
                logger.warning("Task progress file contains invalid format (not a dict).")
                clear_task_progress()
                return None
                
            # Validate that it has the required fields
            if not all(key in progress for key in ["task", "plan", "current_step"]):
                logger.debug("Task progress file doesn't contain a valid task progress structure.")
                # Don't log a warning for an empty or new file
                return None
                
            logger.debug(f"Retrieved task progress: {progress['task']['task']}, step {progress['current_step']} of {len(progress['plan'])}")
            return progress
    except json.JSONDecodeError as e:
        logger.debug(f"Task progress file exists but doesn't contain valid JSON: {e}")
        # Reset the file instead of keeping invalid content
        clear_task_progress()
        return None
    except Exception as e:
        logger.error(f"Error reading task progress file: {e}")
        return None

def save_task_progress(task: Dict[str, Any], plan: List[str], current_step: int, results: List[str] = None) -> None:
    """
    Save the current task progress to continue in the next run.
    
    Args:
        task: The current task
        plan: The full plan for the task
        current_step: Index of the current step in the plan
        results: Optional list of execution results so far
    """
    progress = {
        "task": task,
        "plan": plan,
        "current_step": current_step,
        "results": results or [],
        "last_updated": datetime.now().isoformat()
    }
    
    try:
        with open(config.TASK_PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=4)
        logger.info(f"Saved progress for task '{task.get('task')}' at step {current_step}/{len(plan)}")
    except Exception as e:
        logger.error(f"Error saving task progress: {e}")

def clear_task_progress() -> None:
    """Clear the current task progress file."""
    if os.path.exists(config.TASK_PROGRESS_FILE):
        try:
            os.remove(config.TASK_PROGRESS_FILE)
            logger.info("Cleared task progress file")
        except Exception as e:
            logger.error(f"Error clearing task progress file: {e}")
            
def is_task_complete(task_progress: Dict[str, Any]) -> bool:
    """
    Check if a task is complete based on its progress.
    
    Args:
        task_progress: The task progress dictionary
        
    Returns:
        True if the task is complete, False otherwise
    """
    if not task_progress:
        return False
        
    # Check if we've executed all steps in the plan
    return task_progress.get("current_step", 0) >= len(task_progress.get("plan", [])) 