"""
Task executor module for Telos AI.

This module handles task execution and preparation.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple

# Import configurations
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

logger = logging.getLogger(__name__)

def execute_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a task using the decision-making and review process.
    
    This is the main entry point for task execution that orchestrates
    the entire process of context gathering, planning, reviewing, 
    and execution.
    
    Args:
        task: Task information
        
    Returns:
        Results dictionary with execution outcomes
    """
    # Import here to avoid circular imports
    from core import context_gatherer
    from core.review.decision import make_decision, execute_decision
    from core.notification_service import notify_task_started, notify_task_completed
    from core.tasks.status import save_task_progress
    
    # Get task details
    task_id = task.get('id')
    task_name = task.get('task', 'N/A')
    
    start_time = time.time()
    logger.info(f"Starting execution of task: {task_name}")
    
    # Notify that task has started
    notify_task_started(task)
    
    try:
        # Prepare for execution by gathering context
        context = prepare_execution(task)
        save_task_progress(task_id, 20, "Context gathered")
        
        # Make a decision on what to do (planning and reviewing)
        plan, decision_metadata = make_decision(task, context)
        save_task_progress(task_id, 50, "Plan created and reviewed")
        
        # Execute the decision
        execution_results = execute_decision(task, plan, decision_metadata)
        
        # Record execution time
        execution_time = time.time() - start_time
        execution_results['execution_time'] = execution_time
        
        # Log and notify completion
        if execution_results.get('success'):
            notify_task_completed(task)
            logger.info(f"Task completed successfully: {task_name} (took {execution_time:.2f}s)")
        else:
            logger.warning(f"Task execution had issues: {task_name} (took {execution_time:.2f}s)")
            logger.warning(f"Error: {execution_results.get('error')}")
        
        return execution_results
    except Exception as e:
        logger.error(f"Error executing task {task_name}: {e}")
        return {
            'success': False,
            'error': f"Error executing task: {str(e)}",
            'execution_time': time.time() - start_time,
            'task_status': 'failed'
        }

def prepare_execution(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare for task execution by gathering context.
    
    Args:
        task: Task information
        
    Returns:
        Context dictionary
    """
    # Import here to avoid circular imports
    from core import context_gatherer
    
    task_name = task.get('task', 'N/A')
    logger.info(f"Preparing execution for task: {task_name}")
    
    try:
        # Gather context
        context = context_gatherer.gather_context(task)
        logger.info(f"Gathered {len(context)} context items for task {task_name}")
        
        # Add task metadata to context if available
        if 'metadata' in task:
            context['task_metadata'] = task['metadata']
        
        return context
    except Exception as e:
        logger.error(f"Error preparing execution for task {task_name}: {e}")
        # Return minimal context to avoid complete failure
        return {'error': f"Error gathering context: {str(e)}"}

def run_subtask(
    parent_task: Dict[str, Any],
    subtask_name: str,
    subtask_details: str
) -> Dict[str, Any]:
    """
    Run a subtask within the context of a parent task.
    
    Args:
        parent_task: Parent task information
        subtask_name: Name of the subtask
        subtask_details: Details of the subtask
        
    Returns:
        Results of the subtask execution
    """
    # Import here to avoid circular imports
    from core.tasks.manager import add_task, get_task
    
    # Create metadata linking to parent task
    metadata = {
        'parent_task_id': parent_task.get('id'),
        'is_subtask': True
    }
    
    # Inherit parent task tags if they exist
    tags = parent_task.get('tags', [])
    if 'subtask' not in tags:
        tags = tags + ['subtask']
    
    # Add the subtask
    subtask_id = add_task(
        task_name=subtask_name,
        details=subtask_details,
        priority=parent_task.get('priority'),
        tags=tags,
        metadata=metadata
    )
    
    if not subtask_id:
        logger.error(f"Failed to create subtask {subtask_name}")
        return {
            'success': False,
            'error': "Failed to create subtask"
        }
    
    # Get the subtask
    subtask = get_task(subtask_id)
    if not subtask:
        logger.error(f"Failed to retrieve created subtask {subtask_id}")
        return {
            'success': False,
            'error': "Failed to retrieve created subtask"
        }
    
    # Execute the subtask
    logger.info(f"Executing subtask {subtask_name} (ID: {subtask_id})")
    return execute_task(subtask) 