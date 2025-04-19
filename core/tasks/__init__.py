"""
Tasks module for Telos AI.

This module handles task management, execution, and status tracking.
"""

from core.tasks.manager import (
    get_task,
    add_task,
    update_task,
    get_all_tasks,
    delete_task,
    add_tasks,
    add_architecture_task,
    add_testing_task,
    validate_task
)
from core.tasks.executor import (
    execute_task,
    prepare_execution
)
from core.tasks.status import (
    get_task_progress,
    save_task_progress,
    clear_task_progress,
    is_task_complete,
    get_task_status
)

__all__ = [
    'get_task',
    'add_task',
    'update_task',
    'get_all_tasks',
    'delete_task',
    'execute_task',
    'prepare_execution',
    'get_task_progress',
    'save_task_progress',
    'clear_task_progress',
    'is_task_complete',
    'get_task_status',
    'add_tasks',
    'add_architecture_task',
    'add_testing_task',
    'validate_task'
]
