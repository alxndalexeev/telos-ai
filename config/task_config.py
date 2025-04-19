"""
Task configuration for Telos AI.

This module provides configuration settings for task management,
including task types, priorities, scheduling, and execution settings.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
import datetime

class TaskPriority(Enum):
    """Enumeration of task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"

class TaskStatus(Enum):
    """Enumeration of task status states."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class TaskType(Enum):
    """Enumeration of task types."""
    SYSTEM = "system"
    USER = "user"
    TIMER = "timer"
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    MAINTENANCE = "maintenance"
    LEARNING = "learning"
    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"

class ScheduleType(Enum):
    """Enumeration of schedule types."""
    ONCE = "once"
    INTERVAL = "interval"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"
    ON_EVENT = "on_event"
    ON_CONDITION = "on_condition"

class ExecutionMode(Enum):
    """Enumeration of task execution modes."""
    SYNC = "sync"
    ASYNC = "async"
    PARALLELIZED = "parallelized"
    BATCHED = "batched"
    SEQUENTIAL = "sequential"
    DISTRIBUTED = "distributed"

# Global task settings
GLOBAL_TASK_CONFIG = {
    "enabled": True,
    "max_concurrent_tasks": 10,
    "max_queue_size": 100,
    "default_timeout_seconds": 300,
    "default_retry_count": 3,
    "default_priority": TaskPriority.MEDIUM,
    "default_execution_mode": ExecutionMode.ASYNC,
    "log_task_execution": True,
    "log_task_results": True,
    "enable_profiling": True,
    "task_history_days": 7,
    "allow_task_cancellation": True,
    "db_storage_path": "./data/tasks",
    "heartbeat_interval_seconds": 10,
    "enable_deadlock_detection": True,
    "scheduler_polling_interval_seconds": 1,
    "task_queue_check_interval_seconds": 0.5,
    "worker_count": 4,
    "max_memory_percent": 80,
    "max_cpu_percent": 90,
    "enable_priority_queue": True,
    "task_result_ttl_days": 30,
    "enable_periodic_cleanup": True,
    "cleanup_interval_hours": 24,
    "task_progress_update_interval_seconds": 2,
    "enable_task_dependencies": True,
    "default_schedule_type": ScheduleType.ONCE,
    "enable_task_scheduling": True,
    "scheduler_resolution_seconds": 1,
}

# Task type configurations
TASK_TYPE_CONFIGS = {
    TaskType.SYSTEM: {
        "priority": TaskPriority.HIGH,
        "timeout_seconds": 120,
        "retry_count": 5,
        "execution_mode": ExecutionMode.ASYNC,
        "allow_cancellation": False,
        "log_level": "INFO",
        "worker_threads": 2,
    },
    TaskType.USER: {
        "priority": TaskPriority.HIGH,
        "timeout_seconds": 300,
        "retry_count": 2,
        "execution_mode": ExecutionMode.ASYNC,
        "allow_cancellation": True,
        "log_level": "INFO",
        "worker_threads": 4,
    },
    TaskType.TIMER: {
        "priority": TaskPriority.MEDIUM,
        "timeout_seconds": 60,
        "retry_count": 3,
        "execution_mode": ExecutionMode.ASYNC,
        "allow_cancellation": True,
        "log_level": "INFO",
        "worker_threads": 1,
    },
    TaskType.REACTIVE: {
        "priority": TaskPriority.HIGH,
        "timeout_seconds": 180,
        "retry_count": 3,
        "execution_mode": ExecutionMode.ASYNC,
        "allow_cancellation": True,
        "log_level": "INFO",
        "worker_threads": 2,
    },
    TaskType.PROACTIVE: {
        "priority": TaskPriority.MEDIUM,
        "timeout_seconds": 600,
        "retry_count": 2,
        "execution_mode": ExecutionMode.ASYNC,
        "allow_cancellation": True,
        "log_level": "INFO",
        "worker_threads": 2,
    },
    TaskType.MAINTENANCE: {
        "priority": TaskPriority.LOW,
        "timeout_seconds": 1800,
        "retry_count": 3,
        "execution_mode": ExecutionMode.ASYNC,
        "allow_cancellation": True,
        "log_level": "INFO",
        "worker_threads": 1,
    },
    TaskType.LEARNING: {
        "priority": TaskPriority.LOW,
        "timeout_seconds": 3600,
        "retry_count": 2,
        "execution_mode": ExecutionMode.ASYNC,
        "allow_cancellation": True,
        "log_level": "INFO",
        "worker_threads": 2,
    },
    TaskType.PLANNING: {
        "priority": TaskPriority.MEDIUM,
        "timeout_seconds": 300,
        "retry_count": 2,
        "execution_mode": ExecutionMode.ASYNC,
        "allow_cancellation": True,
        "log_level": "INFO",
        "worker_threads": 1,
    },
    TaskType.EXECUTION: {
        "priority": TaskPriority.HIGH,
        "timeout_seconds": 600,
        "retry_count": 3,
        "execution_mode": ExecutionMode.ASYNC,
        "allow_cancellation": True,
        "log_level": "INFO",
        "worker_threads": 2,
    },
    TaskType.REFLECTION: {
        "priority": TaskPriority.LOW,
        "timeout_seconds": 900,
        "retry_count": 1,
        "execution_mode": ExecutionMode.ASYNC,
        "allow_cancellation": True,
        "log_level": "INFO",
        "worker_threads": 1,
    },
}

# Priority configurations
PRIORITY_CONFIGS = {
    TaskPriority.CRITICAL: {
        "queue_position": "front",
        "preemption_enabled": True,
        "max_runtime_seconds": 600,
        "resource_allocation_factor": 1.5,
        "notification_on_start": True,
        "notification_on_complete": True,
        "notification_on_fail": True,
        "log_level": "INFO",
    },
    TaskPriority.HIGH: {
        "queue_position": "front",
        "preemption_enabled": False,
        "max_runtime_seconds": 1200,
        "resource_allocation_factor": 1.2,
        "notification_on_start": False,
        "notification_on_complete": True,
        "notification_on_fail": True,
        "log_level": "INFO",
    },
    TaskPriority.MEDIUM: {
        "queue_position": "middle",
        "preemption_enabled": False,
        "max_runtime_seconds": 1800,
        "resource_allocation_factor": 1.0,
        "notification_on_start": False,
        "notification_on_complete": False,
        "notification_on_fail": True,
        "log_level": "INFO",
    },
    TaskPriority.LOW: {
        "queue_position": "back",
        "preemption_enabled": False,
        "max_runtime_seconds": 3600,
        "resource_allocation_factor": 0.8,
        "notification_on_start": False,
        "notification_on_complete": False,
        "notification_on_fail": False,
        "log_level": "INFO",
    },
    TaskPriority.BACKGROUND: {
        "queue_position": "back",
        "preemption_enabled": False,
        "max_runtime_seconds": 7200,
        "resource_allocation_factor": 0.5,
        "notification_on_start": False,
        "notification_on_complete": False,
        "notification_on_fail": False,
        "log_level": "INFO",
    },
}

# Scheduling configurations
SCHEDULE_TYPE_CONFIGS = {
    ScheduleType.ONCE: {
        "retry_on_failure": True,
        "reschedule_on_system_restart": True,
        "run_missed": True,
        "notification_threshold_seconds": 300,  # Notify if task runs >5min late
    },
    ScheduleType.INTERVAL: {
        "default_interval_seconds": 300,
        "min_interval_seconds": 10,
        "retry_on_failure": True,
        "skip_if_still_running": True,
        "notification_threshold_seconds": 120,
    },
    ScheduleType.DAILY: {
        "default_time": "00:00:00",
        "retry_on_failure": True,
        "run_missed": True,
        "notification_threshold_minutes": 15,
    },
    ScheduleType.WEEKLY: {
        "default_day": "Monday",
        "default_time": "00:00:00",
        "retry_on_failure": True,
        "run_missed": True,
        "notification_threshold_minutes": 30,
    },
    ScheduleType.MONTHLY: {
        "default_day": 1,
        "default_time": "00:00:00",
        "retry_on_failure": True,
        "run_missed": True,
        "notification_threshold_minutes": 30,
    },
    ScheduleType.ON_EVENT: {
        "retry_on_failure": True,
        "event_deduplication_seconds": 10,
        "max_trigger_frequency_seconds": 1,
    },
    ScheduleType.ON_CONDITION: {
        "polling_interval_seconds": 60,
        "retry_on_failure": True,
        "condition_check_timeout_seconds": 10,
    },
}

# Common task presets
TASK_PRESETS = {
    "quick_execution": {
        "priority": TaskPriority.HIGH,
        "timeout_seconds": 60,
        "retry_count": 1,
        "execution_mode": ExecutionMode.SYNC,
        "description": "For tasks that need to complete quickly with immediate results",
    },
    "background_processing": {
        "priority": TaskPriority.BACKGROUND,
        "timeout_seconds": 3600,
        "retry_count": 3,
        "execution_mode": ExecutionMode.ASYNC,
        "description": "For long-running tasks that can be processed in the background",
    },
    "critical_operation": {
        "priority": TaskPriority.CRITICAL,
        "timeout_seconds": 300,
        "retry_count": 5,
        "execution_mode": ExecutionMode.SYNC,
        "description": "For critical operations that must succeed and can't be interrupted",
    },
    "data_intensive": {
        "priority": TaskPriority.MEDIUM,
        "timeout_seconds": 1800,
        "retry_count": 2,
        "execution_mode": ExecutionMode.ASYNC,
        "description": "For tasks that process large amounts of data and may take time",
    },
    "recurring_maintenance": {
        "priority": TaskPriority.LOW,
        "timeout_seconds": 1200,
        "retry_count": 3,
        "execution_mode": ExecutionMode.ASYNC,
        "schedule_type": ScheduleType.INTERVAL,
        "interval_seconds": 3600 * 24,  # Daily
        "description": "For recurring system maintenance tasks",
    },
    "user_interactive": {
        "priority": TaskPriority.HIGH,
        "timeout_seconds": 120,
        "retry_count": 2,
        "execution_mode": ExecutionMode.SYNC,
        "description": "For tasks that require interaction with the user and quick responses",
    },
    "distributed_processing": {
        "priority": TaskPriority.MEDIUM,
        "timeout_seconds": 3600,
        "retry_count": 3,
        "execution_mode": ExecutionMode.DISTRIBUTED,
        "description": "For tasks that can be split and processed across multiple workers",
    },
}

# System task configurations
SYSTEM_TASKS = {
    "memory_cleanup": {
        "type": TaskType.MAINTENANCE,
        "priority": TaskPriority.LOW,
        "schedule_type": ScheduleType.INTERVAL,
        "interval_seconds": 3600 * 24,  # Daily
        "timeout_seconds": 1800,
        "description": "Cleans up old memories and optimizes memory storage",
        "enabled": True,
    },
    "heartbeat": {
        "type": TaskType.SYSTEM,
        "priority": TaskPriority.MEDIUM,
        "schedule_type": ScheduleType.INTERVAL,
        "interval_seconds": 60,
        "timeout_seconds": 10,
        "description": "Regular system heartbeat to ensure system health",
        "enabled": True,
    },
    "self_monitor": {
        "type": TaskType.SYSTEM,
        "priority": TaskPriority.HIGH,
        "schedule_type": ScheduleType.INTERVAL,
        "interval_seconds": 300,
        "timeout_seconds": 120,
        "description": "Monitors system health and performance",
        "enabled": True,
    },
    "log_rotation": {
        "type": TaskType.MAINTENANCE,
        "priority": TaskPriority.LOW,
        "schedule_type": ScheduleType.DAILY,
        "default_time": "01:00:00",
        "timeout_seconds": 600,
        "description": "Rotates and compresses log files",
        "enabled": True,
    },
    "database_backup": {
        "type": TaskType.MAINTENANCE,
        "priority": TaskPriority.MEDIUM,
        "schedule_type": ScheduleType.DAILY,
        "default_time": "03:00:00",
        "timeout_seconds": 1800,
        "description": "Backs up the database",
        "enabled": True,
    },
    "knowledge_update": {
        "type": TaskType.LEARNING,
        "priority": TaskPriority.LOW,
        "schedule_type": ScheduleType.INTERVAL,
        "interval_seconds": 3600 * 6,  # Every 6 hours
        "timeout_seconds": 1800,
        "description": "Updates knowledge base with new information",
        "enabled": True,
    },
    "memory_consolidation": {
        "type": TaskType.LEARNING,
        "priority": TaskPriority.LOW,
        "schedule_type": ScheduleType.DAILY,
        "default_time": "02:00:00",
        "timeout_seconds": 3600,
        "description": "Consolidates recent memories into long-term storage",
        "enabled": True,
    },
    "system_reflection": {
        "type": TaskType.REFLECTION,
        "priority": TaskPriority.LOW,
        "schedule_type": ScheduleType.DAILY,
        "default_time": "04:00:00",
        "timeout_seconds": 1800,
        "description": "Reviews system performance and generates improvement ideas",
        "enabled": True,
    },
}

# Function to get global task configuration
def get_global_task_config() -> Dict[str, Any]:
    """
    Get global task configuration.
    
    Returns:
        Dictionary with global task configuration
    """
    return GLOBAL_TASK_CONFIG.copy()

# Function to get task type configuration
def get_task_type_config(task_type: TaskType) -> Dict[str, Any]:
    """
    Get configuration for a specific task type.
    
    Args:
        task_type: The task type
        
    Returns:
        Dictionary with task type configuration
    """
    if task_type in TASK_TYPE_CONFIGS:
        return TASK_TYPE_CONFIGS[task_type].copy()
    
    return {}

# Function to get task priority configuration
def get_priority_config(priority: TaskPriority) -> Dict[str, Any]:
    """
    Get configuration for a specific task priority.
    
    Args:
        priority: The task priority
        
    Returns:
        Dictionary with priority configuration
    """
    if priority in PRIORITY_CONFIGS:
        return PRIORITY_CONFIGS[priority].copy()
    
    return {}

# Function to get schedule type configuration
def get_schedule_type_config(schedule_type: ScheduleType) -> Dict[str, Any]:
    """
    Get configuration for a specific schedule type.
    
    Args:
        schedule_type: The schedule type
        
    Returns:
        Dictionary with schedule type configuration
    """
    if schedule_type in SCHEDULE_TYPE_CONFIGS:
        return SCHEDULE_TYPE_CONFIGS[schedule_type].copy()
    
    return {}

# Function to get task preset configuration
def get_task_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get configuration for a task preset.
    
    Args:
        preset_name: The name of the preset
        
    Returns:
        Dictionary with preset configuration
    """
    if preset_name in TASK_PRESETS:
        return TASK_PRESETS[preset_name].copy()
    
    return {}

# Function to get system task configuration
def get_system_task_config(task_name: str) -> Dict[str, Any]:
    """
    Get configuration for a system task.
    
    Args:
        task_name: The name of the system task
        
    Returns:
        Dictionary with system task configuration
    """
    if task_name in SYSTEM_TASKS:
        return SYSTEM_TASKS[task_name].copy()
    
    return {}

# Function to create a task configuration with defaults
def create_task_config(
    task_type: TaskType,
    priority: Optional[TaskPriority] = None,
    timeout_seconds: Optional[int] = None,
    retry_count: Optional[int] = None,
    execution_mode: Optional[ExecutionMode] = None,
    schedule_type: Optional[ScheduleType] = None,
    schedule_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a task configuration with defaults from the task type.
    
    Args:
        task_type: The type of task
        priority: Optional priority override
        timeout_seconds: Optional timeout override
        retry_count: Optional retry count override
        execution_mode: Optional execution mode override
        schedule_type: Optional schedule type
        schedule_params: Optional schedule parameters
        
    Returns:
        Dictionary with complete task configuration
    """
    # Get base configuration from task type
    task_config = get_task_type_config(task_type)
    
    # Apply overrides if provided
    if priority is not None:
        task_config["priority"] = priority
    
    if timeout_seconds is not None:
        task_config["timeout_seconds"] = timeout_seconds
    
    if retry_count is not None:
        task_config["retry_count"] = retry_count
    
    if execution_mode is not None:
        task_config["execution_mode"] = execution_mode
    
    # Add scheduling information if provided
    if schedule_type is not None:
        task_config["schedule_type"] = schedule_type
        
        # Get default schedule configuration
        schedule_config = get_schedule_type_config(schedule_type)
        
        # Apply schedule parameters
        if schedule_params is not None:
            for key, value in schedule_params.items():
                schedule_config[key] = value
        
        task_config["schedule_config"] = schedule_config
    
    return task_config

# Function to create a scheduled task
def create_scheduled_task(
    task_type: TaskType,
    schedule_type: ScheduleType,
    schedule_params: Dict[str, Any],
    priority: Optional[TaskPriority] = None,
    timeout_seconds: Optional[int] = None,
    retry_count: Optional[int] = None,
    execution_mode: Optional[ExecutionMode] = None
) -> Dict[str, Any]:
    """
    Create a configuration for a scheduled task.
    
    Args:
        task_type: The type of task
        schedule_type: The schedule type
        schedule_params: Schedule parameters
        priority: Optional priority override
        timeout_seconds: Optional timeout override
        retry_count: Optional retry count override
        execution_mode: Optional execution mode override
        
    Returns:
        Dictionary with complete task configuration
    """
    return create_task_config(
        task_type=task_type,
        priority=priority,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
        execution_mode=execution_mode,
        schedule_type=schedule_type,
        schedule_params=schedule_params
    )

# Function to create a one-time scheduled task
def create_one_time_task(
    task_type: TaskType,
    run_at: datetime.datetime,
    priority: Optional[TaskPriority] = None,
    timeout_seconds: Optional[int] = None,
    retry_count: Optional[int] = None,
    execution_mode: Optional[ExecutionMode] = None
) -> Dict[str, Any]:
    """
    Create a configuration for a one-time scheduled task.
    
    Args:
        task_type: The type of task
        run_at: When to run the task
        priority: Optional priority override
        timeout_seconds: Optional timeout override
        retry_count: Optional retry count override
        execution_mode: Optional execution mode override
        
    Returns:
        Dictionary with complete task configuration
    """
    schedule_params = {
        "run_at": run_at,
        "retry_on_failure": True,
        "reschedule_on_system_restart": True,
    }
    
    return create_scheduled_task(
        task_type=task_type,
        schedule_type=ScheduleType.ONCE,
        schedule_params=schedule_params,
        priority=priority,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
        execution_mode=execution_mode
    )

# Function to create an interval scheduled task
def create_interval_task(
    task_type: TaskType,
    interval_seconds: int,
    priority: Optional[TaskPriority] = None,
    timeout_seconds: Optional[int] = None,
    retry_count: Optional[int] = None,
    execution_mode: Optional[ExecutionMode] = None
) -> Dict[str, Any]:
    """
    Create a configuration for an interval scheduled task.
    
    Args:
        task_type: The type of task
        interval_seconds: Interval between executions in seconds
        priority: Optional priority override
        timeout_seconds: Optional timeout override
        retry_count: Optional retry count override
        execution_mode: Optional execution mode override
        
    Returns:
        Dictionary with complete task configuration
    """
    schedule_params = {
        "interval_seconds": interval_seconds,
        "retry_on_failure": True,
        "skip_if_still_running": True,
    }
    
    return create_scheduled_task(
        task_type=task_type,
        schedule_type=ScheduleType.INTERVAL,
        schedule_params=schedule_params,
        priority=priority,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
        execution_mode=execution_mode
    )

# Function to create a daily scheduled task
def create_daily_task(
    task_type: TaskType,
    time_str: str,
    priority: Optional[TaskPriority] = None,
    timeout_seconds: Optional[int] = None,
    retry_count: Optional[int] = None,
    execution_mode: Optional[ExecutionMode] = None
) -> Dict[str, Any]:
    """
    Create a configuration for a daily scheduled task.
    
    Args:
        task_type: The type of task
        time_str: Time of day to run (HH:MM:SS)
        priority: Optional priority override
        timeout_seconds: Optional timeout override
        retry_count: Optional retry count override
        execution_mode: Optional execution mode override
        
    Returns:
        Dictionary with complete task configuration
    """
    schedule_params = {
        "time": time_str,
        "retry_on_failure": True,
        "run_missed": True,
    }
    
    return create_scheduled_task(
        task_type=task_type,
        schedule_type=ScheduleType.DAILY,
        schedule_params=schedule_params,
        priority=priority,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
        execution_mode=execution_mode
    )

# Function to create a weekly scheduled task
def create_weekly_task(
    task_type: TaskType,
    day_of_week: str,
    time_str: str,
    priority: Optional[TaskPriority] = None,
    timeout_seconds: Optional[int] = None,
    retry_count: Optional[int] = None,
    execution_mode: Optional[ExecutionMode] = None
) -> Dict[str, Any]:
    """
    Create a configuration for a weekly scheduled task.
    
    Args:
        task_type: The type of task
        day_of_week: Day of week to run
        time_str: Time of day to run (HH:MM:SS)
        priority: Optional priority override
        timeout_seconds: Optional timeout override
        retry_count: Optional retry count override
        execution_mode: Optional execution mode override
        
    Returns:
        Dictionary with complete task configuration
    """
    schedule_params = {
        "day": day_of_week,
        "time": time_str,
        "retry_on_failure": True,
        "run_missed": True,
    }
    
    return create_scheduled_task(
        task_type=task_type,
        schedule_type=ScheduleType.WEEKLY,
        schedule_params=schedule_params,
        priority=priority,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
        execution_mode=execution_mode
    )

# Function to create a monthly scheduled task
def create_monthly_task(
    task_type: TaskType,
    day_of_month: int,
    time_str: str,
    priority: Optional[TaskPriority] = None,
    timeout_seconds: Optional[int] = None,
    retry_count: Optional[int] = None,
    execution_mode: Optional[ExecutionMode] = None
) -> Dict[str, Any]:
    """
    Create a configuration for a monthly scheduled task.
    
    Args:
        task_type: The type of task
        day_of_month: Day of month to run (1-31)
        time_str: Time of day to run (HH:MM:SS)
        priority: Optional priority override
        timeout_seconds: Optional timeout override
        retry_count: Optional retry count override
        execution_mode: Optional execution mode override
        
    Returns:
        Dictionary with complete task configuration
    """
    schedule_params = {
        "day": day_of_month,
        "time": time_str,
        "retry_on_failure": True,
        "run_missed": True,
    }
    
    return create_scheduled_task(
        task_type=task_type,
        schedule_type=ScheduleType.MONTHLY,
        schedule_params=schedule_params,
        priority=priority,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
        execution_mode=execution_mode
    )

# Function to create an event-triggered task
def create_event_triggered_task(
    task_type: TaskType,
    event_type: str,
    priority: Optional[TaskPriority] = None,
    timeout_seconds: Optional[int] = None,
    retry_count: Optional[int] = None,
    execution_mode: Optional[ExecutionMode] = None,
    event_filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a configuration for an event-triggered task.
    
    Args:
        task_type: The type of task
        event_type: Type of event that triggers the task
        priority: Optional priority override
        timeout_seconds: Optional timeout override
        retry_count: Optional retry count override
        execution_mode: Optional execution mode override
        event_filters: Optional filters for event matching
        
    Returns:
        Dictionary with complete task configuration
    """
    schedule_params = {
        "event_type": event_type,
        "retry_on_failure": True,
        "event_deduplication_seconds": 10,
    }
    
    if event_filters is not None:
        schedule_params["event_filters"] = event_filters
    
    return create_scheduled_task(
        task_type=task_type,
        schedule_type=ScheduleType.ON_EVENT,
        schedule_params=schedule_params,
        priority=priority,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
        execution_mode=execution_mode
    )

# Function to create a condition-triggered task
def create_condition_triggered_task(
    task_type: TaskType,
    condition_function: str,
    polling_interval_seconds: int = 60,
    priority: Optional[TaskPriority] = None,
    timeout_seconds: Optional[int] = None,
    retry_count: Optional[int] = None,
    execution_mode: Optional[ExecutionMode] = None,
    condition_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a configuration for a condition-triggered task.
    
    Args:
        task_type: The type of task
        condition_function: Name of function that checks condition
        polling_interval_seconds: Interval to check condition
        priority: Optional priority override
        timeout_seconds: Optional timeout override
        retry_count: Optional retry count override
        execution_mode: Optional execution mode override
        condition_args: Optional arguments for condition function
        
    Returns:
        Dictionary with complete task configuration
    """
    schedule_params = {
        "condition_function": condition_function,
        "polling_interval_seconds": polling_interval_seconds,
        "retry_on_failure": True,
    }
    
    if condition_args is not None:
        schedule_params["condition_args"] = condition_args
    
    return create_scheduled_task(
        task_type=task_type,
        schedule_type=ScheduleType.ON_CONDITION,
        schedule_params=schedule_params,
        priority=priority,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
        execution_mode=execution_mode
    )

# Function to get all system tasks
def get_all_system_tasks() -> Dict[str, Dict[str, Any]]:
    """
    Get configurations for all system tasks.
    
    Returns:
        Dictionary with all system task configurations
    """
    return SYSTEM_TASKS.copy()

# Function to get all enabled system tasks
def get_enabled_system_tasks() -> Dict[str, Dict[str, Any]]:
    """
    Get configurations for all enabled system tasks.
    
    Returns:
        Dictionary with enabled system task configurations
    """
    return {name: config for name, config in SYSTEM_TASKS.items() if config.get("enabled", True)}

# Function to validate task configuration
def validate_task_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a task configuration.
    
    Args:
        config: The task configuration to validate
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Basic validation
    if not isinstance(config, dict):
        errors.append("Configuration must be a dictionary")
        return errors
    
    # Required fields
    required_fields = ["type"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate task type
    if "type" in config:
        task_type = config["type"]
        if not isinstance(task_type, TaskType):
            errors.append("Task type must be a TaskType enum value")
    
    # Validate priority
    if "priority" in config:
        priority = config["priority"]
        if not isinstance(priority, TaskPriority):
            errors.append("Priority must be a TaskPriority enum value")
    
    # Validate execution mode
    if "execution_mode" in config:
        execution_mode = config["execution_mode"]
        if not isinstance(execution_mode, ExecutionMode):
            errors.append("Execution mode must be an ExecutionMode enum value")
    
    # Validate schedule type and config
    if "schedule_type" in config:
        schedule_type = config["schedule_type"]
        if not isinstance(schedule_type, ScheduleType):
            errors.append("Schedule type must be a ScheduleType enum value")
        
        # Check for schedule config based on schedule type
        if "schedule_config" not in config:
            errors.append("Schedule configuration is required when schedule_type is specified")
        else:
            schedule_config = config["schedule_config"]
            
            # Validate specific schedule configs
            if schedule_type == ScheduleType.INTERVAL and "interval_seconds" not in schedule_config:
                errors.append("Interval schedule requires 'interval_seconds' parameter")
                
            if schedule_type == ScheduleType.DAILY and "time" not in schedule_config:
                errors.append("Daily schedule requires 'time' parameter")
                
            if schedule_type == ScheduleType.WEEKLY and ("day" not in schedule_config or "time" not in schedule_config):
                errors.append("Weekly schedule requires 'day' and 'time' parameters")
                
            if schedule_type == ScheduleType.MONTHLY and ("day" not in schedule_config or "time" not in schedule_config):
                errors.append("Monthly schedule requires 'day' and 'time' parameters")
                
            if schedule_type == ScheduleType.ON_EVENT and "event_type" not in schedule_config:
                errors.append("Event-triggered schedule requires 'event_type' parameter")
                
            if schedule_type == ScheduleType.ON_CONDITION and "condition_function" not in schedule_config:
                errors.append("Condition-triggered schedule requires 'condition_function' parameter")
                
    return errors 