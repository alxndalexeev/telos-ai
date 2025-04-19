"""
Logging configuration for Telos AI.

This module defines logging settings, formatters, and handlers
for different components of the Telos system.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional

# Default log levels
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_CONSOLE_LOG_LEVEL = logging.INFO
DEFAULT_FILE_LOG_LEVEL = logging.DEBUG

# Log directories and files
LOG_DIR = Path("logs")
SYSTEM_LOG_FILE = LOG_DIR / "system.log"
TASK_LOG_DIR = LOG_DIR / "tasks"
ERROR_LOG_FILE = LOG_DIR / "error.log"
AUDIT_LOG_FILE = LOG_DIR / "audit.log"

# Log rotation settings
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

# Formatting
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
SIMPLE_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
CONSOLE_FORMAT = "%(levelname)s - %(message)s"

# Special loggers
AUDIT_LOGGER_NAME = "telos.audit"
TASK_LOGGER_NAME = "telos.task"
SYSTEM_LOGGER_NAME = "telos.system"

def ensure_log_directories() -> None:
    """Ensure all log directories exist."""
    LOG_DIR.mkdir(exist_ok=True)
    TASK_LOG_DIR.mkdir(exist_ok=True)

def configure_basic_logging() -> None:
    """
    Configure basic logging for the application.
    This sets up console logging and should be called early in the application lifecycle.
    """
    ensure_log_directories()
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(DEFAULT_LOG_LEVEL)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(DEFAULT_CONSOLE_LOG_LEVEL)
    console_formatter = logging.Formatter(CONSOLE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # System log file handler
    system_handler = RotatingFileHandler(
        SYSTEM_LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )
    system_handler.setLevel(DEFAULT_FILE_LOG_LEVEL)
    system_formatter = logging.Formatter(DETAILED_FORMAT)
    system_handler.setFormatter(system_formatter)
    root_logger.addHandler(system_handler)
    
    # Error log file handler (errors only)
    error_handler = RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(DETAILED_FORMAT)
    error_handler.setFormatter(error_formatter)
    root_logger.addHandler(error_handler)
    
    logging.info("Basic logging configured")

def get_task_logger(task_id: str) -> logging.Logger:
    """
    Get a logger for a specific task with appropriate handlers.
    
    Args:
        task_id: Unique identifier for the task
        
    Returns:
        Logger configured for the task
    """
    ensure_log_directories()
    
    # Create task-specific log file
    task_log_file = TASK_LOG_DIR / f"{task_id}.log"
    
    # Create logger
    logger_name = f"{TASK_LOGGER_NAME}.{task_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(DEFAULT_LOG_LEVEL)
    
    # Check if handlers already exist
    if not logger.handlers:
        # Task-specific file handler
        task_handler = RotatingFileHandler(
            task_log_file,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT
        )
        task_handler.setLevel(DEFAULT_FILE_LOG_LEVEL)
        task_formatter = logging.Formatter(DETAILED_FORMAT)
        task_handler.setFormatter(task_formatter)
        logger.addHandler(task_handler)
    
    return logger

def get_audit_logger() -> logging.Logger:
    """
    Get the audit logger for security and compliance logging.
    
    Returns:
        Logger configured for audit logging
    """
    ensure_log_directories()
    
    # Create logger
    logger = logging.getLogger(AUDIT_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist
    if not logger.handlers:
        # Audit file handler
        audit_handler = RotatingFileHandler(
            AUDIT_LOG_FILE,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT
        )
        audit_handler.setLevel(logging.INFO)
        audit_formatter = logging.Formatter(DETAILED_FORMAT)
        audit_handler.setFormatter(audit_formatter)
        logger.addHandler(audit_handler)
    
    return logger

def get_system_logger() -> logging.Logger:
    """
    Get the system logger for core system operations.
    
    Returns:
        Logger configured for system logging
    """
    return logging.getLogger(SYSTEM_LOGGER_NAME)

def set_log_level(level: int, logger_name: Optional[str] = None) -> None:
    """
    Set the log level for a specific logger or the root logger.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        logger_name: Optional name of logger to configure, None for root logger
    """
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(level)
    
    # Also update handlers
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            # Keep console handlers at INFO or higher to avoid terminal clutter
            handler.setLevel(max(level, logging.INFO))
        else:
            handler.setLevel(level)

def get_log_config() -> Dict[str, Any]:
    """
    Get the current logging configuration as a dictionary.
    
    Returns:
        Dictionary with log configuration details
    """
    return {
        "log_directory": str(LOG_DIR),
        "system_log": str(SYSTEM_LOG_FILE),
        "error_log": str(ERROR_LOG_FILE),
        "audit_log": str(AUDIT_LOG_FILE),
        "task_logs_directory": str(TASK_LOG_DIR),
        "default_level": logging.getLevelName(DEFAULT_LOG_LEVEL),
        "console_level": logging.getLevelName(DEFAULT_CONSOLE_LOG_LEVEL),
        "file_level": logging.getLevelName(DEFAULT_FILE_LOG_LEVEL),
        "max_log_size": MAX_LOG_SIZE,
        "backup_count": BACKUP_COUNT,
    } 