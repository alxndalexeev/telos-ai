"""
System configuration for Telos AI.

This module defines system-wide constants, paths, and configuration values
used throughout the Telos system. It serves as a central location for 
system-related configuration to avoid hardcoded values scattered throughout
the codebase.
"""

import os
import logging
from pathlib import Path
import datetime
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MEMORY_DIR = os.path.join(PROJECT_ROOT, "memory")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TASK_DIR = os.path.join(PROJECT_ROOT, "tasks")
AGENT_DIR = os.path.join(PROJECT_ROOT, "agents")

# Ensure required directories exist
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(MEMORY_DIR, "errors"), exist_ok=True)

# System heartbeat configuration
HEARTBEAT_INTERVAL = 30  # seconds
MIN_HEARTBEAT_INTERVAL = 15  # minimum seconds between heartbeats
MAX_HEARTBEAT_INTERVAL = 300  # maximum seconds between heartbeats
MAX_IDLE_TIME = 3600  # maximum seconds to run without activity
HEARTBEAT_FILE = os.path.join(MEMORY_DIR, "last_heartbeat.txt")
PID_FILE = os.path.join(MEMORY_DIR, "heartbeat.pid")
TASKS_FILE = os.path.join(MEMORY_DIR, "tasks.json")

# Resource limits
MAX_MEMORY_PERCENT = 80  # maximum memory usage before taking action
MAX_CPU_PERCENT = 90  # maximum CPU usage before taking action
MIN_DISK_SPACE_GB = 1  # minimum free disk space in GB

# Required and optional environment variables
REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY"
]

OPTIONAL_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT"
]

# Task configuration
DEFAULT_TASK_TIMEOUT = 300  # default timeout for tasks in seconds
MAX_CONCURRENT_TASKS = 3  # maximum number of tasks to run concurrently
TASK_HISTORY_LIMIT = 50  # number of completed tasks to keep in history

# API configuration
API_TIMEOUT = 30  # default timeout for API calls in seconds
API_RETRY_ATTEMPTS = 3  # number of times to retry API calls
API_RETRY_DELAY = 2  # seconds to wait between API call retries
API_BACKOFF_FACTOR = 2.0  # multiplicative factor for API call retry delay

# Default system prompts
DEFAULT_SYSTEM_PROMPT = """
You are Telos, an autonomous AI system designed to operate independently
and perform tasks to achieve your goals. Your purpose is to continuously
learn, improve, and provide value through completing tasks efficiently.
"""

def get_timestamp() -> str:
    """
    Get the current timestamp in a consistent format.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_config_summary() -> Dict[str, Any]:
    """
    Get a summary of the current configuration.
    
    Returns:
        Dictionary containing configuration summary
    """
    return {
        "project_root": str(PROJECT_ROOT),
        "memory_dir": MEMORY_DIR,
        "heartbeat_interval": HEARTBEAT_INTERVAL,
        "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
        "timestamp": get_timestamp(),
    }

def log_config() -> None:
    """
    Log the current configuration at startup.
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Telos AI System starting with configuration:")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Memory directory: {MEMORY_DIR}")
    logger.info(f"Heartbeat interval: {HEARTBEAT_INTERVAL} seconds")
    logger.info(f"Max concurrent tasks: {MAX_CONCURRENT_TASKS}")
    
    # Check for required environment variables
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing_vars:
        logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Log available optional variables
    available_optional = [var for var in OPTIONAL_ENV_VARS if os.environ.get(var)]
    if available_optional:
        logger.info(f"Available optional environment variables: {', '.join(available_optional)}")

def should_terminate() -> bool:
    """
    Check if the system should terminate.
    
    This checks for the existence of a termination signal file.
    
    Returns:
        True if the system should terminate, False otherwise
    """
    termination_file = os.path.join(MEMORY_DIR, "terminate")
    return os.path.exists(termination_file)

def get_system_version() -> str:
    """
    Get the current system version.
    
    Returns:
        Version string
    """
    version_file = os.path.join(PROJECT_ROOT, "VERSION")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            return f.read().strip()
    return "0.1.0"  # Default version if VERSION file doesn't exist 