"""
System Resource Monitoring for Telos.

This module provides functionality for monitoring system resources
to prevent overload and ensure stable operation.
"""

import logging
import os
import psutil

# Import from configuration
from config import MEMORY_DIR

logger = logging.getLogger(__name__)

def check_system_resources():
    """Monitor system resources to avoid overloading."""
    try:
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent(interval=1)  # 1 second measurement
        disk_percent = psutil.disk_usage('/').percent
        
        resource_warning = False
        warning_msg = ""
        
        if memory_percent > 90:
            resource_warning = True
            warning_msg += f"Memory usage critically high: {memory_percent}%. "
            
        if cpu_percent > 85:
            resource_warning = True
            warning_msg += f"CPU usage critically high: {cpu_percent}%. "
            
        if disk_percent > 95:
            resource_warning = True
            warning_msg += f"Disk usage critically high: {disk_percent}%. "
            
        if resource_warning:
            logger.warning(warning_msg)
            # Import here to avoid circular imports
            from logging.logger import log_thought
            log_thought(f"System resource warning: {warning_msg}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return True  # Continue execution despite monitoring error 