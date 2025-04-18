"""
Performance Metrics Collection for Telos.

This module provides functionality for collecting and storing performance metrics
to enable performance analysis and optimization.
"""

import logging
import json
import os
import time
from datetime import datetime

# Import from configuration
import config

logger = logging.getLogger(__name__)

def log_performance_metrics(cycle_start_time, task_name, plan_length, results):
    """Log performance metrics for analytics."""
    try:
        cycle_duration = time.time() - cycle_start_time
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "task": task_name,
            "cycle_duration_sec": cycle_duration,
            "plan_steps": plan_length,
            "results_count": len(results),
            "success": not any("error" in str(r).lower() for r in results)
        }
        
        metrics_file = os.path.join(config.MEMORY_DIR, "performance_metrics.jsonl")
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    except Exception as e:
        logger.error(f"Failed to log performance metrics: {e}")

def adjust_heartbeat_interval(task_duration):
    """Adjust heartbeat interval based on task complexity."""
    # Default interval
    interval = config.HEARTBEAT_INTERVAL
    
    # If tasks are taking too long, increase interval
    if task_duration > config.HEARTBEAT_INTERVAL * 0.8:
        interval = max(task_duration * 1.2, config.HEARTBEAT_INTERVAL)
        logger.info(f"Adjusting heartbeat interval to {interval} seconds due to task duration")
    
    return interval 