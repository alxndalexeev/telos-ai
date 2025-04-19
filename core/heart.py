"""
Heart module for Telos AI.

This module contains the main autonomous loop (heart) of Telos AI, 
implementing the Memento pattern. It focuses on orchestrating the components 
rather than performing detailed work itself.
"""

import time
import logging
import sys
import os
from datetime import datetime
import psutil
from typing import Dict, List, Optional, Any

# Import configurations
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import system as system_config

# Import from modules
from core.tasks import get_task, update_task, execute_task
from core.notification_service import (
    notify_startup,
    notify_shutdown,
    notify_error
)
from monitoring.resource_monitor import check_system_resources
from monitoring.performance import log_performance_metrics
from core.error_handling import (
    adjust_error_handling, 
    recover_heartbeat,
    log_exception,
    safe_execute
)

logger = logging.getLogger(__name__)

def check_environment() -> bool:
    """Check if all required environment variables are set."""
    import os
    
    missing_keys = []
    # Critical APIs
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
        
    # Optional but recommended APIs
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY not found in environment. Search functionality will be limited.")
    
    # Check for Pinecone API key if vector DB is enabled
    from config import vector_db
    if not os.getenv("PINECONE_API_KEY") and vector_db.USE_VECTOR_DB:
        logger.warning("PINECONE_API_KEY not found but vector DB is enabled. Vector storage will not work.")
        
    if missing_keys:
        logger.critical(f"Missing critical environment variables: {', '.join(missing_keys)}")
        print(f"\n⚠️  ERROR: Missing required API keys: {', '.join(missing_keys)}")
        print("Please create a .env file with these keys or set them in your environment.")
        print("See .env.example for the required format.")
        return False
    
    return True

def record_heartbeat_status() -> None:
    """Record heartbeat pulse for monitoring."""
    try:
        pid = os.getpid()
        with open(os.path.join(system_config.MEMORY_DIR, "heartbeat.pid"), "w") as f:
            f.write(str(pid))
        
        # Record timestamp of last successful beat
        with open(os.path.join(system_config.MEMORY_DIR, "last_heartbeat.txt"), "w") as f:
            f.write(datetime.now().isoformat())
    except Exception as e:
        logger.error(f"Failed to record heartbeat status: {e}")

def heart_beat() -> None:
    """
    Main heart loop for Telos:
    1. Wake up with no memory of previous state
    2. Check system resources
    3. Get the next task
    4. Execute the task using specialized modules
    5. Go to sleep (forgetting this session)
    
    This implementation focuses on orchestrating the components rather than
    performing detailed work itself.
    """
    # Ensure memory directory exists
    os.makedirs(system_config.MEMORY_DIR, exist_ok=True)
    
    if not check_environment():
        logger.critical("Cannot start heart_beat due to missing environment variables.")
        return
    
    logger.info("Telos Heartbeat loop starting...")
    notify_startup()
    consecutive_errors = 0
    
    while True:
        logger.info("=== Telos wakes up (with no memory of previous state) ===")
        cycle_start_time = time.time()
        record_heartbeat_status()
        
        # Check system resources
        if not check_system_resources():
            logger.warning("Physical resources too limited. Returning to sleep...")
            time.sleep(system_config.HEARTBEAT_INTERVAL)
            continue
            
        logger.info("--- Heartbeat cycle: Task execution ---")
        
        try:
            # Get the next task
            task = get_task()
            if not task:
                task = {"task": "self-improvement", "details": "Review recent actions and suggest improvements."}
            
            task_name = task.get('task', 'N/A')
            logger.info(f"Selected task: {task}")
            
            # Execute the task
            execution_results = execute_task(task)
            
            # Log performance metrics
            log_performance_metrics(
                cycle_start_time, 
                task_name, 
                execution_results.get("plan_steps", 0), 
                execution_results
            )
            
            # Handle successful execution
            if execution_results.get('success', False):
                # Reset consecutive errors
                consecutive_errors = 0
                
                # Gradually restore heartbeat interval if it was increased
                recover_heartbeat(consecutive_errors)
            else:
                # Log the error but don't increment consecutive_errors
                # since we handled it gracefully
                logger.warning(f"Task execution failed: {execution_results.get('error')}")
                
        except Exception as e:
            consecutive_errors += 1
            log_exception(e, {'cycle_time': time.time() - cycle_start_time})
            notify_error(f"Heartbeat cycle error: {str(e)[:100]}", str(e))
            consecutive_errors = adjust_error_handling(consecutive_errors)
            
        finally:
            logger.info(f"=== Telos goes to sleep for {system_config.HEARTBEAT_INTERVAL} seconds (forgetting this session) ===")
            time.sleep(system_config.HEARTBEAT_INTERVAL)

def close_down(exit_code: int = 0) -> None:
    """
    Handle graceful shutdown.
    
    Args:
        exit_code: Exit code to use (0 for success, non-zero for error)
    """
    logger.info("Telos is shutting down...")
    notify_shutdown()
    
    # Perform any additional cleanup here
    try:
        # Close any open resources, connections, etc.
        pass
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        exit_code = 1
        
    sys.exit(exit_code)

if __name__ == "__main__":
    print("Starting Telos Heart... (press Ctrl+C to stop)")
    try:
        heart_beat()
    except KeyboardInterrupt:
        print("\nTelos Heart stopped by user.")
        close_down()
    except Exception as e:
        # Catch errors during initial startup or catastrophic loop failures
        logging.critical(f"Telos Heart encountered a critical error and stopped: {e}", exc_info=True)
        print(f"Telos Heart encountered a critical error and stopped: {e}")
        close_down(1)  # Exit with an error code 