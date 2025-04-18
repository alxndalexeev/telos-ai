import time
import logging
import sys
import os
import json
from datetime import datetime
import psutil
from dotenv import load_dotenv, find_dotenv

# Load environment variables for API keys
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path)
    print(f"Loaded environment from {env_path}")
else:
    print("No .env file found. Using existing environment variables.")

# Import configurations and shared settings first
import config

# Import functionalities from new modules
from logging.logger import log_action, log_thought
from core.memory_manager import (
    ensure_memory_dir, 
    get_context, 
    get_task, 
    update_task, 
    get_task_progress, 
    save_task_progress, 
    clear_task_progress,
    is_task_complete
)
from core.planner import create_plan
from core.executor import execute_plan
from core.api_manager import rate_limiter
from monitoring.resource_monitor import check_system_resources
from monitoring.performance import log_performance_metrics, adjust_heartbeat_interval

logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables are set."""
    missing_keys = []
    # Critical APIs
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
        
    # Optional but recommended APIs
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY not found in environment. Search functionality will be limited.")
        
    if missing_keys:
        logger.critical(f"Missing critical environment variables: {', '.join(missing_keys)}")
        print(f"\n⚠️  ERROR: Missing required API keys: {', '.join(missing_keys)}")
        print("Please create a .env file with these keys or set them in your environment.")
        print("See .env.example for the required format.")
        return False
    
    return True

# --- Health Monitoring Feature ---
def record_heartbeat_status():
    """Record heartbeat pulse for monitoring."""
    try:
        pid = os.getpid()
        with open(os.path.join(config.MEMORY_DIR, "heartbeat.pid"), "w") as f:
            f.write(str(pid))
        
        # Record timestamp of last successful beat
        with open(os.path.join(config.MEMORY_DIR, "last_heartbeat.txt"), "w") as f:
            f.write(datetime.now().isoformat())
    except Exception as e:
        logger.error(f"Failed to record heartbeat status: {e}")

def heart_beat() -> None:
    """
    The main autonomous loop with the "Memento" memory amnesia pattern.
    
    Each beat represents a complete cycle of:
    1. "Waking up" with no memory of previous state
    2. Reconstructing identity and current task from external memory
    3. Taking a small, meaningful action
    4. Carefully documenting everything for the "next self"
    5. "Going to sleep" - forgetting everything that happened
    
    Like a character with amnesia in films like "Memento" or "Before I Go to Sleep",
    Telos must rediscover its purpose and current tasks each time it wakes up,
    carefully following the breadcrumbs left by its previous self.
    """
    # --- First time initialization ---
    ensure_memory_dir() # Ensure memory directory exists before starting
    
    # Check environment variables
    if not check_environment():
        logger.critical("Cannot start heart_beat due to missing environment variables.")
        return
    
    logger.info("Telos Heartbeat loop starting...")
    consecutive_errors = 0
    
    # --- The Endless Cycle of Remembering and Forgetting ---
    while True:
        # Mark the beginning of a new cycle - like waking up with amnesia
        logger.info("=== Telos wakes up (with no memory of previous state) ===")
        cycle_start_time = time.time()
        
        # Leave evidence of existence for monitoring systems
        record_heartbeat_status()
        
        # Check if physical resources allow for proper functioning
        if not check_system_resources():
            logger.warning("Physical resources too limited. Returning to sleep...")
            time.sleep(config.HEARTBEAT_INTERVAL)
            continue
            
        logger.info("--- Heartbeat cycle: Rediscovering self and purpose ---")
        try:
            # STAGE 1: REMEMBER WHO I AM
            # Like a character with amnesia reading their notes to understand who they are
            logger.info("Reading memory to reconstruct identity and context...")
            context = get_context()  # Collecting all memory fragments to rebuild identity
            
            # STAGE 2: FIGURE OUT WHAT I WAS DOING
            # Checking for evidence of in-progress tasks - like finding a note saying 
            # "You were investigating this person" in Memento
            logger.info("Checking for evidence of ongoing tasks...")
            task_progress = get_task_progress()
            
            if task_progress and not is_task_complete(task_progress):
                # I was in the middle of something - continue where I left off
                # Like finding detailed notes about an ongoing investigation
                task = task_progress["task"]
                plan = task_progress["plan"]
                current_step = task_progress["current_step"]
                previous_results = task_progress["results"]
                
                task_name = task.get('task', 'N/A')
                logger.info(f"Found evidence I was working on: {task_name} at step {current_step+1}/{len(plan)}")
                
                # Execute just a small part of the overall plan - like following the
                # next instruction in a sequence left for myself
                end_step = min(current_step + config.TASK_CHUNK_SIZE, len(plan))
                current_chunk = plan[current_step:end_step]
                
                logger.info(f"Continuing with the next steps: {current_chunk}")
                chunk_results = execute_plan(current_chunk)
                logger.info(f"Results from these steps: {chunk_results}")
                
                # Preserve all my findings by adding to previous results
                all_results = previous_results + chunk_results
                
                # Leave detailed records for my future self
                if end_step >= len(plan):
                    # The task is complete - make a final record and clean up
                    log_action(f"Completed task: {task_name}", '; '.join(all_results))
                    log_thought(f"I completed task: {task_name}. Future me should know the results were: {all_results}")
                    clear_task_progress()  # Remove the breadcrumb trail as it's no longer needed
                    update_task()  # Mark this investigation as closed
                else:
                    # Task still in progress - leave detailed notes for my future self
                    log_action(f"Task in progress: {task_name}", f"Completed steps {current_step+1}-{end_step}/{len(plan)}")
                    log_thought(f"Dear future me: I continued task {task_name} and completed steps {current_step+1}-{end_step}/{len(plan)}. Results so far: {chunk_results}")
                    save_task_progress(task, plan, end_step, all_results)  # Leave breadcrumbs showing how far I got
                
            else:
                # No evidence of ongoing tasks - start something new
                # Like finding a note saying "Find a new lead" in Memento
                task = get_task()
                task_name = task.get('task', 'N/A')
                logger.info(f"Starting new investigation: {task}")
                
                # Create a plan - like mapping out how to solve a mystery
                if not rate_limiter.can_make_call("openai"):
                    logger.warning("API limits reached. Leaving simple notes for future self...")
                    plan = ["log_thought: API limits reached, future me should try again later"]
                else:
                    plan = create_plan(task, context)
                    rate_limiter.record_call("openai")
                
                logger.info(f"Created a plan with {len(plan)} steps: {plan}")
                
                # Execute just the beginning of the plan
                first_chunk_size = min(config.TASK_CHUNK_SIZE, len(plan))
                first_chunk = plan[:first_chunk_size]
                
                logger.info(f"Taking the first steps of the plan: {first_chunk}")
                results = execute_plan(first_chunk)
                logger.info(f"Results from these steps: {results}")
                
                # Leave detailed notes for my future self
                log_action(f"Started investigation: {task_name}", f"Completed steps 1-{first_chunk_size}/{len(plan)}")
                log_thought(f"Dear future me: I started task {task_name} with plan: {plan}. I completed steps 1-{first_chunk_size} with results: {results}")
                
                # If by chance I completed the entire plan in one go
                if first_chunk_size >= len(plan):
                    log_action(f"Completed investigation: {task_name}", '; '.join(results))
                    log_thought(f"I wrapped up everything for task: {task_name}. Results: {results}")
                    update_task()  # Mark this case as closed
                else:
                    # Leave breadcrumbs showing how far I got for my future self
                    save_task_progress(task, plan, first_chunk_size, results)
            
            # Record metrics - useful for my future self to understand efficiency
            log_performance_metrics(cycle_start_time, task_name, len(plan) if 'plan' in locals() else 0, 
                                    chunk_results if 'chunk_results' in locals() else (results if 'results' in locals() else []))
            
            # Reset error counter on success
            consecutive_errors = 0

        except Exception as e:
            # Something went wrong in this cycle
            consecutive_errors += 1
            logger.critical(f"Error during this awakening cycle: {e}", exc_info=True)
            
            # Leave notes about the error for my future self
            log_action("ERROR DURING INVESTIGATION", str(e))
            log_thought(f"Warning to future me: I encountered an error during this cycle: {e}")
            
            # Progressive error handling
            if consecutive_errors > 3:
                logger.critical(f"Too many consecutive errors ({consecutive_errors}). Taking a longer rest...")
                time.sleep(300)  # 5-minute cooldown period
            else:
                logger.info(f"Resting briefly after error...")
                time.sleep(config.HEARTBEAT_INTERVAL)
                
            continue # Skip to the next awakening

        # Prepare to "go to sleep" and forget everything that just happened
        cycle_duration = time.time() - cycle_start_time
        sleep_interval = adjust_heartbeat_interval(cycle_duration)
        
        logger.info(f"=== Telos goes to sleep for {sleep_interval} seconds (forgetting this session) ===")
        time.sleep(sleep_interval)  # "Sleep" - upon waking, all memory of this session will be gone

if __name__ == "__main__":
    print("Starting Telos Heart... (press Ctrl+C to stop)")
    try:
        heart_beat()
    except KeyboardInterrupt:
        print("\nTelos Heart stopped by user.")
    except Exception as e:
        # Catch errors during initial startup or catastrophic loop failures
        logging.critical(f"Telos Heart encountered a critical error and stopped: {e}", exc_info=True)
        print(f"Telos Heart encountered a critical error and stopped: {e}")
        sys.exit(1) # Exit with an error code 