"""
Decision-making module.

This module handles the decision-making process after reviewing plans.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

# Import configurations
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from core.planner import create_plan
from core.notification_service import notify_decision

logger = logging.getLogger(__name__)

def make_decision(
    task: Dict[str, Any],
    context: Dict[str, Any],
    max_attempts: int = 5
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Make a decision about what plan to execute after review.
    
    This function handles the loop of planning, reviewing, and deciding
    on a final plan to execute.
    
    Args:
        task: Task information
        context: Context for the task
        max_attempts: Maximum number of review attempts
        
    Returns:
        Tuple of (final_plan, decision_metadata)
    """
    # Import here to avoid circular imports
    from core.review.reviewer import review_plan
    from tools.search import perform_search
    
    # Initialize variables
    reviewer_hint = None
    best_plan = None
    reviewer_agrees = False
    review_comments = None
    search_results = None
    review_attempt = 0
    decision_metadata = {
        "attempts": 0,
        "reviewer_agreed": False,
        "search_performed": False,
        "final_plan_source": "default"
    }
    
    # Perform online search if needed
    task_name = task.get('task', 'N/A')
    try:
        search_results = perform_search(task)
        if search_results:
            logger.info("Online search completed successfully")
            decision_metadata["search_performed"] = True
    except Exception as e:
        logger.warning(f"Online search failed: {e}")
        search_results = None
    
    # Decision/review loop
    while not reviewer_agrees and review_attempt < max_attempts:
        # Create plan with reviewer hint if available
        plan = create_plan(task, context, reviewer_hint=reviewer_hint)
        if not plan:
            logger.warning("Failed to create a plan, using default plan")
            plan = [{"action": "research", "details": f"Research about {task_name}"}]
            decision_metadata["final_plan_source"] = "default"
        
        logger.info(f"Proposed plan: {plan}")
        notify_decision(f"Proposed plan for {task_name}", f"Plan: {plan}")
        
        # Review the plan
        reviewer_agrees, review_comments = review_plan(task, plan, context, search_results)
        decision_metadata["attempts"] += 1
        
        if reviewer_agrees:
            best_plan = plan
            logger.info("Reviewer agrees with the plan")
            decision_metadata["reviewer_agreed"] = True
            decision_metadata["final_plan_source"] = "approved"
        else:
            logger.info(f"Reviewer suggests improvements: {review_comments}")
            reviewer_hint = review_comments
            review_attempt += 1
            
            # If we're on the last attempt, use this plan anyway
            if review_attempt >= max_attempts:
                logger.warning(f"Reached maximum review attempts ({max_attempts}), proceeding with last plan")
                best_plan = plan
                decision_metadata["final_plan_source"] = "last_attempt"
    
    # If no plan was approved, use the last generated plan
    if not best_plan:
        logger.warning("No plan was approved, using the last generated plan")
        best_plan = plan
        decision_metadata["final_plan_source"] = "fallback"
    
    return best_plan, decision_metadata

def execute_decision(
    task: Dict[str, Any],
    plan: List[Dict[str, Any]],
    decision_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute the decision by running the approved plan.
    
    Args:
        task: Task information
        plan: Plan to execute
        decision_metadata: Metadata about the decision-making process
        
    Returns:
        Results from plan execution
    """
    # Import here to avoid circular imports
    from core.executor import execute_plan
    from core.tasks import update_task
    
    task_name = task.get('task', 'N/A')
    
    logger.info(f"Executing decision for task '{task_name}'")
    notify_decision(f"Executing agreed plan for {task_name}", f"Plan: {plan}")
    
    execution_results = {
        "success": False,
        "results": None,
        "error": None,
        "task_status": "attempted"
    }
    
    try:
        # Execute the plan
        results = execute_plan(plan)
        
        logger.info(f"Results from execution: {results}")
        execution_results["success"] = True
        execution_results["results"] = results
        execution_results["task_status"] = "completed"
        
        # Mark task as completed if execution was successful
        if task.get('id'):
            update_task(task.get('id'), status="completed")
    except Exception as e:
        logger.error(f"Error executing plan: {e}")
        error_message = f"Error executing plan: {str(e)}"
        execution_results["error"] = error_message
        execution_results["task_status"] = "failed"
        
        # Mark task as failed
        if task.get('id'):
            update_task(task.get('id'), status="failed", notes=f"Failed with error: {str(e)}")
    
    return execution_results 