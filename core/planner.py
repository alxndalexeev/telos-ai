import logging
import json
import os
# import openai # Removed direct import
from typing import List, Dict, Any

# Import from the new module structure
import config
# from core.api_manager import rate_limiter # Removed direct import
from core.openai_helper import openai_call # Import the helper

# Define type aliases for clarity
Context = Dict[str, str]
Task = Dict[str, Any]
Plan = List[str]

logger = logging.getLogger(__name__)

# Load OpenAI API Key - REMOVED (handled in openai_helper)
# openai.api_key = os.getenv("OPENAI_API_KEY")
# if not openai.api_key:
#     logger.warning("OPENAI_API_KEY environment variable not found. Planning LLM will fail.")

# Store the system prompt here to make it easily replaceable
# This may be moved to config.py in future
SYSTEM_PROMPT = config.PLANNER_SYSTEM_PROMPT # Use prompt from config

def create_plan(task: Task, context: Context, reviewer_hint: str = None) -> Plan:
    """Create a plan using an LLM based on the task, context, and optional reviewer hint."""
    logger.info(f"Creating LLM-powered plan for task: {task.get('task')}")

    # API Key check is handled by openai_call
    # if not openai.api_key:
    #     logger.error("Cannot create LLM plan: OpenAI API key is not configured.")
    #     return _create_fallback_plan(task)
        
    # Rate limit check is handled by openai_call
    # if not rate_limiter.can_make_call("openai"):
    #     logger.warning("API rate limit reached. Using fallback plan")
    #     return _create_fallback_plan(task)

    # Define the valid step prefixes at the beginning of the function
    valid_prefixes = [
        "log_thought:", "execute_task:", "analyze_logs:", 
        "update_task_list:", "tool_call:", "code_generation:",
        "apply_code:", "generate_unit_tests:", "run_tests:",
        "analyze_test_coverage:", "generate_coverage_report:",
        # Architecture prefixes
        "analyze_architecture:", "propose_architecture_improvements:",
        "implement_architecture_change:", "test_architecture:",
        "rollback_architecture:"
    ]

    # --- Construct the Prompt --- 
    # Select relevant context to provide to the LLM (avoiding excessive length)
    # Prioritize core identity, goals, and open questions.
    context_summary = "\nRelevant Context Files:\n"
    core_files = ['who_i_am.md', 'my_goal.md', 'open_questions_todo.md']
    for filename, content in context.items():
        if filename in core_files:
             # Include full content for core files (if reasonably short)
             context_summary += f"- {filename}:\n{content[:500]}...\n"
        else:
             # Include only filenames for other context
             context_summary += f"- {filename}\n"

    prompt = f"""
Given the following task:
Task Type: {task.get('task')}
Task Details: {task.get('details')}

And the following context about the agent:
{context_summary}
"""
    if reviewer_hint:
        prompt += f"\nReviewer feedback to consider when planning: {reviewer_hint}\n"
    prompt += """
Generate a plan as a JSON list of strings according to the available step formats.
Output ONLY the JSON list.
Plan:
"""

    logger.debug(f"Planner LLM Prompt:\n{prompt}")

    # --- Call the LLM --- 
    try:
        # Use the helper function
        response = openai_call(
            model=config.PLANNER_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=config.PLANNER_LLM_TEMPERATURE,
            max_tokens=config.PLANNER_LLM_MAX_TOKENS,
            response_format={"type": "json_object"},
            trace_name=f"planner-create-plan",
            trace_metadata={
                "task": task,
                "reviewer_hint": reviewer_hint,
                "context_keys": list(context.keys())
            }
        )
        message = response.choices[0].message
        if not message or not getattr(message, "content", None):
            logger.error("LLM returned no content in response")
            return _create_fallback_plan(task)
        response_content = message.content.strip()
        logger.debug(f"Planner LLM Raw Response: {response_content}")
        
        # Record the API call - REMOVED (handled in openai_helper)
        # rate_limiter.record_call("openai")

        # --- Parse the Response --- 
        if not response_content:
            logger.error("LLM returned empty response")
            return _create_fallback_plan(task)

        try:
            plan_data = json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            # Attempt to clean the response if it looks like markdown
            if response_content.startswith("```json") and response_content.endswith("```"):
                logger.info("Attempting to clean markdown JSON response...")
                cleaned_content = '\n'.join(response_content.split('\n')[1:-1])
                try:
                    plan_data = json.loads(cleaned_content)
                    logger.info("Successfully parsed cleaned JSON.")
                except json.JSONDecodeError as e2:
                    logger.error(f"Failed to parse cleaned JSON: {e2}")
                    return _create_fallback_plan(task)
            else:
                 return _create_fallback_plan(task)

        # Handle different response formats
        if isinstance(plan_data, list):
            # If it's a list of strings, use it directly
            if all(isinstance(step, str) for step in plan_data):
                parsed_plan = plan_data
            # If it's a list of dictionaries, extract steps
            elif all(isinstance(step, dict) for step in plan_data):
                parsed_plan = []
                for step in plan_data:
                    if isinstance(step.get('step'), str):
                        parsed_plan.append(step['step'])
            else:
                logger.error(f"Invalid list format in LLM response: {plan_data}")
                return _create_fallback_plan(task)
        elif isinstance(plan_data, dict):
            # If it's a dictionary with a 'plan' key (common pattern)
            plan_key_found = None
            for key in ["plan", "steps", "plan_steps"]:
                 if key in plan_data and isinstance(plan_data[key], list):
                     plan_key_found = key
                     break
            
            if plan_key_found:
                parsed_plan = plan_data[plan_key_found]
            # Special case: handle dictionaries where keys are steps
            elif all(isinstance(k, str) for k in plan_data.keys()):
                # Check if dict keys look like steps 
                step_keys = [k for k in plan_data.keys() if any(k.startswith(prefix) for prefix in valid_prefixes)]
                if step_keys:
                    logger.info(f"LLM returned dictionary with {len(step_keys)} keys that appear to be plan steps")
                    parsed_plan = step_keys
                # Check if the dictionary looks like a task itself
                elif "task" in plan_data and isinstance(plan_data["task"], str):
                    task_name = plan_data["task"]
                    task_details = plan_data.get("details", "")
                    logger.info(f"LLM returned a task definition instead of a plan. Creating plan for task: {task_name}")
                    parsed_plan = [
                        f"log_thought: Processing task '{task_name}'",
                        f"execute_task: {task_name} - {task_details}",
                        "analyze_logs: Review execution results"
                    ]
                # Also check if there's no explicit task key but we have details that look like a task
                elif "details" in plan_data and isinstance(plan_data["details"], str) and not any(k in plan_data for k in ["plan", "steps"]):
                    task_details = plan_data["details"]
                    logger.info(f"LLM returned details without explicit plan. Creating plan based on details")
                    parsed_plan = [
                        "log_thought: Processing task based on provided details",
                        f"execute_task: {task.get('task')} - {task_details}",
                        "analyze_logs: Review execution results"
                    ]
                else:
                    logger.warning(f"LLM returned a dictionary without valid plan steps in keys: {list(plan_data.keys())[:3]}...")
                    return _create_fallback_plan(task)
            # If it's a dictionary of key-value pairs (less likely for plan)
            else:
                 logger.warning(f"LLM returned a dictionary but no obvious plan list key found: {list(plan_data.keys())[:3]}...")
                 return _create_fallback_plan(task)
        else:
            logger.error(f"Unexpected response type from LLM: {type(plan_data)}")
            return _create_fallback_plan(task)

        # Validate plan steps
        if not parsed_plan:
            logger.error("LLM returned empty or unparseable plan")
            return _create_fallback_plan(task)

        # Ensure all steps are strings and have the correct format
        valid_steps = []
        for step in parsed_plan:
            if isinstance(step, str) and any(step.startswith(prefix) for prefix in valid_prefixes):
                 valid_steps.append(step)
            else:
                logger.warning(f"Skipping invalid step format or non-string step: {step}")

        if not valid_steps:
            logger.error("No valid steps found in plan after validation")
            return _create_fallback_plan(task)

        return valid_steps

    # Catch errors from openai_call or JSON parsing
    except (ValueError, RuntimeError, json.JSONDecodeError) as e:
        logger.error(f"Error creating plan ({type(e).__name__}): {e}", exc_info=True)
        return _create_fallback_plan(task)
    except Exception as e:
        logger.error(f"Unexpected error creating plan: {e}", exc_info=True)
        return _create_fallback_plan(task)

def _create_fallback_plan(task: Task) -> Plan:
    """Create a simple fallback plan when LLM planning fails."""
    logger.info("Creating fallback plan")
    return [
        "log_thought: Using fallback plan due to planning error.",
        f"execute_task: {task.get('task')} - {task.get('details')}",
        "analyze_logs: Identify any errors or issues from this execution."
    ]

# --- Keep the old rule-based logic commented out for reference/fallback --- 
# def create_plan_rule_based(task: Task, context: Context) -> Plan:
#     """Create a plan based on the task and context."""
#     logger.info(f"Creating plan for task: {task.get('task')}")
#     # Future: Use context (identity, goals, experience logs) for more sophisticated planning.
#     # Example: logger.debug(f"Planning context keys: {context.keys()}")
#     # Future: Add specific planning logic for 'self-improvement' task if needed.
#     # e.g., if task.get('task') == 'self-improvement':
#     #    return ['reflect_on_recent_actions', 'identify_improvement_areas']
# 
#     # The 'idle' task is no longer used, default is 'self-improvement' handled below.
# 
#     details = task.get('details', '').lower()
# 
#     # Simple keyword-based planning
#     if 'search online for' in details or 'research' in details:
#         query = details.split('search online for')[-1].split('research')[-1].strip()
#         if query:
#             logger.info(f"Planning step: Search online for '{query}'")
#             return [f"tool_call: tavily_search {query}"]
#         else:
#             logger.warning("Search task found, but no query specified in details.")
#     elif 'run script' in details:
#          script_path = details.split('run script')[-1].strip()
#          if script_path:
#              logger.info(f"Planning step: Run script '{script_path}'")
#              return [f"script: {script_path}"]
#          else:
#             logger.warning("Run script task found, but no script path specified.")
#     elif 'write code for' in details or 'implement' in details:
#          prompt = details.split('write code for')[-1].split('implement')[-1].strip()
#          if prompt:
#              logger.info(f"Planning step: Generate code for '{prompt[:50]}...'")
#              return [f"code_generation: {prompt}"]
#          else:
#              logger.warning("Code generation task found, but no prompt specified.")
# 
#     # Default plan if no specific keywords match or details are missing
#     logger.info("Using default task execution plan.")
#     return [f"execute_task: {task.get('task', 'Unknown')} - {task.get('details', 'No details')}"] 