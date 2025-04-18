import logging
import json
import os
import openai
from typing import List, Dict, Any

# Import from the new module structure
import config
from core.api_manager import rate_limiter

# Define type aliases used from memory_manager
Context = Dict[str, str]
Task = Dict[str, Any]
Plan = List[str]

logger = logging.getLogger(__name__)

# Load OpenAI API Key (needed for planning)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.warning("OPENAI_API_KEY environment variable not found. Planning LLM will fail.")

# Store the system prompt here to make it easily replaceable
# This may be moved to config.py in future
SYSTEM_PROMPT = """
You are an expert planner for an autonomous AI agent named Telos.
Your goal is to create a step-by-step plan (a JSON list of strings) to achieve the given task, considering the agent's context.
Output ONLY a valid JSON list of strings, where each string is an actionable step.

Available step formats (MUST start with one of these prefixes):
- 'log_thought: <message>' (To record a thought or reflection)
- 'execute_task: <description>' (For general tasks)
- 'analyze_logs: <optional_focus>' (To analyze logs)
- 'update_task_list: <json_list_of_tasks>' (To add new tasks)
- 'tool_call: <tool_name> <query>' (To use external tools)
- 'code_generation: <prompt>' (To generate code)
- 'apply_code: <target_file> <generated_code_file>' (To apply code)
- 'generate_unit_tests: <module_path>' (To generate tests)
- 'run_tests: <type> <options>' (To run tests)
- 'analyze_test_coverage:' (To analyze coverage)
- 'generate_coverage_report:' (To generate report)

IMPORTANT: When using 'update_task_list:', the task list MUST be a JSON array of task objects with 'task' and 'details' keys.
Example for update_task_list:
"update_task_list: [{\"task\": \"implement_cache\", \"details\": \"Implement a memory cache for API calls\"}, {\"task\": \"add_metrics\", \"details\": \"Add better instrumentation with metrics tracking\"}]"

Example Task: {'task': 'implement', 'details': 'Add metrics tracking'}
Example Plan Output:
[
  "log_thought: Starting implementation of metrics tracking system",
  "code_generation: Create a metrics tracking module with functions for recording and analyzing performance metrics",
  "apply_code: metrics_tracker.py memory/generated_code/metrics_tracker.py",
  "generate_unit_tests: metrics_tracker.py",
  "run_tests: unit filter=metrics_tracker",
  "log_thought: Metrics tracking system implementation complete"
]

Constraints:
- Each step MUST start with one of the defined prefixes
- Steps must be in a logical sequence
- Code generation steps should include detailed prompts
- Output ONLY the JSON list of strings
- Do not include explanations or comments
"""

def create_plan(task: Task, context: Context) -> Plan:
    """Create a plan using an LLM based on the task and context."""
    logger.info(f"Creating LLM-powered plan for task: {task.get('task')}")

    if not openai.api_key:
        logger.error("Cannot create LLM plan: OpenAI API key is not configured.")
        return _create_fallback_plan(task)
        
    # Check API rate limit before making the call
    if not rate_limiter.can_make_call("openai"):
        logger.warning("API rate limit reached. Using fallback plan")
        return _create_fallback_plan(task)

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

Generate a plan as a JSON list of strings according to the available step formats.
Output ONLY the JSON list.
Plan:
"""

    logger.debug(f"Planner LLM Prompt:\n{prompt}")

    # --- Call the LLM --- 
    try:
        response = openai.chat.completions.create(
            model=config.PLANNER_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=config.PLANNER_LLM_TEMPERATURE,
            max_tokens=config.PLANNER_LLM_MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        response_content = response.choices[0].message.content.strip()
        logger.debug(f"Planner LLM Raw Response: {response_content}")
        
        # Record the API call
        rate_limiter.record_call("openai")

        # --- Parse the Response --- 
        if not response_content:
            logger.error("LLM returned empty response")
            return _create_fallback_plan(task)

        try:
            plan_data = json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
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
            # If it's a dictionary with a 'plan' key
            if isinstance(plan_data.get('plan'), list):
                parsed_plan = plan_data['plan']
            # If it's a dictionary of key-value pairs
            else:
                parsed_plan = []
                for key, value in plan_data.items():
                    if isinstance(key, str) and isinstance(value, str):
                        parsed_plan.append(key)
                        parsed_plan.append(value)
                    else:
                        logger.warning(f"Skipping invalid plan step: {key} -> {value}")
        else:
            logger.error(f"Unexpected response type from LLM: {type(plan_data)}")
            return _create_fallback_plan(task)

        # Validate plan steps
        if not parsed_plan:
            logger.error("LLM returned empty plan")
            return _create_fallback_plan(task)

        # Ensure all steps are strings and have the correct format
        valid_steps = []
        for step in parsed_plan:
            if isinstance(step, str):
                # Check if the step matches any of our expected formats
                if any(step.startswith(prefix) for prefix in [
                    "log_thought:", "execute_task:", "analyze_logs:", 
                    "update_task_list:", "tool_call:", "code_generation:",
                    "apply_code:", "generate_unit_tests:", "run_tests:",
                    "analyze_test_coverage:", "generate_coverage_report:"
                ]):
                    valid_steps.append(step)
                else:
                    logger.warning(f"Skipping invalid step format: {step}")
            else:
                logger.warning(f"Skipping non-string step: {step}")

        if not valid_steps:
            logger.error("No valid steps found in plan")
            return _create_fallback_plan(task)

        return valid_steps

    except Exception as e:
        logger.error(f"Error creating plan: {e}", exc_info=True)
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