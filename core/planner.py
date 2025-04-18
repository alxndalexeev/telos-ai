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

Available step formats:
- 'execute_task: <description>' (For general tasks not covered by tools/analysis)
- 'tool_call: tavily_search <query>' (To search online)
- 'script: <relative_path_to_script.py>' (To run a Python script relative to the agent's base directory)
- 'code_generation: <detailed_prompt_for_llm_including_target_file_and_goal>' (To generate code for a specific file/purpose)
- 'apply_code: <target_file_relative_path> <generated_code_file_path>' (To apply generated code from a target file)
- 'analyze_logs: <optional_focus_description>' (To analyze logs and suggest improvement tasks)
- 'update_task_list: <json_list_of_new_tasks_as_string>' (To add new tasks to the queue)
- 'log_thought: <message>' (To record a thought or reflection during execution)

Advanced architecture management steps:
- 'analyze_architecture:' (Analyze the current codebase structure and relationships)
- 'propose_architecture_improvements:' (Generate improvement proposals for the architecture)
- 'implement_architecture_change: <proposal_file> <target_component>' (Implement a proposed architectural change)
- 'test_architecture:' (Test if current architecture is valid and functional)
- 'rollback_architecture: <backup_dir>' (Rollback to a previous architecture version)

Testing framework steps:
- 'run_tests: [unit|integration|all] [parallel] [filter=pattern]' (Run tests with optional parameters)
- 'generate_unit_tests: <module_path>' (Generate unit tests for a specific module)
- 'generate_integration_tests: <module_path1>,<module_path2>' (Generate integration tests for specified modules)
- 'analyze_test_coverage:' (Analyze test coverage across all modules)
- 'generate_test_report:' (Generate report from the most recent test run)
- 'generate_coverage_report:' (Generate a code coverage report)

Example Task: {'task': 'implement', 'details': 'Refactor executor.py to use a class structure'}
Example Plan Output:
[
  "code_generation: Rewrite the entire executor.py file to use a class 'Executor' containing the execution methods. Ensure all current functionality is preserved.",
  "log_thought: Generated refactored code for executor.py. Saved to memory/generated_code/executor_refactor_xyz.py.",
  "apply_code: executor.py memory/generated_code/executor_refactor_xyz.py",
  "log_thought: Applied the refactored code to executor.py."
]

Example Task: {'task': 'self-improvement', 'details': 'Review performance'}
Example Plan Output:
[
  "log_thought: Starting self-improvement cycle: Reviewing performance.",
  "analyze_logs: Identify frequent errors or inefficiencies from recent logs.",
  "log_thought: Analysis complete. Found potential areas for improvement.",
  "update_task_list: [{\"task\": \"investigate_executor_errors\", \"details\": \"Analyze recurring errors in executor logs related to script execution.\"}]"
]

Example Task: {'task': 'architecture', 'details': 'Analyze and improve system architecture'}
Example Plan Output:
[
  "log_thought: Starting architectural analysis to identify improvement opportunities.",
  "analyze_architecture:",
  "log_thought: Architecture analysis complete. Reviewing results for improvement opportunities.",
  "propose_architecture_improvements:",
  "log_thought: Generated architecture improvement proposals.",
  "implement_architecture_change: memory/architecture/proposals/proposals_20230901_120000.json memory_manager.py",
  "test_architecture:",
  "log_thought: Architectural changes implemented and tested successfully."
]

Example Task: {'task': 'testing', 'details': 'Set up tests for memory_manager.py'}
Example Plan Output:
[
  "log_thought: Starting test generation for memory_manager.py.",
  "generate_unit_tests: memory_manager.py",
  "log_thought: Generated unit tests for memory_manager.py.",
  "run_tests: unit filter=memory_manager",
  "log_thought: Executed tests for memory_manager.py. Will now generate coverage report.",
  "analyze_test_coverage:",
  "generate_coverage_report:",
  "log_thought: Completed testing setup for memory_manager.py."
]

Constraints:
- When generating code (`code_generation`), the prompt must be detailed enough for another LLM to write the *complete* target file or a specific function/change.
- The `apply_code` step *replaces the entire content* of the target file with the content from the generated code file. Use with extreme caution.
- Architecture changes should be well-tested before being applied to core components like heart.py.
- Output ONLY the JSON list of strings. Do not include explanations.
Analyze the task and context carefully. Be concise and actionable.
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
            response_format={ "type": "json_object" }, # Request JSON output
        )
        response_content = response.choices[0].message.content
        logger.debug(f"Planner LLM Raw Response: {response_content}")
        
        # Record the API call
        rate_limiter.record_call("openai")

        # --- Parse the Response --- 
        # The response should contain a JSON object, potentially with a key like "plan"
        # Adjust parsing based on observed LLM behavior if it doesn't return just the list.
        plan_data = json.loads(response_content) 

        # Assuming the LLM returns a JSON object like {"plan": [...]}
        # or just the list [...] directly.
        if isinstance(plan_data, list):
            parsed_plan = plan_data
        elif isinstance(plan_data, dict) and 'plan' in plan_data and isinstance(plan_data['plan'], list):
             parsed_plan = plan_data['plan']
        else:
             logger.error(f"LLM planner returned unexpected JSON structure: {response_content}")
             return [f"log_thought: Error - Planner LLM returned invalid plan structure."]

        # Validate plan steps (basic check)
        if not all(isinstance(step, str) for step in parsed_plan):
             logger.error(f"LLM planner returned non-string elements in plan: {parsed_plan}")
             return [f"log_thought: Error - Planner LLM returned invalid plan content."]

        logger.info(f"LLM generated plan: {parsed_plan}")
        return parsed_plan

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON plan from LLM response: {e}")
        logger.error(f"LLM Raw Response was: {response_content}")
        return [f"log_thought: Error - Failed to parse plan from LLM: {e}"]
    except Exception as e:
        logger.error(f"Error calling Planner LLM or processing response: {e}", exc_info=True)
        return _create_fallback_plan(task)

# Add this fallback function
def _create_fallback_plan(task: Task) -> Plan:
    """Create a rule-based fallback plan when LLM planning fails."""
    logger.warning("Using rule-based fallback planning")
    
    task_type = task.get('task', '').lower()
    details = task.get('details', '').lower()
    
    # Default plan for self-improvement
    if task_type == 'self-improvement':
        return [
            "log_thought: Starting self-improvement cycle with rule-based fallback planning due to LLM unavailability.",
            "execute_task: Review memory structure for organization improvements",
            "execute_task: Check for better error handling opportunities"
        ]
    
    # Simple keyword-based planning for other tasks
    elif 'search' in task_type or 'research' in task_type or 'search' in details:
        query = details.split('search for')[-1].strip() if 'search for' in details else details
        return [f"tool_call: tavily_search {query}"]
    
    elif 'code' in task_type or 'implement' in task_type or 'create' in task_type:
        return [f"log_thought: Would generate code for: {details}. Skipping due to LLM unavailability."]
    
    # Generic fallback
    return [f"execute_task: {task.get('task')} - {details}"]

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