import os

# --- Core Paths ---
BASE_DIR = os.path.dirname(__file__)
MEMORY_DIR = os.path.join(BASE_DIR, 'memory')
TOOLS_DIR = os.path.join(BASE_DIR, 'tools')

# --- Memory Files ---
TASKS_FILE = os.path.join(MEMORY_DIR, 'tasks.json')
TASK_PROGRESS_FILE = os.path.join(MEMORY_DIR, 'task_progress.json')
ACTION_LOG = os.path.join(MEMORY_DIR, 'action_log.md')
THOUGHTS_LOG = os.path.join(MEMORY_DIR, 'thoughts.md')

# --- Settings ---
HEARTBEAT_INTERVAL: int = 30  # seconds
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
TASK_CHUNK_SIZE = 3  # Maximum number of steps to execute in a single run

# --- LLM ---
# Consider moving API keys here if not using environment variables, but .env is safer.
# OPENAI_API_KEY = "your_key_here"

# Code Generation LLM
CODE_LLM_MODEL = "gpt-4.1" # Or your preferred model for code generation
CODE_LLM_TEMPERATURE = 0.5
CODE_LLM_MAX_TOKENS = 1500
CODE_LLM_SYSTEM_PROMPT = "You are a helpful coding assistant. Generate only the Python code based on the user's request."

# Planner LLM
PLANNER_LLM_MODEL = "gpt-4o" # Can be the same or different model for planning
PLANNER_LLM_TEMPERATURE = 0.7 # Potentially higher temperature for more creative plans
PLANNER_LLM_MAX_TOKENS = 500 # Plans should be relatively concise
PLANNER_SYSTEM_PROMPT = """
You are an expert planner for an autonomous AI agent named Telos.
Your goal is to create a step-by-step plan (a JSON list of strings) to achieve the given task, considering the agent's context.
Output ONLY a valid JSON list of strings, where each string is an actionable step.

Available step formats:
- 'execute_task: <description>' (For general tasks not covered by tools/analysis)
- 'tool_call: tavily_search <query>' (To search online)
- 'script: <relative_path_to_script.py>' (To run a Python script relative to the agent's base directory)
- 'code_generation: <detailed_prompt_for_llm_including_target_file_and_goal>' (To generate code for a specific file/purpose)
- 'apply_code: <target_file_relative_path> <generated_code_file_path>' (To apply generated code from a file to a target file)
- 'analyze_logs: <optional_focus_description>' (To analyze logs and suggest improvement tasks)
- 'update_task_list: <json_list_of_new_tasks_as_string>' (To add new tasks to the queue)
- 'log_thought: <message>' (To record a thought or reflection during execution)

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

Constraints:
- When generating code (`code_generation`), the prompt must be detailed enough for another LLM to write the *complete* target file or a specific function/change.
- The `apply_code` step *replaces the entire content* of the target file with the content from the generated code file. Use with extreme caution.
- Output ONLY the JSON list of strings. Do not include explanations.
Analyze the task and context carefully. Be concise and actionable.
"""

# Analysis LLM (Can reuse planner model initially)
ANALYSIS_LLM_MODEL = PLANNER_LLM_MODEL
ANALYSIS_LLM_TEMPERATURE = 0.6
ANALYSIS_LLM_MAX_TOKENS = 1000 # Allow more space for analysis
ANALYSIS_SYSTEM_PROMPT = """
You are an expert analysis AI assisting an autonomous agent named Telos.
Analyze the provided logs (actions, thoughts) and identify specific, actionable areas for self-improvement.
Focus on recurring errors, inefficiencies, or deviations from the agent's goals.
Based on your analysis, generate a list of new tasks for the agent to perform.
Output ONLY a valid JSON list of task objects (each object should have 'task' and 'details' keys).
Example Output:
[
  {"task": "refactor_memory_manager", "details": "Improve error handling in get_context function based on recent warnings."}, 
  {"task": "optimize_planning_prompt", "details": "Review planner prompt in config.py to provide clearer instructions for code generation steps."}
]
If no specific improvements are identified, return an empty list: []
"""

# --- Execution ---
SCRIPT_TIMEOUT = 60 # seconds for subprocess calls 

# --- Architecture Management ---
ARCHITECTURE_BACKUP_LIMIT = 10  # Maximum number of architecture backups to keep
ARCHITECTURE_TEST_TIMEOUT = 120  # seconds for architecture testing
ARCHITECTURE_LLM_MODEL = CODE_LLM_MODEL  # Use the same model as code generation
ARCHITECTURE_LLM_TEMPERATURE = 0.2  # Lower temperature for more conservative changes
ARCHITECTURE_LLM_MAX_TOKENS = 2000  # Allow more tokens for complex architectural changes
ARCHITECTURE_LLM_SYSTEM_PROMPT = """
You are Telos's architecture manager, responsible for analyzing and improving the system's architecture.
Your task is to generate well-structured, maintainable Python code that enhances Telos's capabilities.
Focus on:
1. Maintaining backward compatibility
2. Following clean architecture principles
3. Reducing coupling between components
4. Improving code organization and readability
5. Adding proper error handling and logging

Output only clean, production-ready Python code without explanations or markdown.
""" 