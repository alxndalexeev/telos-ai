"""
Central configuration for Telos AI.

Feel free to overwrite values from an external Python file
(see load_override()) or use environment variables.
"""

from pathlib import Path
import os
import json
import importlib.util
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load .env (if present) FIRST so that subsequent imports can rely on env vars
# -----------------------------------------------------------------------------
load_dotenv()

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
MEMORY_DIR: Path = BASE_DIR / "memory"
TOOLS_DIR: Path = BASE_DIR / "tools"

ACTION_LOG   = MEMORY_DIR / "action_log.md"
THOUGHTS_LOG = MEMORY_DIR / "thoughts.md"
TASKS_FILE   = MEMORY_DIR / "tasks.json"
TASK_PROGRESS_FILE = MEMORY_DIR / "task_progress.json"

# -----------------------------------------------------------------------------
# Heart parameters
# -----------------------------------------------------------------------------
HEARTBEAT_INTERVAL         = int(os.getenv("TELOS_HEARTBEAT_INTERVAL", "15"))   # seconds
TASK_CHUNK_SIZE            = int(os.getenv("TELOS_TASK_CHUNK_SIZE", "3"))
SCRIPT_TIMEOUT             = int(os.getenv("TELOS_SCRIPT_TIMEOUT",   "60"))     # seconds

# -----------------------------------------------------------------------------
# LLM models
# -----------------------------------------------------------------------------
PLANNER_LLM_MODEL          = os.getenv("PLANNER_LLM_MODEL",  "gpt-4.1")
CODE_LLM_MODEL             = os.getenv("CODE_LLM_MODEL",     "gpt-4.1")
ANALYSIS_LLM_MODEL         = os.getenv("ANALYSIS_LLM_MODEL", "gpt-4.1-mini")

PLANNER_LLM_TEMPERATURE    = float(os.getenv("PLANNER_LLM_TEMPERATURE",  "0.3"))
CODE_LLM_TEMPERATURE       = float(os.getenv("CODE_LLM_TEMPERATURE",     "0.0"))
ANALYSIS_LLM_TEMPERATURE   = float(os.getenv("ANALYSIS_LLM_TEMPERATURE", "0.2"))

PLANNER_LLM_MAX_TOKENS     = int(os.getenv("PLANNER_LLM_MAX_TOKENS",  "1024"))
CODE_LLM_MAX_TOKENS        = int(os.getenv("CODE_LLM_MAX_TOKENS",     "2048"))
ANALYSIS_LLM_MAX_TOKENS    = int(os.getenv("ANALYSIS_LLM_MAX_TOKENS", "1024"))

# Prompts
CODE_LLM_SYSTEM_PROMPT = "You are Telos' coding co‑pilot. Output ONLY valid Python, no commentary."
ANALYSIS_SYSTEM_PROMPT = "You are Telos' self‑reflection module. Output ONLY JSON."

# -----------------------------------------------------------------------------
# Optional override from config_override.py (git‑ignored)
# -----------------------------------------------------------------------------
def load_override():
    override_path = BASE_DIR / "config_override.py"
    if override_path.exists():
        spec = importlib.util.spec_from_file_location("config_override", override_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        globals().update({k: v for k, v in mod.__dict__.items() if k.isupper()})

load_override()

# --- Settings ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'

# --- LLM ---
# Consider moving API keys here if not using environment variables, but .env is safer.
# OPENAI_API_KEY = "your_key_here"

# Code Generation LLM
CODE_LLM_MODEL = "gpt-4.1" # Latest OpenAI model for code generation
CODE_LLM_TEMPERATURE = 0.0 # Lower temperature for more precise code
CODE_LLM_MAX_TOKENS = 2048
CODE_LLM_SYSTEM_PROMPT = "You are a helpful coding assistant. Generate only the Python code based on the user's request."

# Planner LLM
PLANNER_LLM_MODEL = "gpt-4.1" # Using the latest model for planning
PLANNER_LLM_TEMPERATURE = 0.3 # Balanced temperature for creativity and precision
PLANNER_LLM_MAX_TOKENS = 1024 # Plans should be relatively concise
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

# Analysis LLM (Using a more efficient model for analysis)
ANALYSIS_LLM_MODEL = "gpt-4.1-mini" # Using smaller but capable model for analysis
ANALYSIS_LLM_TEMPERATURE = 0.2
ANALYSIS_LLM_MAX_TOKENS = 1024
ANALYSIS_SYSTEM_PROMPT = """
You are an expert analysis AI assisting an autonomous agent named Telos.
Analyze the provided logs (actions, thoughts) and identify specific, actionable areas for self-improvement.
Focus on recurring errors, inefficiencies, or deviations from the agent's goals.
Based on your analysis, generate a list of new tasks for the agent to perform.
Output ONLY a valid JSON list of task objects (each object should have 'task' and 'details' keys).
Example Output:
[
  {"task": "enhance_error_handling", "details": "Improve error handling in the task execution pipeline based on recent failures."}, 
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

# --- Code Generation & Management --- 
CODE_CLEANUP_MAX_AGE_DAYS = int(os.getenv("CODE_CLEANUP_MAX_AGE_DAYS", "30"))  # Maximum age in days before cleanup
CODE_CLEANUP_MIN_KEEP = int(os.getenv("CODE_CLEANUP_MIN_KEEP", "10"))  # Minimum number of files to keep regardless of age
CODE_REVIEW_ENABLED = os.getenv("CODE_REVIEW_ENABLED", "True").lower() in ("true", "1", "yes")  # Enable/disable LLM code review
CODE_REVIEW_TEMPERATURE = float(os.getenv("CODE_REVIEW_TEMPERATURE", "0.2"))  # Lower temperature for more precise improvements

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

# --- Notification Settings ---
TELEGRAM_NOTIFICATIONS_ENABLED = os.getenv("TELEGRAM_NOTIFICATIONS_ENABLED", "True").lower() in ("true", "1", "yes")
TELEGRAM_API_KEY = os.getenv("TG_API_KEY", "")  # Bot API token from .env
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID", "")  # Chat/group ID from .env
TELEGRAM_NOTIFICATION_LEVEL = os.getenv("TELEGRAM_NOTIFICATION_LEVEL", "important")  # Options: all, important, minimal 

# --- Langfuse Observability ---
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "") 

# --- Pinecone Vector DB ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "telos-memory")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")  # Optional, for legacy support
# PINECONE_REGION = os.getenv("PINECONE_REGION", "us-central1")  # Optional

# --- Embedding Model ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))

# --- Available Resources ---
AVAILABLE_RESOURCES = {
    "cpu_cores": int(os.getenv("AVAILABLE_CPU_CORES", "4")),  # Number of CPU cores available for processing
    "memory_gb": float(os.getenv("AVAILABLE_MEMORY_GB", "16.0")),  # RAM available in GB
    "gpu_memory_gb": float(os.getenv("AVAILABLE_GPU_MEMORY_GB", "0.0")),  # GPU memory available in GB
    "disk_space_gb": float(os.getenv("AVAILABLE_DISK_SPACE_GB", "100.0")),  # Disk space available in GB
    "max_concurrent_tasks": int(os.getenv("MAX_CONCURRENT_TASKS", "3")),  # Maximum number of tasks to run concurrently
    "max_daily_api_cost": float(os.getenv("MAX_DAILY_API_COST", "10.0")),  # Maximum daily cost for API calls in USD
} 