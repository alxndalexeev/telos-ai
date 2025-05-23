import os
import sys
import subprocess
import logging
# import openai # Removed direct import
import json # Added for parsing tasks
import ast # Added for code validation
from datetime import datetime, timedelta # Added timedelta for cleanup
import time # Added for cleanup
from typing import List, Any
import re
import shutil # Added for _apply_code_with_git
import tempfile # Added for _apply_code_with_git

# Import from the new module structure
import config
from core.tasks import add_tasks
# from core.api_manager import rate_limiter # Removed, handled by openai_helper
from core.openai_helper import openai_call # Import the helper function
from architecture.manager import analyze_architecture, propose_architectural_improvements
from architecture.manager import implement_architectural_change, test_architecture_integrity, rollback_architectural_change
from tests.framework.test_framework import run_tests, generate_test_for_module, generate_integration_test
from tests.framework.test_framework import analyze_test_coverage, generate_test_report, generate_coverage_report
from telos_logging.logger import log_thought
from core.notification_service import notify_code_generation, notify_code_application

# Define type alias from planner
Plan = List[str]

# --- Setup ---
logger = logging.getLogger(__name__)

# Add tools directory to path - REMOVED (handled by sitecustomize and package structure)
# if config.TOOLS_DIR not in sys.path:
#     sys.path.append(config.TOOLS_DIR)

# Import tools - handle potential import errors
try:
    from tools import tavily_search
    logger.info("Successfully imported tavily_search from tools package.")
except ImportError:
    tavily_search = None
    logger.warning("Could not import tavily_search. Search functionality unavailable.")
except Exception as e:
    logger.error(f"An unexpected error occurred during tool import: {e}")
    tavily_search = None

# Load OpenAI API Key - REMOVED (handled in openai_helper)
# openai.api_key = os.getenv("OPENAI_API_KEY")
# if not openai.api_key:
#     logger.warning("OPENAI_API_KEY environment variable not found. Code generation will fail.")

# --- LLM Code Generation ---
def generate_code_with_llm(prompt: str) -> str:
    """Generates code using the OpenAI API based on the provided prompt."""
    logger.info(f"Sending code generation request to LLM for prompt: {prompt[:100]}...")
    try:
        response = openai_call(
            model=config.CODE_LLM_MODEL,
            messages=[
                {"role": "system", "content": config.CODE_LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=config.CODE_LLM_TEMPERATURE,
            max_tokens=config.CODE_LLM_MAX_TOKENS,
            response_format={"type": "text"},
            trace_name=f"codegen",
            trace_metadata={"prompt": prompt}
        )
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Empty response received from OpenAI API")
        generated_code = response.choices[0].message.content.strip()
        if generated_code.startswith("```") and generated_code.endswith("```"):
            lines = generated_code.split('\n')
            if len(lines) > 2:
                generated_code = '\n'.join(lines[1:-1])
        logger.info(f"LLM generated code snippet: {generated_code[:100]}...")
        return generated_code
    except Exception as e:
        logger.error(f"Error during code generation process: {e}")
        raise

# --- Plan Step Execution Helpers ---

def _execute_task_info(step: str) -> str:
    """Handles the generic 'execute_task:' plan step."""
    task_info = step.replace("execute_task:", "").strip()
    logger.info(f"Executing step: execute_task - {task_info}")
    # Placeholder - In future, could involve more complex logic based on task_info
    return f"Simulated execution of: {task_info}"

def _execute_tavily_search(step: str) -> str:
    """Handles the 'tool_call: tavily_search' plan step."""
    query = step.replace("tool_call: tavily_search", "").strip()
    logger.info(f"Executing step: tavily_search for query: '{query}'")
    
    if tavily_search:
        try:
            # Check if tavily_search has run_search or is already the function itself
            search_func = getattr(tavily_search, 'run_search', tavily_search)
            if callable(search_func):
                search_result = search_func(query)
                # Handle potential error returned in result
                if isinstance(search_result, dict) and "error" in search_result:
                    logger.warning(f"Tavily search returned error: {search_result.get('error')}")
                    return f"Tavily search encountered an issue: {search_result.get('error')}"
                    
                result_str = f"Tavily search successful. Result snippet: {str(search_result)[:200]}..."
                return result_str
            else:
                logger.warning("Tavily search module doesn't have a callable run_search function.")
        except Exception as e:
            logger.error(f"Error running Tavily search for query '{query}': {e}")
            return f"Error during Tavily search: {e}"
    else:
        logger.warning("Tavily search tool is not available or properly configured.")
        return "Tavily search tool unavailable."

def _execute_script(step: str) -> str:
    """Handles the 'script:' plan step."""
    script_path_relative = step.replace("script:", "").strip()
    # Assume script path is relative to the base directory (where heart.py is)
    script_path_absolute = os.path.abspath(os.path.join(config.BASE_DIR, script_path_relative))
    logger.info(f"Executing step: run script '{script_path_absolute}'")

    try:
        if not script_path_absolute.endswith(".py"):
            raise ValueError("Attempting to run non-python script.")
        if not os.path.exists(script_path_absolute):
            raise FileNotFoundError(f"Script not found: {script_path_absolute}")

        process = subprocess.run(
            [sys.executable, script_path_absolute],
            capture_output=True, text=True, check=True, timeout=config.SCRIPT_TIMEOUT
        )

        logger.info(f"Script {script_path_absolute} executed successfully.")
        if process.stdout:
            logger.info(f"Stdout:\n{process.stdout}")
        if process.stderr:
            logger.warning(f"Stderr:\n{process.stderr}")
        return f"Script {script_path_relative} executed successfully. Output snippet: {process.stdout[:100]}..."

    except FileNotFoundError as e:
        logger.error(f"Script execution failed: {e}")
        return f"Script execution failed: {e}"
    except subprocess.CalledProcessError as e:
        logger.error(f"Script {script_path_absolute} failed with exit code {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return f"Script {script_path_relative} failed. Error: {e.stderr[:100]}..."
    except subprocess.TimeoutExpired:
        logger.error(f"Script {script_path_absolute} timed out after {config.SCRIPT_TIMEOUT} seconds.")
        return f"Script {script_path_relative} timed out."
    except Exception as e:
        logger.error(f"Unexpected error running script {script_path_absolute}: {e}")
        return f"Error running script {script_path_relative}: {e}"

# --- Code Generation & Application --- 

GENERATED_CODE_DIR = os.path.join(config.MEMORY_DIR, "generated_code")
os.makedirs(GENERATED_CODE_DIR, exist_ok=True)

def _cleanup_generated_code(max_age_days=30, min_keep=10):
    """Clean up old generated code files to prevent clutter.
    
    Args:
        max_age_days: Maximum age of files to keep in days
        min_keep: Minimum number of most recent files to keep regardless of age
    """
    logger.info(f"Cleaning up generated code files older than {max_age_days} days, keeping at least {min_keep} most recent files")
    
    try:
        # Get all Python files in the generated code directory
        if not os.path.exists(GENERATED_CODE_DIR):
            logger.warning(f"Generated code directory {GENERATED_CODE_DIR} does not exist")
            return
            
        files = [f for f in os.listdir(GENERATED_CODE_DIR) if f.endswith('.py')]
        if not files:
            logger.info("No generated code files found to clean up")
            return
            
        # Sort files by creation time (newest first)
        files_with_time = [(f, os.path.getctime(os.path.join(GENERATED_CODE_DIR, f))) for f in files]
        files_with_time.sort(key=lambda x: x[1], reverse=True)
        
        # Always keep the min_keep most recent files
        files_to_keep = files_with_time[:min_keep]
        files_to_check = files_with_time[min_keep:]
        
        # Set cutoff time for old files
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        # Delete old files
        deleted_count = 0
        for filename, ctime in files_to_check:
            if ctime < cutoff_time:
                file_path = os.path.join(GENERATED_CODE_DIR, filename)
                os.remove(file_path)
                deleted_count += 1
                logger.debug(f"Deleted old generated code file: {filename}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old generated code files")
        else:
            logger.info("No old generated code files to clean up")
            
    except Exception as e:
        logger.error(f"Error cleaning up generated code files: {e}")

def _review_code_with_llm(generated_code, target_file_name=None, prompt=None):
    """Send generated code to LLM for review and improvements.
    
    Args:
        generated_code: The raw code to review
        target_file_name: Optional name of the target file for context
        prompt: Optional original prompt that generated the code
        
    Returns:
        Improved code after LLM review
    """
    logger.info(f"Reviewing generated code with LLM for target: {target_file_name or 'unknown'}")
    
    review_prompt = f"""
Review and improve the following code. Fix any bugs, improve readability, 
add error handling, and ensure best practices are followed.

{f"Target file: {target_file_name}" if target_file_name else ""}
{f"Original prompt: {prompt}" if prompt else ""}

# Code to review:
```python
{generated_code}
```

Return ONLY the improved code without explanations or markdown formatting.
"""
    
    try:
        response = openai_call(
            model=config.CODE_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert code reviewer. Your task is to improve code by fixing bugs, adding error handling, improving readability, and ensuring best practices are followed. Return only the improved code without explanations or markdown."},
                {"role": "user", "content": review_prompt}
            ],
            temperature=config.CODE_REVIEW_TEMPERATURE,
            max_tokens=config.CODE_LLM_MAX_TOKENS,
            response_format={"type": "text"},
            trace_name=f"code-review",
            trace_metadata={
                "target_file": target_file_name,
                "prompt": prompt,
                "code_length": len(generated_code) if generated_code else 0
            }
        )
        
        improved_code = response.choices[0].message.content.strip()
        
        # Remove any markdown code blocks if present
        if improved_code.startswith("```") and improved_code.endswith("```"):
            lines = improved_code.split('\n')
            if len(lines) > 2:
                # Remove first and last lines that contain ```
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                improved_code = '\n'.join(lines)
        
        # Compare code sizes
        original_size = len(generated_code.splitlines())
        improved_size = len(improved_code.splitlines())
        change_percent = ((improved_size - original_size) / original_size) * 100 if original_size > 0 else 0
        
        logger.info(f"Code review complete. Original lines: {original_size}, Improved lines: {improved_size}, Change: {change_percent:.1f}%")
        
        return improved_code
    
    except Exception as e:
        logger.error(f"Error reviewing code with LLM: {e}")
        # Return original code if review fails
        return generated_code

def _execute_code_generation(step: str) -> str:
    """Handles the 'code_generation:' plan step. Saves code to file."""
    prompt = step.replace("code_generation:", "").strip()
    logger.info(f"Executing step: code_generation for prompt: '{prompt[:100]}...'")
    try:
        # Clean up old generated files first
        _cleanup_generated_code(
            max_age_days=config.CODE_CLEANUP_MAX_AGE_DAYS,
            min_keep=config.CODE_CLEANUP_MIN_KEEP
        )
        
        generated_code = generate_code_with_llm(prompt)
        
        # Extract target file name from prompt if possible
        target_file = None
        # Check for common file patterns in the prompt
        file_patterns = [r'\b(\w+\.py)\b', r'create (\w+\.py)', r'implement (\w+\.py)', r'develop (\w+\.py)', r'for (\w+\.py)']
        for pattern in file_patterns:
            matches = re.findall(pattern, prompt.lower())
            if matches:
                target_file = matches[0]
                break
                
        # Review and improve the code with LLM if enabled
        if config.CODE_REVIEW_ENABLED:
            improved_code = _review_code_with_llm(generated_code, target_file, prompt)
        else:
            improved_code = generated_code
            logger.info("Code review is disabled. Using original generated code.")

        # Save generated code to a timestamped file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # If no target file found, use default naming
        if target_file:
            filename = f"{os.path.splitext(target_file)[0]}_{timestamp}.py"
        else:
            filename = f"generated_code_{timestamp}.py"
            
        filepath = os.path.join(GENERATED_CODE_DIR, filename)

        with open(filepath, "w", encoding='utf-8') as f:
            f.write(improved_code)

        # Send notification about the generated code
        purpose_summary = prompt[:50] + "..." if len(prompt) > 50 else prompt
        notify_code_generation(filename, purpose_summary)

        result_str = f"Code generation {'and review ' if config.CODE_REVIEW_ENABLED else ''}successful. Saved to {filepath}. Snippet: {improved_code[:100]}..."
        logger.info(result_str)
        return result_str # Return success message including the path
    except Exception as e:
        # Error already logged in generate_code_with_llm or file op failed
        logger.error(f"Error during code generation or saving: {e}")
        return f"Code generation failed: {e}"

def _apply_code_with_git(target_file_relative, generated_code_file_path, test_cmd="pytest -q"):
    """
    Safer: use temp branch, merge ONLY when tests succeed; auto‑rollback.
    """
    import subprocess, tempfile, os, sys, shutil, datetime
    logger = logging.getLogger(__name__)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = f"ai/{ts}"
    try:
        # Run git commands to set up the branch
        subprocess.run(["git", "checkout", "-B", branch], check=True)
        shutil.copyfile(generated_code_file_path, target_file_relative)
        subprocess.run(["git", "add", target_file_relative], check=True)
        subprocess.run(["git", "commit", "-m", f"Telos AI update {target_file_relative}"], check=True)

        # Only run tests if the test command is specified and not empty
        if test_cmd and test_cmd.strip():
            try:
                res = subprocess.run(test_cmd.split(), capture_output=True, text=True)
                if res.returncode != 0:
                    logger.error("Tests failed:\n%s\n%s", res.stdout, res.stderr)
                    subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
                    subprocess.run(["git", "checkout", "main"], check=True)
                    subprocess.run(["git", "branch", "-D", branch], check=True)
                    return "Tests failed – update aborted."
            except FileNotFoundError as e:
                # Handle missing test command gracefully
                logger.warning(f"Test command '{test_cmd}' not found. Proceeding without tests: {e}")
                # Continue with the merge since we can't run tests
        else:
            logger.info("No test command specified. Proceeding without tests.")
        
        # Complete the merge
        subprocess.run(["git", "checkout", "main"], check=True)
        subprocess.run(["git", "merge", "--ff-only", branch], check=True)
        subprocess.run(["git", "branch", "-D", branch], check=True)
        os.execv(sys.executable, [sys.executable] + sys.argv)  # restart
    except Exception as e:
        logger.exception("self‑update failed")
        return f"Self‑update failed: {e}"

def _apply_code(step: str) -> str:
    """Handles the 'apply_code:' step. Replaces target file content using git workflow."""
    parts = step.replace("apply_code:", "").strip().split()
    
    # Check for command format
    if len(parts) < 2:
        msg = "Invalid apply_code format. Expected: apply_code: <target_file> <generated_code_file> [test_cmd]"
        logger.error(msg)
        return f"Error: {msg}"
        
    # Extract parameters
    if len(parts) >= 3:
        target_file_relative, generated_code_file_path, test_cmd = parts[0], parts[1], " ".join(parts[2:])
    else:
        target_file_relative, generated_code_file_path = parts
        test_cmd = ""  # Empty test command to skip testing
    
    logger.warning(f"Executing step: apply_code - Attempting to overwrite '{target_file_relative}' with content from '{generated_code_file_path}' using git workflow.")
    
    # Check if specified generated code file exists
    full_generated_path = os.path.join(config.BASE_DIR, generated_code_file_path)
    if not os.path.exists(full_generated_path):
        # If the specified file doesn't exist, try to find the latest generated code file
        logger.warning(f"Specified code file {generated_code_file_path} not found. Attempting to find most recent generated code file.")
        generated_dir = os.path.dirname(full_generated_path)
        if not os.path.exists(generated_dir):
            return f"Error: Generated code directory {generated_dir} does not exist."
            
        # Find all Python files in the generated code directory
        generated_files = [f for f in os.listdir(generated_dir) if f.endswith('.py')]
        if not generated_files:
            return f"Error: No generated Python files found in {generated_dir}."
            
        # Get the most recently created file
        latest_file = max(generated_files, key=lambda f: os.path.getctime(os.path.join(generated_dir, f)))
        generated_code_file_path = os.path.join(os.path.relpath(generated_dir, config.BASE_DIR), latest_file)
        logger.info(f"Found most recent generated code file: {generated_code_file_path}")
    
    # Send notification about code application
    notify_code_application(target_file_relative, generated_code_file_path)
    
    return _apply_code_with_git(target_file_relative, generated_code_file_path, test_cmd)

# --- Log Analysis & Task Generation --- 

def _analyze_logs(step: str) -> str:
    """Handles the 'analyze_logs:' plan step."""
    focus = step.replace("analyze_logs:", "").strip()
    logger.info(f"Executing step: analyze_logs. Focus: '{focus or 'General'}'")

    try:
        # Read recent log entries (e.g., last 50 lines of each)
        log_context = "\nRecent Logs:\n"
        log_files = { "Action Log": config.ACTION_LOG, "Thoughts Log": config.THOUGHTS_LOG }
        for name, log_path in log_files.items():
            log_context += f"\n--- {name} ---\n"
            try:
                if os.path.exists(log_path):
                     with open(log_path, 'r', encoding='utf-8') as f:
                         # Simple tail implementation
                         lines = f.readlines()
                         log_context += "".join(lines[-50:]) # Get last 50 lines
                else:
                     log_context += "(Log file not found)\n"
            except Exception as e:
                logger.warning(f"Could not read log file {log_path}: {e}")
                log_context += f"(Error reading log: {e})\n"

        prompt = f"""
Analyze the following recent logs for the AI agent Telos.
Focus: {focus or 'Identify any recurring errors, inefficiencies, or areas for improvement based on the agent goals.'}
{log_context}

Based on your analysis, generate a list of new, specific, actionable tasks for Telos to improve itself.
Tasks MUST be in the format of a list of objects with 'task' and 'details' keys, like this:
[
  {{"task": "task_name", "details": "detailed description of what needs to be done"}}
]

Example:
[
  {{"task": "fix_planner", "details": "Fix the planner module to handle response format issues"}},
  {{"task": "implement_caching", "details": "Implement a caching mechanism for API calls to reduce latency"}}
]

Output ONLY a valid JSON list of task objects (keys: 'task', 'details'). Return [] if no tasks identified.
"""
        logger.debug(f"Analysis LLM Prompt:\n{prompt[:1000]}...") # Log truncated prompt

        # API key check is handled by openai_call
        # if not openai.api_key:
        #     raise ValueError("OpenAI API key is not configured for analysis.")

        # Call Analysis LLM using the helper
        response = openai_call(
            model=config.ANALYSIS_LLM_MODEL,
            messages=[
                {"role": "system", "content": config.ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=config.ANALYSIS_LLM_TEMPERATURE,
            max_tokens=config.ANALYSIS_LLM_MAX_TOKENS,
             response_format={ "type": "json_object" }, # Request JSON output
            trace_name=f"analyze-logs",
            trace_metadata={"step": step, "prompt": prompt}
        )
        response_content = response.choices[0].message.content
        logger.debug(f"Analysis LLM Raw Response: {response_content}")

        # --- Parse the Response --- 
        # Expecting a JSON object containing the list, e.g. {"tasks": [...]} or just [...] 
        tasks_data = json.loads(response_content)
        if isinstance(tasks_data, list):
            suggested_tasks_json = json.dumps(tasks_data) # Return as JSON string for update_task_list step
        elif isinstance(tasks_data, dict):
            if isinstance(tasks_data.get('tasks'), list):
                suggested_tasks_json = json.dumps(tasks_data['tasks'])
            elif all(key in tasks_data for key in ['task', 'details']):
                # Handle single task object
                suggested_tasks_json = json.dumps([tasks_data])
            else:
                logger.error(f"Analysis LLM returned unexpected JSON structure: {response_content}")
                raise ValueError("Analysis LLM returned invalid task structure.")
        else:
            logger.error(f"Analysis LLM returned unexpected JSON structure: {response_content}")
            raise ValueError("Analysis LLM returned invalid task structure.")

        logger.info(f"Log analysis complete. Suggested tasks JSON: {suggested_tasks_json}")
        return f"Log analysis complete. Suggested tasks: {suggested_tasks_json}"

    except Exception as e:
        logger.error(f"Error during log analysis: {e}", exc_info=True)
        return f"Error during log analysis: {e}"

def _update_task_list(step: str) -> str:
    """Handles the 'update_task_list:' plan step."""
    # Remove the prefix and trim
    task_info = step.replace("update_task_list:", "").strip()
    try:
        # Parse task info to get a valid JSON object
        task_data = json.loads(task_info)
        
        # Add task(s) if in proper format
        if isinstance(task_data, list):
            add_tasks(task_data)
            return f"Added {len(task_data)} tasks to the task list"
        elif isinstance(task_data, dict):
            add_tasks([task_data])
            return "Added 1 task to the task list"
        else:
            logger.error(f"Invalid task data format: {type(task_data)}")
            return f"Invalid task data format"
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in task info: {e}")
        return f"Invalid JSON data for tasks: {e}"
    except Exception as e:
        logger.error(f"Failed to add tasks via tasks module: {e}")
        return f"Error adding tasks: {e}"

# --- Misc Execution Helpers ---

def _execute_log_thought(step: str) -> str:
    """Handles the 'log_thought:' plan step."""
    thought = step.replace("log_thought:", "").strip()
    log_thought(f"Plan Step: {thought}")
    return "Thought recorded in logs"

def _execute_unknown(step: str) -> str:
    """Handles unrecognized plan steps."""
    logger.warning(f"Unrecognized plan step format: {step}")
    return f"Unrecognized step: {step}"

# --- Main Execution Function ---
def _execute_plan(plan: Plan) -> List[str]:
    """Execute the plan steps by dispatching to helper functions."""
    results: List[str] = []
    for step in plan:
        result = "Step execution failed unexpectedly."
        try:
            if step.startswith("execute_task:"):
                result = _execute_task_info(step)
            elif step.startswith("tool_call: tavily_search"):
                result = _execute_tavily_search(step)
            elif step.startswith("script:"):
                result = _execute_script(step)
            elif step.startswith("code_generation:"):
                result = _execute_code_generation(step)
            elif step.startswith("apply_code:"):
                result = _apply_code(step)
            elif step.startswith("analyze_logs:"):
                result = _analyze_logs(step)
            elif step.startswith("update_task_list:"):
                 result = _update_task_list(step)
            elif step.startswith("log_thought:"):
                 result = _execute_log_thought(step)
            # Architecture management steps
            elif step.startswith("analyze_architecture:"):
                result = _execute_analyze_architecture(step)
            elif step.startswith("propose_architecture_improvements:"):
                result = _execute_propose_architecture_improvements(step)
            elif step.startswith("implement_architecture_change:"):
                result = _execute_implement_architecture_change(step)
            elif step.startswith("test_architecture:"):
                result = _execute_test_architecture(step)
            elif step.startswith("rollback_architecture:"):
                result = _execute_rollback_architecture(step)
            # Testing framework steps
            elif step.startswith("run_tests:"):
                result = _execute_run_tests(step)
            elif step.startswith("generate_unit_tests:"):
                result = _execute_generate_unit_tests(step)
            elif step.startswith("generate_integration_tests:"):
                result = _execute_generate_integration_tests(step)
            elif step.startswith("analyze_test_coverage:"):
                result = _execute_analyze_test_coverage(step)
            elif step.startswith("generate_test_report:"):
                result = _execute_generate_test_report(step)
            elif step.startswith("generate_coverage_report:"):
                result = _execute_generate_coverage_report(step)
            else:
                result = _execute_unknown(step)
        except Exception as e:
            logger.error(f"Critical error during execution of step '{step}': {e}", exc_info=True)
            result = f"Critical error executing step '{step}': {e}"
            # Decide if plan should stop on critical error
            # break
        results.append(result)
    return results 

# --- Architecture Management Functions ---

def _execute_analyze_architecture(step: str) -> str:
    """Handles the 'analyze_architecture:' plan step."""
    focus = step.replace("analyze_architecture:", "").strip()
    logger.info(f"Executing step: analyze_architecture. Focus: '{focus or 'General'}'")
    
    # Call the architecture_manager function
    return analyze_architecture()

def _execute_propose_architecture_improvements(step: str) -> str:
    """Handles the 'propose_architecture_improvements:' plan step."""
    focus = step.replace("propose_architecture_improvements:", "").strip()
    logger.info(f"Executing step: propose_architecture_improvements. Focus: '{focus or 'General'}'")
    
    # Call the architecture_manager function
    return propose_architectural_improvements()

def _execute_implement_architecture_change(step: str) -> str:
    """Handles the 'implement_architecture_change:' plan step."""
    params = step.replace("implement_architecture_change:", "").strip().split()
    
    if len(params) != 2:
        return "Error: implement_architecture_change requires two parameters: proposal_file and target_component"
    
    proposal_file, target_component = params
    logger.info(f"Executing step: implement_architecture_change. Proposal: '{proposal_file}', Target: '{target_component}'")
    
    # Call the architecture_manager function
    return implement_architectural_change(proposal_file, target_component)

def _execute_test_architecture(step: str) -> str:
    """Handles the 'test_architecture:' plan step."""
    logger.info("Executing step: test_architecture")
    
    # Call the architecture_manager function
    return test_architecture_integrity()

def _execute_rollback_architecture(step: str) -> str:
    """Handles the 'rollback_architecture:' plan step."""
    backup_dir = step.replace("rollback_architecture:", "").strip()
    
    if not backup_dir:
        return "Error: rollback_architecture requires a backup directory parameter"
    
    logger.info(f"Executing step: rollback_architecture. Backup: '{backup_dir}'")
    
    # Call the architecture_manager function
    return rollback_architectural_change(backup_dir)

# --- Testing Framework Functions ---

def _execute_run_tests(step: str) -> str:
    """Handles the 'run_tests:' plan step."""
    params = step.replace("run_tests:", "").strip().split()
    
    # Parse parameters
    test_type = "all"
    parallel = False
    filter_pattern = None
    
    for param in params:
        if param in ["unit", "integration", "all"]:
            test_type = param
        elif param == "parallel":
            parallel = True
        elif "filter=" in param:
            filter_pattern = param.split("=")[1]
    
    logger.info(f"Executing step: run_tests with type={test_type}, parallel={parallel}, filter={filter_pattern}")
    
    # Call the test_framework function
    return run_tests(test_type, parallel, filter_pattern)

def _execute_generate_unit_tests(step: str) -> str:
    """Handles the 'generate_unit_tests:' plan step."""
    module_path = step.replace("generate_unit_tests:", "").strip()
    
    if not module_path:
        return "Error: No module path provided for unit test generation"
    
    logger.info(f"Executing step: generate_unit_tests for module {module_path}")
    
    # Call the test_framework function
    return generate_test_for_module(module_path)

def _execute_generate_integration_tests(step: str) -> str:
    """Handles the 'generate_integration_tests:' plan step."""
    module_paths = step.replace("generate_integration_tests:", "").strip()
    
    if not module_paths:
        return "Error: No module paths provided for integration test generation"
    
    logger.info(f"Executing step: generate_integration_tests for modules {module_paths}")
    
    # Call the test_framework function
    return generate_integration_test(module_paths)

def _execute_analyze_test_coverage(step: str) -> str:
    """Handles the 'analyze_test_coverage:' plan step."""
    logger.info("Executing step: analyze_test_coverage")
    
    # Call the test_framework function
    return analyze_test_coverage()

def _execute_generate_test_report(step: str) -> str:
    """Handles the 'generate_test_report:' plan step."""
    logger.info("Executing step: generate_test_report")
    
    # Call the test_framework function
    return generate_test_report()

def _execute_generate_coverage_report(step: str) -> str:
    """Handles the 'generate_coverage_report:' plan step."""
    logger.info("Executing step: generate_coverage_report")
    
    # Call the test_framework function
    return generate_coverage_report()

# --- Main Execution Function ---
# Rename existing execute_plan to _execute_plan and create a new execute_plan function
# that's compatible with the existing API

def execute_plan(plan: Plan) -> List[str]:
    """Execute the plan steps by dispatching to helper functions."""
    return _execute_plan(plan) 