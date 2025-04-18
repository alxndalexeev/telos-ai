import unittest
import sys
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the modules to test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import planner
import executor
import config
sys.path.pop(0)

class TestPlannerExecutorIntegration(unittest.TestCase):
    """Integration tests for the planner and executor modules."""

    def setUp(self):
        """Set up test fixtures, if any."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.memory_dir = os.path.join(self.test_dir, "memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Save original config paths
        self.original_memory_dir = config.MEMORY_DIR
        self.original_tasks_file = config.TASKS_FILE
        
        # Set config paths for testing
        config.MEMORY_DIR = self.memory_dir
        config.TASKS_FILE = os.path.join(self.memory_dir, "tasks.json")
        
        # Create an empty tasks file
        with open(config.TASKS_FILE, "w") as f:
            json.dump([], f)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        # Restore original config paths
        config.MEMORY_DIR = self.original_memory_dir
        config.TASKS_FILE = self.original_tasks_file
        
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    @patch('openai.chat.completions.create')
    def test_create_plan_and_execute(self, mock_openai):
        """Test creating a plan and executing it."""
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([
            "log_thought: This is a test plan.",
            "execute_task: Run a simple test task"
        ])
        mock_openai.return_value = mock_response
        
        # Define a test task
        test_task = {
            "task": "test",
            "details": "Run a simple test"
        }
        
        # Define test context
        test_context = {
            "who_i_am.md": "I am Telos, an autonomous AI agent."
        }
        
        # Create a plan
        plan = planner.create_plan(test_task, test_context)
        
        # Verify the plan was created correctly
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0], "log_thought: This is a test plan.")
        self.assertEqual(plan[1], "execute_task: Run a simple test task")
        
        # Mock log_thought to avoid side effects
        with patch('executor.log_thought') as mock_log_thought:
            # Execute the plan
            results = executor.execute_plan(plan)
            
            # Verify the results
            self.assertEqual(len(results), 2)
            self.assertIn("log_thought", results[0])
            self.assertIn("Simulated execution", results[1])
            
            # Verify log_thought was called
            mock_log_thought.assert_called_once()

    @patch('openai.chat.completions.create')
    def test_integration_with_code_generation(self, mock_openai):
        """Test integration of planner and executor with code generation."""
        # Set up the mocks for plan creation
        plan_response = MagicMock()
        plan_response.choices = [MagicMock()]
        plan_response.choices[0].message.content = json.dumps([
            "code_generation: Create a simple function to add two numbers.",
            "log_thought: Generated code for simple function."
        ])
        
        # Set up the mocks for code generation
        code_response = MagicMock()
        code_response.choices = [MagicMock()]
        code_response.choices[0].message.content = """
def add_numbers(a, b):
    \"\"\"Add two numbers and return the result.\"\"\"
    return a + b
"""
        
        # Configure the mock to return different responses
        mock_openai.side_effect = [plan_response, code_response]
        
        # Set up test task and context
        test_task = {
            "task": "implement",
            "details": "Create a function to add numbers"
        }
        
        test_context = {
            "my_goal.md": "To create useful functions"
        }
        
        # Create a directory for generated code
        generated_code_dir = os.path.join(self.memory_dir, "generated_code")
        os.makedirs(generated_code_dir, exist_ok=True)
        
        # Patch executor's GENERATED_CODE_DIR
        with patch('executor.GENERATED_CODE_DIR', generated_code_dir):
            # Create a plan
            plan = planner.create_plan(test_task, test_context)
            
            # Verify the plan was created correctly
            self.assertEqual(len(plan), 2)
            
            # Execute the plan (only testing first step)
            with patch('executor.log_action'):
                with patch('executor.log_thought'):
                    # Only execute the code generation step
                    result = executor._execute_code_generation(plan[0])
                    
                    # Verify code generation worked
                    self.assertIn("Code generation successful", result)
                    self.assertIn("def add_numbers", result)
                    
                    # Check that a file was created in the generated code directory
                    files = os.listdir(generated_code_dir)
                    self.assertTrue(any(f.endswith(".py") for f in files))

    def test_execute_log_thought(self):
        """Test the execution of log_thought step."""
        # Patch the log_thought function
        with patch('executor.log_thought') as mock_log_thought:
            # Execute the step
            step = "log_thought: This is a test thought."
            result = executor._execute_log_thought(step)
            
            # Verify the result
            self.assertIn("Logged thought", result)
            
            # Verify log_thought was called with the right argument
            mock_log_thought.assert_called_once_with("Plan Step: This is a test thought.")

    def test_execute_unknown_step(self):
        """Test handling of unknown step types."""
        # Execute an unknown step
        step = "unknown_step: This is an unknown step type."
        result = executor._execute_unknown(step)
        
        # Verify the result indicates an unrecognized step
        self.assertIn("Unrecognized step", result)

if __name__ == '__main__':
    unittest.main() 