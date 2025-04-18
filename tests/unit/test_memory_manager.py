import unittest
import sys
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the module to test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import memory_manager
sys.path.pop(0)

class TestMemoryManager(unittest.TestCase):
    """Unit tests for the memory_manager module."""

    def setUp(self):
        """Set up test fixtures, if any."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Mock the config module
        self.config_patcher = patch('memory_manager.config')
        self.mock_config = self.config_patcher.start()
        
        # Set up config paths for testing
        self.mock_config.MEMORY_DIR = self.test_dir
        self.mock_config.TASKS_FILE = os.path.join(self.test_dir, 'tasks.json')
        self.mock_config.ACTION_LOG = os.path.join(self.test_dir, 'action_log.md')
        self.mock_config.THOUGHTS_LOG = os.path.join(self.test_dir, 'thoughts.md')
        
        # Ensure the test directory is empty
        for f in [self.mock_config.TASKS_FILE, self.mock_config.ACTION_LOG, 
                  self.mock_config.THOUGHTS_LOG]:
            if os.path.exists(f):
                os.remove(f)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        # Stop the mock
        self.config_patcher.stop()
        
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_ensure_memory_dir(self):
        """Test ensure_memory_dir function."""
        # Call the function
        memory_manager.ensure_memory_dir()
        
        # Check that the directory was created
        self.assertTrue(os.path.exists(self.test_dir))
        
        # Check that the log files were created
        self.assertTrue(os.path.exists(self.mock_config.ACTION_LOG))
        self.assertTrue(os.path.exists(self.mock_config.THOUGHTS_LOG))
        
        # Verify log file contents
        with open(self.mock_config.ACTION_LOG, 'r') as f:
            content = f.read()
            self.assertIn("Action Log", content)
        
        with open(self.mock_config.THOUGHTS_LOG, 'r') as f:
            content = f.read()
            self.assertIn("Thoughts Log", content)

    def test_get_context_empty(self):
        """Test get_context with empty directory."""
        # Ensure directory exists but is empty
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Call the function
        context = memory_manager.get_context()
        
        # Verify the result is an empty dict
        self.assertEqual(context, {})

    def test_get_context_with_files(self):
        """Test get_context with files."""
        # Create test markdown files
        os.makedirs(self.test_dir, exist_ok=True)
        
        test_files = {
            'test1.md': 'Test content 1',
            'test2.md': 'Test content 2',
            'not_included.txt': 'Should not be included'
        }
        
        for filename, content in test_files.items():
            with open(os.path.join(self.test_dir, filename), 'w') as f:
                f.write(content)
        
        # Call the function
        context = memory_manager.get_context()
        
        # Verify the result
        self.assertEqual(len(context), 2)
        self.assertEqual(context['test1.md'], 'Test content 1')
        self.assertEqual(context['test2.md'], 'Test content 2')
        self.assertNotIn('not_included.txt', context)

    def test_get_task_no_file(self):
        """Test get_task when tasks file doesn't exist."""
        # Ensure the tasks file doesn't exist
        if os.path.exists(self.mock_config.TASKS_FILE):
            os.remove(self.mock_config.TASKS_FILE)
        
        # Call the function
        task = memory_manager.get_task()
        
        # Verify it returns the default task
        self.assertEqual(task['task'], 'self-improvement')
        self.assertIn('No specific tasks', task['details'])

    def test_get_task_with_tasks(self):
        """Test get_task with a valid tasks file."""
        # Create a test tasks file
        os.makedirs(os.path.dirname(self.mock_config.TASKS_FILE), exist_ok=True)
        
        test_tasks = [
            {
                'task': 'test_task',
                'details': 'Test task details'
            },
            {
                'task': 'another_task',
                'details': 'Another task details'
            }
        ]
        
        with open(self.mock_config.TASKS_FILE, 'w') as f:
            json.dump(test_tasks, f)
        
        # Call the function
        task = memory_manager.get_task()
        
        # Verify it returns the first task
        self.assertEqual(task['task'], 'test_task')
        self.assertEqual(task['details'], 'Test task details')

    def test_update_task(self):
        """Test update_task removes the first task."""
        # Create a test tasks file
        os.makedirs(os.path.dirname(self.mock_config.TASKS_FILE), exist_ok=True)
        
        test_tasks = [
            {
                'task': 'task_to_remove',
                'details': 'This should be removed'
            },
            {
                'task': 'task_to_keep',
                'details': 'This should remain'
            }
        ]
        
        with open(self.mock_config.TASKS_FILE, 'w') as f:
            json.dump(test_tasks, f)
        
        # Call the function
        memory_manager.update_task()
        
        # Verify the first task was removed
        with open(self.mock_config.TASKS_FILE, 'r') as f:
            remaining_tasks = json.load(f)
        
        self.assertEqual(len(remaining_tasks), 1)
        self.assertEqual(remaining_tasks[0]['task'], 'task_to_keep')

    def test_validate_task_valid(self):
        """Test validate_task with a valid task."""
        valid_task = {
            'task': 'test_task',
            'details': 'Test details'
        }
        
        result = memory_manager.validate_task(valid_task)
        self.assertTrue(result)

    def test_validate_task_invalid(self):
        """Test validate_task with invalid tasks."""
        # Not a dictionary
        self.assertFalse(memory_manager.validate_task("not a dict"))
        
        # Missing required field
        self.assertFalse(memory_manager.validate_task({'task': 'test_task'}))
        
        # Empty task field
        self.assertFalse(memory_manager.validate_task({'task': '', 'details': 'details'}))
        
        # Non-string task field
        self.assertFalse(memory_manager.validate_task({'task': 123, 'details': 'details'}))

    def test_add_tasks(self):
        """Test add_tasks with valid tasks."""
        # Ensure the tasks file doesn't exist initially
        if os.path.exists(self.mock_config.TASKS_FILE):
            os.remove(self.mock_config.TASKS_FILE)
        
        # Create test tasks
        new_tasks = [
            {
                'task': 'task1',
                'details': 'Details 1'
            },
            {
                'task': 'task2',
                'details': 'Details 2'
            }
        ]
        
        # Call the function
        memory_manager.add_tasks(new_tasks)
        
        # Verify tasks were added
        with open(self.mock_config.TASKS_FILE, 'r') as f:
            saved_tasks = json.load(f)
        
        self.assertEqual(len(saved_tasks), 2)
        self.assertEqual(saved_tasks[0]['task'], 'task1')
        self.assertEqual(saved_tasks[1]['task'], 'task2')
        
        # Add more tasks and verify they're appended
        more_tasks = [
            {
                'task': 'task3',
                'details': 'Details 3'
            }
        ]
        
        memory_manager.add_tasks(more_tasks)
        
        with open(self.mock_config.TASKS_FILE, 'r') as f:
            updated_tasks = json.load(f)
        
        self.assertEqual(len(updated_tasks), 3)
        self.assertEqual(updated_tasks[2]['task'], 'task3')

    def test_add_architecture_task(self):
        """Test add_architecture_task function."""
        # Ensure the tasks file doesn't exist initially
        if os.path.exists(self.mock_config.TASKS_FILE):
            os.remove(self.mock_config.TASKS_FILE)
        
        # Mock the add_tasks function to test it's called correctly
        with patch('memory_manager.add_tasks') as mock_add_tasks:
            # Call the function
            result = memory_manager.add_architecture_task()
            
            # Verify the result
            self.assertEqual(result['task'], 'architecture')
            self.assertIn('improve system architecture', result['details'])
            self.assertEqual(result['priority'], 5)
            
            # Verify add_tasks was called with the right task
            mock_add_tasks.assert_called_once()
            task_list = mock_add_tasks.call_args[0][0]
            self.assertEqual(len(task_list), 1)
            self.assertEqual(task_list[0]['task'], 'architecture')

    def test_add_testing_task(self):
        """Test add_testing_task function."""
        # Ensure the tasks file doesn't exist initially
        if os.path.exists(self.mock_config.TASKS_FILE):
            os.remove(self.mock_config.TASKS_FILE)
        
        # Mock the add_tasks function to test it's called correctly
        with patch('memory_manager.add_tasks') as mock_add_tasks:
            # Call the function with a specific module
            result = memory_manager.add_testing_task('test_module.py', 'unit')
            
            # Verify the result
            self.assertEqual(result['task'], 'testing')
            self.assertIn('test_module.py', result['details'])
            self.assertEqual(result['module'], 'test_module.py')
            self.assertEqual(result['test_type'], 'unit')
            self.assertEqual(result['priority'], 4)
            
            # Reset the mock
            mock_add_tasks.reset_mock()
            
            # Call the function without a specific module
            result = memory_manager.add_testing_task()
            
            # Verify the result
            self.assertEqual(result['task'], 'testing')
            self.assertIn('coverage', result['details'])
            self.assertEqual(result['test_type'], 'all')

if __name__ == '__main__':
    unittest.main() 