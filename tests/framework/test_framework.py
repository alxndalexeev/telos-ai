import os
import sys
import unittest
import importlib
import inspect
import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable, Set, Union, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from telos_logging.logger import log_action, log_thought

# Configure logging
logger = logging.getLogger(__name__)

# Constants
TEST_DIR = os.path.join(config.BASE_DIR, "tests")
UNIT_TEST_DIR = os.path.join(TEST_DIR, "unit")
INTEGRATION_TEST_DIR = os.path.join(TEST_DIR, "integration")
TEST_RESULTS_DIR = os.path.join(config.MEMORY_DIR, "test_results")
TEST_COVERAGE_FILE = os.path.join(TEST_RESULTS_DIR, "coverage.json")

# Ensure test directories exist
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(UNIT_TEST_DIR, exist_ok=True)
os.makedirs(INTEGRATION_TEST_DIR, exist_ok=True)
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# Create __init__.py files to make directories importable
for directory in [TEST_DIR, UNIT_TEST_DIR, INTEGRATION_TEST_DIR]:
    init_file = os.path.join(directory, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# Test package initialization file\n")

class TestResult:
    """Class to store test results in a structured format."""
    
    def __init__(self, 
                 test_name: str, 
                 test_type: str,
                 status: str, 
                 duration: float, 
                 error_message: Optional[str] = None,
                 module_name: Optional[str] = None):
        self.test_name = test_name
        self.test_type = test_type  # "unit" or "integration"
        self.status = status  # "pass", "fail", "error", "skip"
        self.duration = duration
        self.error_message = error_message
        self.module_name = module_name
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary."""
        return {
            "test_name": self.test_name,
            "test_type": self.test_type,
            "status": self.status,
            "duration": self.duration,
            "error_message": self.error_message,
            "module_name": self.module_name,
            "timestamp": self.timestamp
        }

class TelosTestRunner:
    """Custom test runner for Telos that handles test discovery, execution, and reporting."""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.start_time = 0.0
        self.end_time = 0.0
    
    def discover_tests(self, test_type: str = "all") -> List[unittest.TestCase]:
        """
        Discover all test cases in the tests directory.
        
        Args:
            test_type: Type of tests to discover: "unit", "integration", or "all"
            
        Returns:
            List of TestCase instances
        """
        logger.info(f"Discovering {test_type} tests...")
        
        test_dirs = []
        if test_type in ["unit", "all"]:
            test_dirs.append(UNIT_TEST_DIR)
        if test_type in ["integration", "all"]:
            test_dirs.append(INTEGRATION_TEST_DIR)
        
        discovered_tests = []
        
        for test_dir in test_dirs:
            # Add test directory to path temporarily
            sys.path.insert(0, os.path.dirname(test_dir))
            
            # Discover tests in the directory
            test_loader = unittest.TestLoader()
            dir_name = os.path.basename(test_dir)
            package_name = f"tests.{dir_name}"
            
            try:
                discovered_suite = test_loader.discover(
                    test_dir, 
                    pattern="test_*.py",
                    top_level_dir=os.path.dirname(TEST_DIR)
                )
                
                # Extract test cases from suite
                for suite in discovered_suite:
                    for test_case in suite:
                        if isinstance(test_case, unittest.TestCase):
                            discovered_tests.append(test_case)
                        else:
                            # Handle nested test suites
                            for nested_case in test_case:
                                if isinstance(nested_case, unittest.TestCase):
                                    discovered_tests.append(nested_case)
            except Exception as e:
                logger.error(f"Error discovering tests in {test_dir}: {e}")
            
            # Remove test directory from path
            sys.path.pop(0)
        
        logger.info(f"Discovered {len(discovered_tests)} test cases")
        return discovered_tests
    
    def run_tests(self, test_type: str = "all", parallel: bool = False, 
                  filter_pattern: Optional[str] = None) -> List[TestResult]:
        """
        Run discovered tests and collect results.
        
        Args:
            test_type: Type of tests to run: "unit", "integration", or "all"
            parallel: Whether to run tests in parallel
            filter_pattern: Optional string pattern to filter test names
            
        Returns:
            List of TestResult objects
        """
        logger.info(f"Running {test_type} tests...")
        self.test_results = []
        self.start_time = time.time()
        
        # Discover tests
        test_cases = self.discover_tests(test_type)
        
        # Filter tests if pattern provided
        if filter_pattern:
            filtered_cases = []
            for test_case in test_cases:
                if filter_pattern.lower() in test_case.id().lower():
                    filtered_cases.append(test_case)
            test_cases = filtered_cases
            logger.info(f"Filtered to {len(test_cases)} tests matching pattern '{filter_pattern}'")
        
        if not test_cases:
            logger.warning("No tests found to run")
            self.end_time = time.time()
            return []
        
        # Run tests
        if parallel and len(test_cases) > 1:
            self._run_tests_parallel(test_cases)
        else:
            self._run_tests_sequential(test_cases)
        
        self.end_time = time.time()
        self._save_results()
        
        logger.info(f"Completed {len(self.test_results)} tests in {self.end_time - self.start_time:.2f} seconds")
        return self.test_results
    
    def _run_tests_sequential(self, test_cases: List[unittest.TestCase]) -> None:
        """Run tests sequentially."""
        for test_case in test_cases:
            self._run_single_test(test_case)
    
    def _run_tests_parallel(self, test_cases: List[unittest.TestCase]) -> None:
        """Run tests in parallel using a thread pool."""
        with ThreadPoolExecutor() as executor:
            # Submit all test cases to the executor
            future_to_test = {executor.submit(self._run_single_test, test_case): test_case 
                             for test_case in test_cases}
            
            # Collect results as they complete
            for future in as_completed(future_to_test):
                test_case = future_to_test[future]
                try:
                    future.result()  # Get the result (or exception)
                except Exception as e:
                    logger.error(f"Error running test {test_case.id()}: {e}")
    
    def _run_single_test(self, test_case: unittest.TestCase) -> None:
        """Run a single test case and record the result."""
        test_id = test_case.id()
        test_name = test_id.split('.')[-1]
        module_name = '.'.join(test_id.split('.')[:-1])
        
        # Determine test type (unit or integration)
        test_type = "unit" if "unit" in test_id else "integration"
        
        # Create a test result object
        result = unittest.TestResult()
        
        # Run the test
        start_time = time.time()
        test_case.run(result)
        duration = time.time() - start_time
        
        # Process the result
        if result.wasSuccessful():
            status = "pass"
            error_message = None
        elif result.failures:
            status = "fail"
            error_message = result.failures[0][1]
        elif result.errors:
            status = "error"
            error_message = result.errors[0][1]
        elif result.skipped:
            status = "skip"
            error_message = result.skipped[0][1]
        else:
            status = "unknown"
            error_message = "Unknown test result status"
        
        # Create test result object
        test_result = TestResult(
            test_name=test_name,
            test_type=test_type,
            status=status,
            duration=duration,
            error_message=error_message,
            module_name=module_name
        )
        
        # Add to results list
        self.test_results.append(test_result)
        
        # Log the result
        if status == "pass":
            logger.info(f"PASS: {test_id} ({duration:.2f}s)")
        else:
            logger.error(f"{status.upper()}: {test_id} ({duration:.2f}s)")
            if error_message:
                logger.error(f"Error details: {error_message[:200]}...")
    
    def _save_results(self) -> None:
        """Save test results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(TEST_RESULTS_DIR, f"test_results_{timestamp}.json")
        
        results_data = {
            "timestamp": timestamp,
            "total_duration": self.end_time - self.start_time,
            "total_tests": len(self.test_results),
            "passed": sum(1 for r in self.test_results if r.status == "pass"),
            "failed": sum(1 for r in self.test_results if r.status == "fail"),
            "errors": sum(1 for r in self.test_results if r.status == "error"),
            "skipped": sum(1 for r in self.test_results if r.status == "skip"),
            "tests": [r.to_dict() for r in self.test_results]
        }
        
        try:
            with open(results_file, "w") as f:
                json.dump(results_data, f, indent=2)
            logger.info(f"Test results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving test results: {e}")
    
    def generate_report(self, most_recent: bool = True, 
                        results_file: Optional[str] = None) -> str:
        """
        Generate a human-readable test report.
        
        Args:
            most_recent: Whether to use the most recent results file
            results_file: Path to specific results file to use
            
        Returns:
            Report string
        """
        # Get the results file to use
        if results_file and os.path.exists(results_file):
            file_path = results_file
        elif most_recent:
            result_files = [os.path.join(TEST_RESULTS_DIR, f) for f in os.listdir(TEST_RESULTS_DIR)
                            if f.startswith("test_results_") and f.endswith(".json")]
            if not result_files:
                return "No test results found."
            file_path = max(result_files, key=os.path.getmtime)
        else:
            return "No test results file specified."
        
        # Load the results
        try:
            with open(file_path, "r") as f:
                results_data = json.load(f)
        except Exception as e:
            return f"Error loading test results: {e}"
        
        # Generate the report
        report = []
        report.append("# Telos Test Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Results from: {os.path.basename(file_path)}")
        report.append("")
        report.append("## Summary")
        report.append(f"- Total tests: {results_data['total_tests']}")
        report.append(f"- Passed: {results_data['passed']}")
        report.append(f"- Failed: {results_data['failed']}")
        report.append(f"- Errors: {results_data['errors']}")
        report.append(f"- Skipped: {results_data['skipped']}")
        report.append(f"- Duration: {results_data['total_duration']:.2f} seconds")
        report.append("")
        
        # Add failed tests
        failed_tests = [t for t in results_data["tests"] 
                       if t["status"] in ["fail", "error"]]
        
        if failed_tests:
            report.append("## Failed Tests")
            for test in failed_tests:
                report.append(f"### {test['module_name']}.{test['test_name']}")
                report.append(f"- Status: {test['status']}")
                report.append(f"- Duration: {test['duration']:.2f} seconds")
                if test.get("error_message"):
                    report.append("- Error message:")
                    report.append(f"```\n{test['error_message'][:500]}...\n```")
                report.append("")
        
        return "\n".join(report)

class TestGenerator:
    """Class to generate test cases for modules in the Telos system."""
    
    def __init__(self):
        self.base_dir = config.BASE_DIR
    
    def analyze_module(self, module_path: str) -> Dict[str, Any]:
        """
        Analyze a Python module to gather information for test generation.
        
        Args:
            module_path: Path to the module file relative to BASE_DIR
            
        Returns:
            Dictionary with module information
        """
        abs_path = os.path.join(self.base_dir, module_path)
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        
        if not os.path.exists(abs_path):
            logger.error(f"Module file not found: {abs_path}")
            return {"name": module_name, "functions": [], "classes": []}
        
        # Import the module
        sys.path.insert(0, os.path.dirname(abs_path))
        try:
            module = importlib.import_module(module_name)
            
            # Get all functions and classes
            functions = []
            classes = []
            
            for name, obj in inspect.getmembers(module):
                # Skip private members
                if name.startswith("_"):
                    continue
                
                # Collect function information
                if inspect.isfunction(obj):
                    functions.append({
                        "name": name,
                        "signature": str(inspect.signature(obj)),
                        "docstring": inspect.getdoc(obj) or "",
                        "source_code": inspect.getsource(obj)
                    })
                
                # Collect class information
                elif inspect.isclass(obj):
                    methods = []
                    for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                        if not method_name.startswith("_") or method_name == "__init__":
                            methods.append({
                                "name": method_name,
                                "signature": str(inspect.signature(method)),
                                "docstring": inspect.getdoc(method) or "",
                                "source_code": inspect.getsource(method)
                            })
                    
                    classes.append({
                        "name": name,
                        "docstring": inspect.getdoc(obj) or "",
                        "methods": methods
                    })
            
            return {
                "name": module_name,
                "functions": functions,
                "classes": classes
            }
            
        except Exception as e:
            logger.error(f"Error analyzing module {module_path}: {e}")
            return {"name": module_name, "functions": [], "classes": []}
        finally:
            # Remove the path we added
            sys.path.pop(0)
    
    def generate_unit_test(self, module_path: str) -> str:
        """
        Generate a unit test file for a module.
        
        Args:
            module_path: Path to the module file relative to BASE_DIR
            
        Returns:
            Path to the generated test file
        """
        # Analyze the module
        module_info = self.analyze_module(module_path)
        module_name = module_info["name"]
        
        # Create test file path
        test_file_name = f"test_{module_name}.py"
        test_file_path = os.path.join(UNIT_TEST_DIR, test_file_name)
        
        # Generate test file content
        content = []
        content.append("import unittest")
        content.append("import sys")
        content.append("import os")
        content.append("import json")
        content.append("from unittest.mock import patch, MagicMock")
        content.append("")
        content.append(f"# Import the module to test")
        content.append(f"sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))")
        content.append(f"import {module_name}")
        content.append("sys.path.pop(0)")
        content.append("")
        content.append(f"class Test{module_name.capitalize()}(unittest.TestCase):")
        content.append(f"    \"\"\"Unit tests for the {module_name} module.\"\"\"")
        content.append("")
        content.append("    def setUp(self):")
        content.append("        \"\"\"Set up test fixtures, if any.\"\"\"")
        content.append("        pass")
        content.append("")
        content.append("    def tearDown(self):")
        content.append("        \"\"\"Tear down test fixtures, if any.\"\"\"")
        content.append("        pass")
        content.append("")
        
        # Generate test methods for functions
        for func in module_info["functions"]:
            func_name = func["name"]
            content.append(f"    def test_{func_name}(self):")
            content.append(f"        \"\"\"Test {func_name} function.\"\"\"")
            content.append(f"        # TODO: Implement test for {func_name}")
            content.append("        pass")
            content.append("")
        
        # Generate test methods for classes
        for cls in module_info["classes"]:
            cls_name = cls["name"]
            
            # Generate class setup method
            content.append(f"    def test_{cls_name}_initialization(self):")
            content.append(f"        \"\"\"Test {cls_name} class initialization.\"\"\"")
            content.append(f"        # TODO: Implement test for {cls_name} initialization")
            content.append("        pass")
            content.append("")
            
            # Generate test methods for class methods
            for method in cls["methods"]:
                method_name = method["name"]
                if method_name == "__init__":  # Skip init, covered in initialization test
                    continue
                
                content.append(f"    def test_{cls_name}_{method_name}(self):")
                content.append(f"        \"\"\"Test {cls_name}.{method_name} method.\"\"\"")
                content.append(f"        # TODO: Implement test for {cls_name}.{method_name}")
                content.append("        pass")
                content.append("")
        
        # Add main block
        content.append("if __name__ == '__main__':")
        content.append("    unittest.main()")
        
        # Write to file
        with open(test_file_path, "w") as f:
            f.write("\n".join(content))
        
        logger.info(f"Generated unit test file: {test_file_path}")
        return test_file_path
    
    def generate_integration_test(self, module_paths: List[str]) -> str:
        """
        Generate an integration test file for multiple modules.
        
        Args:
            module_paths: List of paths to module files relative to BASE_DIR
            
        Returns:
            Path to the generated test file
        """
        # Get module names
        module_names = [os.path.splitext(os.path.basename(path))[0] for path in module_paths]
        
        # Create test file name based on the modules
        if len(module_names) <= 2:
            test_name = "_".join(module_names)
        else:
            test_name = f"{module_names[0]}_and_others"
        
        # Create test file path
        test_file_name = f"test_integration_{test_name}.py"
        test_file_path = os.path.join(INTEGRATION_TEST_DIR, test_file_name)
        
        # Generate test file content
        content = []
        content.append("import unittest")
        content.append("import sys")
        content.append("import os")
        content.append("import json")
        content.append("from unittest.mock import patch, MagicMock")
        content.append("")
        
        # Import the modules to test
        content.append("# Import the modules to test")
        content.append("sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))")
        for module_name in module_names:
            content.append(f"import {module_name}")
        content.append("sys.path.pop(0)")
        content.append("")
        
        # Create test class
        content.append(f"class Test{test_name.capitalize()}Integration(unittest.TestCase):")
        content.append(f"    \"\"\"Integration tests for the {', '.join(module_names)} modules.\"\"\"")
        content.append("")
        content.append("    def setUp(self):")
        content.append("        \"\"\"Set up test fixtures, if any.\"\"\"")
        content.append("        pass")
        content.append("")
        content.append("    def tearDown(self):")
        content.append("        \"\"\"Tear down test fixtures, if any.\"\"\"")
        content.append("        pass")
        content.append("")
        
        # Add sample integration test methods
        content.append("    def test_integration_workflow(self):")
        content.append(f"        \"\"\"Test the integration between {', '.join(module_names)}.\"\"\"")
        content.append("        # TODO: Implement integration test")
        content.append("        pass")
        content.append("")
        
        # Add main block
        content.append("if __name__ == '__main__':")
        content.append("    unittest.main()")
        
        # Write to file
        with open(test_file_path, "w") as f:
            f.write("\n".join(content))
        
        logger.info(f"Generated integration test file: {test_file_path}")
        return test_file_path

class CoverageAnalyzer:
    """Class to analyze test coverage for Telos modules."""
    
    def __init__(self):
        self.base_dir = config.BASE_DIR
        self.coverage_data = {
            "timestamp": datetime.now().isoformat(),
            "modules": {}
        }
    
    def analyze_code_coverage(self) -> Dict[str, Any]:
        """
        Analyze code coverage for all Python modules in the project.
        
        Returns:
            Dictionary with coverage information
        """
        # Find all Python modules
        python_modules = []
        for root, _, files in os.walk(self.base_dir):
            # Skip test directories, virtual environments, etc.
            if any(excluded in root for excluded in ["tests", ".venv", "__pycache__", ".git"]):
                continue
            
            for file in files:
                if file.endswith(".py"):
                    rel_path = os.path.relpath(os.path.join(root, file), self.base_dir)
                    python_modules.append(rel_path)
        
        # For each module, check if there are unit tests
        for module_path in python_modules:
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            test_file = os.path.join(UNIT_TEST_DIR, f"test_{module_name}.py")
            
            # Check if test file exists
            has_unit_tests = os.path.exists(test_file)
            
            # Analyze the module to count testable items
            test_gen = TestGenerator()
            module_info = test_gen.analyze_module(module_path)
            
            # Count functions and methods
            function_count = len(module_info["functions"])
            class_count = len(module_info["classes"])
            method_count = sum(len(cls["methods"]) for cls in module_info["classes"])
            
            # Count actual test methods if the test file exists
            test_method_count = 0
            test_class_count = 0
            
            if has_unit_tests:
                try:
                    # Analyze the test file
                    with open(test_file, "r") as f:
                        test_content = f.read()
                    
                    # Count test methods (simple approach)
                    test_method_count = test_content.count("def test_")
                    
                    # Count test classes
                    test_class_count = test_content.count("class Test")
                except Exception as e:
                    logger.error(f"Error analyzing test file {test_file}: {e}")
            
            # Calculate coverage metrics
            testable_items = function_count + method_count
            test_items = test_method_count
            
            coverage_percent = (test_items / testable_items * 100) if testable_items > 0 else 0
            
            # Store in coverage data
            self.coverage_data["modules"][module_name] = {
                "path": module_path,
                "has_unit_tests": has_unit_tests,
                "function_count": function_count,
                "class_count": class_count,
                "method_count": method_count,
                "test_method_count": test_method_count,
                "test_class_count": test_class_count,
                "coverage_percent": coverage_percent
            }
        
        # Calculate overall coverage
        total_testable = sum(m["function_count"] + m["method_count"] 
                              for m in self.coverage_data["modules"].values())
        total_tested = sum(m["test_method_count"] 
                            for m in self.coverage_data["modules"].values())
        
        self.coverage_data["total_coverage_percent"] = (
            total_tested / total_testable * 100 if total_testable > 0 else 0
        )
        
        # Save coverage data
        try:
            with open(TEST_COVERAGE_FILE, "w") as f:
                json.dump(self.coverage_data, f, indent=2)
            logger.info(f"Coverage data saved to {TEST_COVERAGE_FILE}")
        except Exception as e:
            logger.error(f"Error saving coverage data: {e}")
        
        return self.coverage_data
    
    def generate_coverage_report(self) -> str:
        """
        Generate a human-readable coverage report.
        
        Returns:
            Report string
        """
        # Load coverage data if not already analyzed
        if not self.coverage_data.get("modules"):
            try:
                if os.path.exists(TEST_COVERAGE_FILE):
                    with open(TEST_COVERAGE_FILE, "r") as f:
                        self.coverage_data = json.load(f)
                else:
                    self.analyze_code_coverage()
            except Exception as e:
                return f"Error loading coverage data: {e}"
        
        # Generate the report
        report = []
        report.append("# Telos Test Coverage Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("## Summary")
        report.append(f"- Overall coverage: {self.coverage_data.get('total_coverage_percent', 0):.2f}%")
        report.append(f"- Total modules analyzed: {len(self.coverage_data.get('modules', {}))}")
        report.append(f"- Modules with tests: {sum(1 for m in self.coverage_data.get('modules', {}).values() if m.get('has_unit_tests'))}")
        report.append("")
        
        # Add details for each module
        report.append("## Module Details")
        report.append("")
        
        # Sort modules by coverage (lowest first)
        sorted_modules = sorted(
            self.coverage_data.get("modules", {}).items(),
            key=lambda x: x[1].get("coverage_percent", 0)
        )
        
        for module_name, module_data in sorted_modules:
            report.append(f"### {module_name}")
            report.append(f"- Path: {module_data.get('path')}")
            report.append(f"- Coverage: {module_data.get('coverage_percent', 0):.2f}%")
            report.append(f"- Functions: {module_data.get('function_count', 0)}")
            report.append(f"- Classes: {module_data.get('class_count', 0)}")
            report.append(f"- Methods: {module_data.get('method_count', 0)}")
            report.append(f"- Has unit tests: {module_data.get('has_unit_tests', False)}")
            if module_data.get('has_unit_tests', False):
                report.append(f"- Test methods: {module_data.get('test_method_count', 0)}")
                report.append(f"- Test classes: {module_data.get('test_class_count', 0)}")
            report.append("")
        
        return "\n".join(report)

# --- Functions for integration with executor.py ---

def run_tests(test_type: str = "all", parallel: bool = False, 
             filter_pattern: Optional[str] = None) -> str:
    """
    Run tests and return the results as a string.
    
    Args:
        test_type: Type of tests to run: "unit", "integration", or "all"
        parallel: Whether to run tests in parallel
        filter_pattern: Optional string pattern to filter test names
        
    Returns:
        Result message
    """
    try:
        test_runner = TelosTestRunner()
        results = test_runner.run_tests(test_type, parallel, filter_pattern)
        
        if not results:
            return "No tests were run."
        
        # Generate basic report
        passed = sum(1 for r in results if r.status == "pass")
        failed = sum(1 for r in results if r.status == "fail")
        errors = sum(1 for r in results if r.status == "error")
        skipped = sum(1 for r in results if r.status == "skip")
        
        report = (f"Ran {len(results)} tests. "
                 f"Passed: {passed}, Failed: {failed}, Errors: {errors}, Skipped: {skipped}")
        
        # Log the result
        log_action("Test Run", report)
        
        return report
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return f"Error running tests: {e}"

def generate_test_for_module(module_path: str) -> str:
    """
    Generate unit tests for a module.
    
    Args:
        module_path: Path to the module file relative to BASE_DIR
        
    Returns:
        Result message
    """
    try:
        test_gen = TestGenerator()
        test_file = test_gen.generate_unit_test(module_path)
        
        log_action("Test Generation", f"Generated unit tests for {module_path} at {test_file}")
        
        return f"Successfully generated unit tests for {module_path} at {test_file}"
    except Exception as e:
        logger.error(f"Error generating tests for {module_path}: {e}")
        return f"Error generating tests for {module_path}: {e}"

def generate_integration_test(module_paths: str) -> str:
    """
    Generate integration tests for multiple modules.
    
    Args:
        module_paths: Comma-separated list of module paths relative to BASE_DIR
        
    Returns:
        Result message
    """
    try:
        # Split the comma-separated list
        module_list = [path.strip() for path in module_paths.split(",")]
        
        test_gen = TestGenerator()
        test_file = test_gen.generate_integration_test(module_list)
        
        log_action("Test Generation", 
                  f"Generated integration tests for {', '.join(module_list)} at {test_file}")
        
        return f"Successfully generated integration tests at {test_file}"
    except Exception as e:
        logger.error(f"Error generating integration tests: {e}")
        return f"Error generating integration tests: {e}"

def analyze_test_coverage() -> str:
    """
    Analyze test coverage for all modules.
    
    Returns:
        Result message
    """
    try:
        coverage_analyzer = CoverageAnalyzer()
        coverage_data = coverage_analyzer.analyze_code_coverage()
        
        # Get summary information
        total_coverage = coverage_data.get("total_coverage_percent", 0)
        total_modules = len(coverage_data.get("modules", {}))
        modules_with_tests = sum(1 for m in coverage_data.get("modules", {}).values() 
                                 if m.get("has_unit_tests"))
        
        summary = (f"Test coverage analysis complete. "
                  f"Overall coverage: {total_coverage:.2f}%. "
                  f"{modules_with_tests}/{total_modules} modules have tests.")
        
        log_action("Test Coverage Analysis", summary)
        
        return summary
    except Exception as e:
        logger.error(f"Error analyzing test coverage: {e}")
        return f"Error analyzing test coverage: {e}"

def generate_test_report() -> str:
    """
    Generate a test report from the most recent test run.
    
    Returns:
        Result message
    """
    try:
        test_runner = TelosTestRunner()
        report = test_runner.generate_report()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(TEST_RESULTS_DIR, f"test_report_{timestamp}.md")
        
        with open(report_file, "w") as f:
            f.write(report)
        
        log_action("Test Reporting", f"Generated test report at {report_file}")
        
        return f"Test report generated and saved to {report_file}"
    except Exception as e:
        logger.error(f"Error generating test report: {e}")
        return f"Error generating test report: {e}"

def generate_coverage_report() -> str:
    """
    Generate a coverage report.
    
    Returns:
        Result message
    """
    try:
        coverage_analyzer = CoverageAnalyzer()
        report = coverage_analyzer.generate_coverage_report()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(TEST_RESULTS_DIR, f"coverage_report_{timestamp}.md")
        
        with open(report_file, "w") as f:
            f.write(report)
        
        log_action("Test Coverage Reporting", f"Generated coverage report at {report_file}")
        
        return f"Coverage report generated and saved to {report_file}"
    except Exception as e:
        logger.error(f"Error generating coverage report: {e}")
        return f"Error generating coverage report: {e}" 