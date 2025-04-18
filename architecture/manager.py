import os
import sys
import ast
import inspect
import importlib
import logging
import json
import shutil
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import config
from logging.logger import log_action, log_thought

logger = logging.getLogger(__name__)

# Constants
ARCHITECTURE_DIR = os.path.join(config.MEMORY_DIR, "architecture")
ARCHITECTURE_BACKUPS = os.path.join(ARCHITECTURE_DIR, "backups")
ARCHITECTURE_PROPOSALS = os.path.join(ARCHITECTURE_DIR, "proposals")
MODULE_DEPENDENCIES = os.path.join(ARCHITECTURE_DIR, "dependencies.json")

# Ensure directories exist
os.makedirs(ARCHITECTURE_DIR, exist_ok=True)
os.makedirs(ARCHITECTURE_BACKUPS, exist_ok=True)
os.makedirs(ARCHITECTURE_PROPOSALS, exist_ok=True)

class ArchitectureManager:
    """
    Manages architectural self-improvement for Telos.
    
    This class provides capabilities for:
    1. Analyzing the current architecture
    2. Proposing architectural improvements
    3. Implementing and testing changes
    4. Rolling back changes if needed
    """
    
    def __init__(self):
        self.modules = {}
        self.dependencies = {}
        self.entry_points = ["heart.py"]
        self._load_dependencies()
    
    def _load_dependencies(self) -> None:
        """Load the saved module dependencies if available."""
        if os.path.exists(MODULE_DEPENDENCIES):
            try:
                with open(MODULE_DEPENDENCIES, 'r') as f:
                    self.dependencies = json.load(f)
                logger.info(f"Loaded {len(self.dependencies)} module dependencies")
            except Exception as e:
                logger.error(f"Error loading module dependencies: {e}")
    
    def _save_dependencies(self) -> None:
        """Save the current module dependencies."""
        try:
            with open(MODULE_DEPENDENCIES, 'w') as f:
                json.dump(self.dependencies, f, indent=2)
            logger.info(f"Saved {len(self.dependencies)} module dependencies")
        except Exception as e:
            logger.error(f"Error saving module dependencies: {e}")
    
    def analyze_architecture(self) -> Dict[str, Any]:
        """
        Analyze the current architecture and return a description.
        
        Returns:
            Dict containing architecture information including:
            - modules: list of module files
            - dependencies: dependency graph
            - entry_points: main entry points
            - metrics: code metrics like complexity, cohesion
        """
        architecture_info = {
            "timestamp": datetime.now().isoformat(),
            "modules": [],
            "dependencies": {},
            "entry_points": self.entry_points,
            "metrics": {}
        }
        
        # Find Python modules
        python_files = []
        for root, _, files in os.walk(config.BASE_DIR):
            # Skip virtual environment, git, and memory directories
            if any(excluded in root for excluded in ['.venv', '.git', 'memory', '__pycache__']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    rel_path = os.path.relpath(os.path.join(root, file), config.BASE_DIR)
                    python_files.append(rel_path)
        
        # Analyze each module
        for py_file in python_files:
            try:
                module_info = self._analyze_module(py_file)
                architecture_info["modules"].append(module_info)
                
                # Update dependencies
                if module_info["name"] not in self.dependencies:
                    self.dependencies[module_info["name"]] = module_info["imports"]
            except Exception as e:
                logger.error(f"Error analyzing module {py_file}: {e}")
        
        # Update the dependencies graph
        architecture_info["dependencies"] = self.dependencies
        self._save_dependencies()
        
        # Calculate metrics
        architecture_info["metrics"] = self._calculate_metrics(architecture_info)
        
        return architecture_info
    
    def _analyze_module(self, module_path: str) -> Dict[str, Any]:
        """
        Analyze a single module file.
        
        Args:
            module_path: Relative path to the module from BASE_DIR
            
        Returns:
            Dict with module information
        """
        abs_path = os.path.join(config.BASE_DIR, module_path)
        
        with open(abs_path, 'r') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Extract imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # Extract classes and functions
        classes = []
        functions = []
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        
        return {
            "name": module_name,
            "path": module_path,
            "size": len(content),
            "imports": imports,
            "classes": classes,
            "functions": functions,
            "docstring": ast.get_docstring(tree)
        }
    
    def _calculate_metrics(self, architecture_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate architecture metrics.
        
        Args:
            architecture_info: The architecture information dictionary
            
        Returns:
            Dict with metrics
        """
        modules = architecture_info["modules"]
        dependencies = architecture_info["dependencies"]
        
        # Basic metrics
        metrics = {
            "module_count": len(modules),
            "total_loc": sum(m["size"] for m in modules),
            "avg_module_size": sum(m["size"] for m in modules) / max(len(modules), 1),
            "dependency_count": sum(len(deps) for deps in dependencies.values()),
            "cohesion": 0,  # Will be calculated
            "coupling": 0,  # Will be calculated
        }
        
        # Calculate coupling (average dependencies per module)
        if modules:
            metrics["coupling"] = metrics["dependency_count"] / len(modules)
        
        # Calculate cohesion (average internal connections)
        # This is a simple approximation
        internal_connections = 0
        for module in modules:
            # Count internal relationships between functions and classes
            internal_connections += len(module["functions"]) * len(module["classes"])
        
        if modules:
            metrics["cohesion"] = internal_connections / len(modules)
        
        return metrics
    
    def backup_architecture(self) -> str:
        """
        Create a backup of the current architecture.
        
        Returns:
            Path to the backup directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(ARCHITECTURE_BACKUPS, f"backup_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        
        logger.info(f"Creating architecture backup at {backup_dir}")
        
        # Copy all Python files
        for root, _, files in os.walk(config.BASE_DIR):
            # Skip virtual environment, git, and memory directories
            if any(excluded in root for excluded in ['.venv', '.git', 'memory', '__pycache__']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    src_path = os.path.join(root, file)
                    rel_path = os.path.relpath(src_path, config.BASE_DIR)
                    dst_path = os.path.join(backup_dir, rel_path)
                    
                    # Create directory structure if needed
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    
                    # Copy the file
                    shutil.copy2(src_path, dst_path)
        
        # Store the analysis
        analysis = self.analyze_architecture()
        with open(os.path.join(backup_dir, "architecture_analysis.json"), 'w') as f:
            json.dump(analysis, f, indent=2)
            
        log_action("Architecture Backup", f"Created architecture backup at {backup_dir}")
        return backup_dir
    
    def propose_improvements(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Based on architectural analysis, propose improvements.
        This will typically call an LLM to analyze the architecture.
        
        Args:
            analysis: The architecture analysis from analyze_architecture()
            
        Returns:
            List of improvement proposals
        """
        # This would interface with an LLM in a real implementation
        # For now, return a placeholder
        
        # Simplified example (typically would be done via LLM)
        proposals = []
        
        # Identify potential improvements based on metrics
        metrics = analysis["metrics"]
        
        # Check for high coupling
        if metrics["coupling"] > 5:  # Arbitrary threshold
            proposals.append({
                "type": "refactoring",
                "target": "architecture",
                "description": "Reduce coupling between modules by extracting common functionality",
                "priority": "high"
            })
        
        # Check for large modules
        for module in analysis["modules"]:
            if module["size"] > 500:  # Arbitrary threshold
                proposals.append({
                    "type": "refactoring",
                    "target": module["path"],
                    "description": f"Split large module {module['name']} into smaller components",
                    "priority": "medium"
                })
        
        # Check for missing documentation
        for module in analysis["modules"]:
            if not module["docstring"]:
                proposals.append({
                    "type": "documentation",
                    "target": module["path"],
                    "description": f"Add module-level documentation to {module['name']}",
                    "priority": "low"
                })
        
        return proposals
    
    def implement_proposal(self, proposal_id: str, llm_prompt: str) -> Tuple[bool, str]:
        """
        Implement an architectural improvement proposal.
        
        Args:
            proposal_id: Identifier for the proposal
            llm_prompt: Prompt for the LLM to generate the implementation
            
        Returns:
            (success, message) tuple
        """
        # In a real implementation, this would:
        # 1. Create a backup
        # 2. Generate implementation code (via executor.py code_generation)
        # 3. Apply the changes
        # 4. Test the changes
        # 5. Rollback if tests fail
        
        backup_dir = self.backup_architecture()
        
        # Implementation would be integrated with executor.py's code generation
        # For now, just return success and a placeholder message
        
        return True, f"Implementation would be generated here. Backup created at {backup_dir}"
    
    def test_architecture(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Test the current architecture to ensure it's valid.
        
        Returns:
            (success, test_results) tuple
        """
        # This is a placeholder for actual tests
        # In a real implementation, this would:
        # 1. Run unit tests
        # 2. Check for import errors
        # 3. Verify core functionality
        
        test_results = {
            "import_tests": {},
            "syntax_tests": {},
            "functional_tests": {}
        }
        
        # Test imports
        for module_name, imports in self.dependencies.items():
            test_results["import_tests"][module_name] = {"success": True, "errors": []}
            
            try:
                # Try to import the module
                if os.path.exists(os.path.join(config.BASE_DIR, f"{module_name}.py")):
                    sys.path.insert(0, config.BASE_DIR)
                    importlib.import_module(module_name)
                    sys.path.pop(0)
            except Exception as e:
                test_results["import_tests"][module_name] = {
                    "success": False,
                    "errors": [str(e)]
                }
        
        # Test syntax of all Python files
        for root, _, files in os.walk(config.BASE_DIR):
            # Skip virtual environment, git, and memory directories
            if any(excluded in root for excluded in ['.venv', '.git', 'memory', '__pycache__']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, config.BASE_DIR)
                    
                    test_results["syntax_tests"][rel_path] = {"success": True, "errors": []}
                    
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        ast.parse(content)
                    except SyntaxError as e:
                        test_results["syntax_tests"][rel_path] = {
                            "success": False,
                            "errors": [str(e)]
                        }
        
        # Determine overall success
        import_success = all(test["success"] for test in test_results["import_tests"].values())
        syntax_success = all(test["success"] for test in test_results["syntax_tests"].values())
        
        # Overall success requires both import and syntax tests to pass
        success = import_success and syntax_success
        
        return success, test_results
    
    def rollback(self, backup_dir: str) -> Tuple[bool, str]:
        """
        Rollback to a previous architecture backup.
        
        Args:
            backup_dir: Path to the backup directory
            
        Returns:
            (success, message) tuple
        """
        try:
            # Check if backup exists
            if not os.path.exists(backup_dir):
                return False, f"Backup directory {backup_dir} does not exist"
            
            # Create new backup of current state before rollback
            current_backup = self.backup_architecture()
            
            # Copy files from backup to BASE_DIR
            for root, _, files in os.walk(backup_dir):
                for file in files:
                    if file.endswith('.py'):
                        src_path = os.path.join(root, file)
                        rel_path = os.path.relpath(src_path, backup_dir)
                        dst_path = os.path.join(config.BASE_DIR, rel_path)
                        
                        # Create directory structure if needed
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        
                        # Copy the file
                        shutil.copy2(src_path, dst_path)
            
            log_action("Architecture Rollback", f"Rolled back to backup {backup_dir}")
            return True, f"Successfully rolled back to {backup_dir}. Current state backed up to {current_backup}."
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False, f"Rollback failed: {e}"
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current architecture.
        
        Returns:
            Dict with architecture summary
        """
        analysis = self.analyze_architecture()
        
        # Create a simplified summary
        summary = {
            "module_count": len(analysis["modules"]),
            "entry_points": analysis["entry_points"],
            "core_modules": [m["name"] for m in analysis["modules"] if m["name"] in ["heart", "executor", "planner", "memory_manager"]],
            "metrics": analysis["metrics"]
        }
        
        return summary


# --- Functions for integration with executor.py ---

def analyze_architecture() -> str:
    """Execute architecture analysis and return the results as a string."""
    try:
        arch_manager = ArchitectureManager()
        analysis = arch_manager.analyze_architecture()
        
        # Save the analysis to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = os.path.join(ARCHITECTURE_DIR, f"analysis_{timestamp}.json")
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Create a summary for logging
        summary = arch_manager.get_architecture_summary()
        summary_str = json.dumps(summary, indent=2)
        
        log_action("Architecture Analysis", f"Completed architecture analysis. Results in {analysis_file}")
        
        return f"Architecture analysis complete. Summary: {summary_str}. Full analysis saved to {analysis_file}"
        
    except Exception as e:
        logger.error(f"Error in architecture analysis: {e}")
        return f"Architecture analysis failed: {e}"

def propose_architectural_improvements() -> str:
    """Generate architectural improvement proposals."""
    try:
        arch_manager = ArchitectureManager()
        analysis = arch_manager.analyze_architecture()
        proposals = arch_manager.propose_improvements(analysis)
        
        # Save proposals to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        proposals_file = os.path.join(ARCHITECTURE_PROPOSALS, f"proposals_{timestamp}.json")
        
        with open(proposals_file, 'w') as f:
            json.dump(proposals, f, indent=2)
        
        # Create a summary for logging
        summary = f"Generated {len(proposals)} architectural improvement proposals"
        if proposals:
            summary += f": " + ", ".join(p["description"][:50] + "..." for p in proposals[:3])
            if len(proposals) > 3:
                summary += f" and {len(proposals)-3} more"
        
        log_action("Architecture Proposals", summary)
        
        return f"Architecture proposals generated: {summary}. Saved to {proposals_file}"
        
    except Exception as e:
        logger.error(f"Error generating architecture proposals: {e}")
        return f"Failed to generate architecture proposals: {e}"

def implement_architectural_change(proposal_file: str, target_component: str) -> str:
    """
    Implement an architectural change based on a proposal.
    
    Args:
        proposal_file: Path to the proposal file
        target_component: The component to modify
        
    Returns:
        Result message
    """
    try:
        # Create architecture manager
        arch_manager = ArchitectureManager()
        
        # Load the proposal
        with open(proposal_file, 'r') as f:
            proposals = json.load(f)
        
        # Find the matching proposal
        target_proposal = None
        for proposal in proposals:
            if proposal.get("target") == target_component:
                target_proposal = proposal
                break
        
        if not target_proposal:
            return f"No proposal found for component {target_component} in {proposal_file}"
        
        # Create a prompt for the implementation
        llm_prompt = f"""
        Implement the following architectural change for Telos:
        
        Component: {target_component}
        Description: {target_proposal['description']}
        Type: {target_proposal['type']}
        Priority: {target_proposal['priority']}
        
        Current architecture summary:
        {json.dumps(arch_manager.get_architecture_summary(), indent=2)}
        
        Please provide the complete implementation for this architectural change.
        The implementation should maintain or improve the system's functionality.
        """
        
        # Implement the proposal
        success, message = arch_manager.implement_proposal(
            proposal_id=f"{target_component}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            llm_prompt=llm_prompt
        )
        
        if success:
            log_action("Architecture Implementation", 
                       f"Implemented proposal for {target_component}: {target_proposal['description']}")
            return f"Successfully implemented architectural change: {message}"
        else:
            return f"Failed to implement architectural change: {message}"
        
    except Exception as e:
        logger.error(f"Error implementing architectural change: {e}")
        return f"Error implementing architectural change: {e}"

def test_architecture_integrity() -> str:
    """Test the integrity of the current architecture."""
    try:
        arch_manager = ArchitectureManager()
        success, test_results = arch_manager.test_architecture()
        
        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(ARCHITECTURE_DIR, f"test_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Count failures
        import_failures = sum(1 for test in test_results["import_tests"].values() if not test["success"])
        syntax_failures = sum(1 for test in test_results["syntax_tests"].values() if not test["success"])
        
        result_msg = f"Architecture test {'passed' if success else 'failed'}. "
        result_msg += f"Import tests: {len(test_results['import_tests'])-import_failures}/{len(test_results['import_tests'])} passed. "
        result_msg += f"Syntax tests: {len(test_results['syntax_tests'])-syntax_failures}/{len(test_results['syntax_tests'])} passed."
        
        log_action("Architecture Test", result_msg)
        
        return f"{result_msg} Full results saved to {results_file}"
        
    except Exception as e:
        logger.error(f"Error testing architecture: {e}")
        return f"Architecture testing failed: {e}"

def rollback_architectural_change(backup_dir: str) -> str:
    """
    Rollback to a previous architecture state.
    
    Args:
        backup_dir: Path to the backup directory
        
    Returns:
        Result message
    """
    try:
        arch_manager = ArchitectureManager()
        success, message = arch_manager.rollback(backup_dir)
        
        if success:
            log_action("Architecture Rollback", f"Rolled back to {backup_dir}")
            return f"Successfully rolled back architectural changes: {message}"
        else:
            return f"Failed to rollback architectural changes: {message}"
        
    except Exception as e:
        logger.error(f"Error rolling back architectural change: {e}")
        return f"Error rolling back architectural change: {e}" 