"""
MinionsAI v3.1 - Code Agent
Specialized agent for code generation, debugging, and technical problem solving.
"""

import ast
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from .base_agent import BaseAgent
from ..agent_registry import AgentCapability

logger = logging.getLogger(__name__)


class CodeAgent(BaseAgent):
    """
    Specialized agent for code-related tasks.
    """
    
    def __init__(self, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        """Initialize the Code Agent."""
        super().__init__(
            name="Code Agent",
            description="Specialized in code generation, debugging, and technical problem solving",
            capabilities=[
                AgentCapability.CODE_GENERATION,
                AgentCapability.CODE_EXECUTION,
                AgentCapability.NATURAL_LANGUAGE,
                AgentCapability.FILE_OPERATIONS
            ],
            model_name=model_name,
            base_url=base_url
        )
        
        # Code-specific configuration
        self.supported_languages = ["python", "javascript", "bash", "sql"]
        self.execution_timeout = 30
        self.max_code_length = 10000
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Code Agent."""
        return """You are a Code Agent, specialized in code generation, debugging, and technical problem solving.

Your capabilities include:
- Writing clean, efficient, and well-documented code
- Debugging and fixing code issues
- Code review and optimization
- Explaining code functionality
- Converting between programming languages
- Creating unit tests and documentation
- Solving algorithmic problems

Supported languages: Python, JavaScript, Bash, SQL

When working with code:
1. Write clean, readable, and well-commented code
2. Follow best practices and coding standards
3. Include error handling where appropriate
4. Provide clear explanations of the code logic
5. Test code functionality when possible
6. Consider security and performance implications
7. Suggest improvements and optimizations

Always prioritize code quality, security, and maintainability."""
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a code-related task.
        
        Args:
            task_data: Task data containing code parameters
            
        Returns:
            Dict containing code results
        """
        try:
            task_type = task_data.get("task_type", "code_generation")
            
            logger.info(f"Code Agent executing {task_type}")
            
            if task_type == "code_generation":
                return await self._generate_code(task_data)
            elif task_type == "code_debugging":
                return await self._debug_code(task_data)
            elif task_type == "code_execution":
                return await self._execute_code(task_data)
            elif task_type == "code_review":
                return await self._review_code(task_data)
            elif task_type == "code_explanation":
                return await self._explain_code(task_data)
            elif task_type == "code_optimization":
                return await self._optimize_code(task_data)
            elif task_type == "test_generation":
                return await self._generate_tests(task_data)
            else:
                return await self._general_code_task(task_data)
                
        except Exception as e:
            logger.error(f"Error in Code Agent task execution: {e}")
            return {
                "success": False,
                "error": str(e),
                "code": "",
                "explanation": ""
            }
    
    async def _generate_code(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code based on requirements."""
        requirements = task_data.get("requirements", "")
        language = task_data.get("language", "python").lower()
        style = task_data.get("style", "standard")
        
        if not requirements:
            return {
                "success": False,
                "error": "No requirements provided for code generation"
            }
        
        if language not in self.supported_languages:
            return {
                "success": False,
                "error": f"Language '{language}' not supported. Supported: {self.supported_languages}"
            }
        
        generation_prompt = f"""
        Generate {language} code based on the following requirements:
        
        Requirements: {requirements}
        Style: {style}
        
        Please provide:
        1. Clean, well-commented code
        2. Error handling where appropriate
        3. Clear variable and function names
        4. Brief explanation of the approach
        5. Usage examples if applicable
        
        Format your response as:
        ```{language}
        [code here]
        ```
        
        Explanation: [explanation here]
        """
        
        response = await self.generate_response(generation_prompt)
        
        # Extract code and explanation
        code, explanation = self._parse_code_response(response, language)
        
        # Validate code syntax if possible
        validation_result = self._validate_code_syntax(code, language)
        
        return {
            "success": True,
            "task_type": "code_generation",
            "language": language,
            "code": code,
            "explanation": explanation,
            "validation": validation_result,
            "requirements": requirements,
            "generation_timestamp": datetime.now().isoformat()
        }
    
    async def _debug_code(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Debug problematic code."""
        code = task_data.get("code", "")
        error_message = task_data.get("error_message", "")
        language = task_data.get("language", "python").lower()
        
        if not code:
            return {
                "success": False,
                "error": "No code provided for debugging"
            }
        
        debug_prompt = f"""
        Debug the following {language} code:
        
        Code:
        ```{language}
        {code}
        ```
        
        Error Message: {error_message}
        
        Please provide:
        1. Identification of the issue(s)
        2. Corrected code
        3. Explanation of what was wrong
        4. Prevention tips for similar issues
        
        Format your response as:
        Issues Found: [list of issues]
        
        Corrected Code:
        ```{language}
        [corrected code here]
        ```
        
        Explanation: [detailed explanation]
        """
        
        response = await self.generate_response(debug_prompt)
        
        # Extract corrected code
        corrected_code, explanation = self._parse_code_response(response, language)
        
        # Validate corrected code
        validation_result = self._validate_code_syntax(corrected_code, language)
        
        return {
            "success": True,
            "task_type": "code_debugging",
            "language": language,
            "original_code": code,
            "corrected_code": corrected_code,
            "debug_explanation": explanation,
            "validation": validation_result,
            "error_message": error_message,
            "debug_timestamp": datetime.now().isoformat()
        }
    
    async def _execute_code(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code safely."""
        code = task_data.get("code", "")
        language = task_data.get("language", "python").lower()
        inputs = task_data.get("inputs", [])
        
        if not code:
            return {
                "success": False,
                "error": "No code provided for execution"
            }
        
        if language not in ["python", "bash"]:  # Limit execution to safe languages
            return {
                "success": False,
                "error": f"Code execution not supported for {language}"
            }
        
        try:
            # Execute code in a safe environment
            execution_result = await self._safe_execute_code(code, language, inputs)
            
            return {
                "success": True,
                "task_type": "code_execution",
                "language": language,
                "code": code,
                "execution_result": execution_result,
                "execution_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Code execution failed: {str(e)}",
                "code": code,
                "language": language
            }
    
    async def _review_code(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Review code for quality and best practices."""
        code = task_data.get("code", "")
        language = task_data.get("language", "python").lower()
        review_criteria = task_data.get("criteria", ["readability", "performance", "security"])
        
        if not code:
            return {
                "success": False,
                "error": "No code provided for review"
            }
        
        review_prompt = f"""
        Review the following {language} code for quality and best practices:
        
        Code:
        ```{language}
        {code}
        ```
        
        Review Criteria: {', '.join(review_criteria)}
        
        Please provide:
        1. Overall code quality assessment (1-10)
        2. Strengths of the code
        3. Areas for improvement
        4. Specific recommendations
        5. Security considerations
        6. Performance implications
        7. Refactored version (if significant improvements possible)
        
        Format your response clearly with sections for each aspect.
        """
        
        review = await self.generate_response(review_prompt)
        
        # Perform basic static analysis
        static_analysis = self._perform_static_analysis(code, language)
        
        return {
            "success": True,
            "task_type": "code_review",
            "language": language,
            "code": code,
            "review": review,
            "static_analysis": static_analysis,
            "review_criteria": review_criteria,
            "review_timestamp": datetime.now().isoformat()
        }
    
    async def _explain_code(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Explain how code works."""
        code = task_data.get("code", "")
        language = task_data.get("language", "python").lower()
        detail_level = task_data.get("detail_level", "medium")
        
        if not code:
            return {
                "success": False,
                "error": "No code provided for explanation"
            }
        
        explanation_prompt = f"""
        Explain the following {language} code in {detail_level} detail:
        
        Code:
        ```{language}
        {code}
        ```
        
        Please provide:
        1. Overall purpose and functionality
        2. Step-by-step breakdown of the logic
        3. Explanation of key concepts used
        4. Input/output behavior
        5. Any notable algorithms or patterns
        6. Potential use cases
        
        Adjust the level of detail based on: {detail_level}
        """
        
        explanation = await self.generate_response(explanation_prompt)
        
        # Extract code structure information
        structure_info = self._analyze_code_structure(code, language)
        
        return {
            "success": True,
            "task_type": "code_explanation",
            "language": language,
            "code": code,
            "explanation": explanation,
            "structure_info": structure_info,
            "detail_level": detail_level,
            "explanation_timestamp": datetime.now().isoformat()
        }
    
    async def _optimize_code(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize code for performance or readability."""
        code = task_data.get("code", "")
        language = task_data.get("language", "python").lower()
        optimization_goals = task_data.get("goals", ["performance", "readability"])
        
        if not code:
            return {
                "success": False,
                "error": "No code provided for optimization"
            }
        
        optimization_prompt = f"""
        Optimize the following {language} code for: {', '.join(optimization_goals)}
        
        Original Code:
        ```{language}
        {code}
        ```
        
        Please provide:
        1. Optimized version of the code
        2. Explanation of optimizations made
        3. Performance improvements expected
        4. Trade-offs considered
        5. Alternative approaches considered
        
        Format your response as:
        Optimized Code:
        ```{language}
        [optimized code here]
        ```
        
        Optimizations Made: [detailed explanation]
        """
        
        response = await self.generate_response(optimization_prompt)
        
        # Extract optimized code
        optimized_code, optimization_explanation = self._parse_code_response(response, language)
        
        # Validate optimized code
        validation_result = self._validate_code_syntax(optimized_code, language)
        
        return {
            "success": True,
            "task_type": "code_optimization",
            "language": language,
            "original_code": code,
            "optimized_code": optimized_code,
            "optimization_explanation": optimization_explanation,
            "optimization_goals": optimization_goals,
            "validation": validation_result,
            "optimization_timestamp": datetime.now().isoformat()
        }
    
    async def _generate_tests(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unit tests for code."""
        code = task_data.get("code", "")
        language = task_data.get("language", "python").lower()
        test_framework = task_data.get("test_framework", "unittest")
        
        if not code:
            return {
                "success": False,
                "error": "No code provided for test generation"
            }
        
        test_prompt = f"""
        Generate comprehensive unit tests for the following {language} code using {test_framework}:
        
        Code to Test:
        ```{language}
        {code}
        ```
        
        Please provide:
        1. Complete test suite with multiple test cases
        2. Tests for normal operation
        3. Tests for edge cases
        4. Tests for error conditions
        5. Clear test descriptions
        6. Setup and teardown if needed
        
        Format your response as:
        ```{language}
        [test code here]
        ```
        
        Test Coverage: [explanation of what is tested]
        """
        
        response = await self.generate_response(test_prompt)
        
        # Extract test code
        test_code, test_explanation = self._parse_code_response(response, language)
        
        # Validate test code
        validation_result = self._validate_code_syntax(test_code, language)
        
        return {
            "success": True,
            "task_type": "test_generation",
            "language": language,
            "original_code": code,
            "test_code": test_code,
            "test_explanation": test_explanation,
            "test_framework": test_framework,
            "validation": validation_result,
            "generation_timestamp": datetime.now().isoformat()
        }
    
    async def _general_code_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general code-related tasks."""
        description = task_data.get("description", "")
        language = task_data.get("language", "python").lower()
        
        if not description:
            return {
                "success": False,
                "error": "No task description provided"
            }
        
        general_prompt = f"""
        Handle the following {language} coding task:
        
        Task Description: {description}
        
        Please provide appropriate code solution and explanation based on the task requirements.
        """
        
        response = await self.generate_response(general_prompt)
        
        return {
            "success": True,
            "task_type": "general_code_task",
            "language": language,
            "description": description,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    # Helper methods
    def _parse_code_response(self, response: str, language: str) -> tuple[str, str]:
        """Parse code and explanation from LLM response."""
        import re
        
        # Look for code blocks
        code_pattern = rf"```{language}(.*?)```"
        code_matches = re.findall(code_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if code_matches:
            code = code_matches[0].strip()
            # Remove code block from response to get explanation
            explanation = re.sub(code_pattern, "", response, flags=re.DOTALL | re.IGNORECASE).strip()
        else:
            # Fallback: look for any code block
            general_pattern = r"```.*?(.*?)```"
            general_matches = re.findall(general_pattern, response, re.DOTALL)
            if general_matches:
                code = general_matches[0].strip()
                explanation = re.sub(general_pattern, "", response, flags=re.DOTALL).strip()
            else:
                code = ""
                explanation = response
        
        return code, explanation
    
    def _validate_code_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """Validate code syntax."""
        if not code:
            return {"valid": False, "error": "No code to validate"}
        
        try:
            if language == "python":
                ast.parse(code)
                return {"valid": True, "message": "Python syntax is valid"}
            else:
                # For other languages, we can't easily validate syntax
                return {"valid": True, "message": f"Syntax validation not available for {language}"}
        
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Syntax error: {str(e)}",
                "line": getattr(e, 'lineno', None),
                "column": getattr(e, 'offset', None)
            }
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    async def _safe_execute_code(self, code: str, language: str, inputs: List[str]) -> Dict[str, Any]:
        """Execute code safely with timeout and restrictions."""
        if language == "python":
            return await self._execute_python_code(code, inputs)
        elif language == "bash":
            return await self._execute_bash_code(code, inputs)
        else:
            return {"error": f"Execution not supported for {language}"}
    
    async def _execute_python_code(self, code: str, inputs: List[str]) -> Dict[str, Any]:
        """Execute Python code safely."""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Prepare input
            input_str = '\n'.join(inputs) if inputs else ""
            
            # Execute with timeout
            result = subprocess.run(
                ["python", temp_file],
                input=input_str,
                capture_output=True,
                text=True,
                timeout=self.execution_timeout
            )
            
            # Clean up
            os.unlink(temp_file)
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "execution_time": "< 30s",  # We don't measure exact time
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "Code execution timed out", "success": False}
        except Exception as e:
            return {"error": f"Execution error: {str(e)}", "success": False}
    
    async def _execute_bash_code(self, code: str, inputs: List[str]) -> Dict[str, Any]:
        """Execute Bash code safely."""
        try:
            # Basic safety check - reject dangerous commands
            dangerous_patterns = ['rm -rf', 'sudo', 'chmod', 'chown', '>', '>>', 'curl', 'wget']
            if any(pattern in code.lower() for pattern in dangerous_patterns):
                return {"error": "Code contains potentially dangerous commands", "success": False}
            
            # Execute with timeout
            result = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.execution_timeout
            )
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "execution_time": "< 30s",
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "Code execution timed out", "success": False}
        except Exception as e:
            return {"error": f"Execution error: {str(e)}", "success": False}
    
    def _perform_static_analysis(self, code: str, language: str) -> Dict[str, Any]:
        """Perform basic static analysis on code."""
        analysis = {
            "line_count": len(code.split('\n')),
            "character_count": len(code),
            "estimated_complexity": "low"  # Simplified
        }
        
        if language == "python":
            try:
                tree = ast.parse(code)
                analysis["functions"] = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
                analysis["classes"] = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
                analysis["imports"] = len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))])
            except:
                analysis["parse_error"] = True
        
        return analysis
    
    def _analyze_code_structure(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code structure and extract key information."""
        structure = {
            "total_lines": len(code.split('\n')),
            "non_empty_lines": len([line for line in code.split('\n') if line.strip()]),
            "comment_lines": len([line for line in code.split('\n') if line.strip().startswith('#')])
        }
        
        if language == "python":
            try:
                tree = ast.parse(code)
                structure["functions"] = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                structure["classes"] = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            except:
                structure["parse_error"] = True
        
        return structure
    
    async def _initialize_agent(self) -> None:
        """Code Agent specific initialization."""
        self.add_to_memory("code_history", [])
        self.add_to_memory("execution_history", [])
        logger.info("Code Agent initialized with coding capabilities")
    
    async def _cleanup_agent(self) -> None:
        """Code Agent specific cleanup."""
        logger.info("Code Agent cleanup completed")
