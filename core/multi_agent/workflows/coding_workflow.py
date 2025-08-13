"""
MinionsAI v3.1 - Coding Workflow
Pre-defined workflow for code development tasks.
"""

from typing import Dict, List, Any
from ..coordinator import WorkflowStep, Workflow
from ..agent_registry import AgentCapability


class CodingWorkflow:
    """Pre-defined coding workflow."""
    
    @staticmethod
    def create_code_development_workflow(requirements: str) -> Workflow:
        """Create a code development workflow."""
        steps = [
            WorkflowStep(
                name="Planning",
                task_template={
                    "description": f"Plan code development for: {requirements}",
                    "required_capabilities": [AgentCapability.TASK_PLANNING.value],
                    "task_type": "decompose_task",
                    "input_data": {"task": requirements}
                }
            ),
            WorkflowStep(
                name="Code Generation",
                task_template={
                    "description": f"Generate code for: {requirements}",
                    "required_capabilities": [AgentCapability.CODE_GENERATION.value],
                    "task_type": "code_generation",
                    "input_data": {"requirements": requirements}
                },
                dependencies=["Planning"]
            ),
            WorkflowStep(
                name="Code Review",
                task_template={
                    "description": f"Review generated code for: {requirements}",
                    "required_capabilities": [AgentCapability.CODE_GENERATION.value],
                    "task_type": "code_review"
                },
                dependencies=["Code Generation"]
            )
        ]
        
        return Workflow(
            name=f"Code Development: {requirements}",
            description=f"Multi-agent code development workflow",
            steps=steps
        )
