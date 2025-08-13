"""
MinionsAI v3.1 - General Workflow
Pre-defined workflow for general problem-solving tasks.
"""

from typing import Dict, List, Any
from ..coordinator import WorkflowStep, Workflow
from ..agent_registry import AgentCapability


class GeneralWorkflow:
    """Pre-defined general workflow."""
    
    @staticmethod
    def create_problem_solving_workflow(problem: str) -> Workflow:
        """Create a general problem-solving workflow."""
        steps = [
            WorkflowStep(
                name="Problem Analysis",
                task_template={
                    "description": f"Analyze the problem: {problem}",
                    "required_capabilities": [AgentCapability.TASK_PLANNING.value],
                    "task_type": "create_plan",
                    "input_data": {"objective": problem}
                }
            ),
            WorkflowStep(
                name="Information Gathering",
                task_template={
                    "description": f"Gather information about: {problem}",
                    "required_capabilities": [AgentCapability.WEB_SEARCH.value],
                    "task_type": "comprehensive_research",
                    "input_data": {"topic": problem}
                },
                dependencies=["Problem Analysis"],
                parallel_execution=True
            ),
            WorkflowStep(
                name="Solution Synthesis",
                task_template={
                    "description": f"Synthesize solution for: {problem}",
                    "required_capabilities": [AgentCapability.INFORMATION_SYNTHESIS.value],
                    "task_type": "create_report",
                    "input_data": {"topic": f"Solution for {problem}"}
                },
                dependencies=["Problem Analysis", "Information Gathering"]
            )
        ]
        
        return Workflow(
            name=f"Problem Solving: {problem}",
            description=f"Multi-agent problem-solving workflow",
            steps=steps
        )
