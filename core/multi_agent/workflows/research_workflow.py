"""
MinionsAI v3.1 - Research Workflow
Pre-defined workflow for research and analysis tasks.
"""

from typing import Dict, List, Any
from ..coordinator import WorkflowStep, Workflow
from ..agent_registry import AgentCapability


class ResearchWorkflow:
    """Pre-defined research workflow."""
    
    @staticmethod
    def create_comprehensive_research_workflow(topic: str) -> Workflow:
        """Create a comprehensive research workflow."""
        steps = [
            WorkflowStep(
                name="Initial Research",
                task_template={
                    "description": f"Conduct initial research on {topic}",
                    "required_capabilities": [AgentCapability.WEB_SEARCH.value],
                    "task_type": "web_search",
                    "input_data": {"query": topic, "max_results": 10}
                }
            ),
            WorkflowStep(
                name="Analysis",
                task_template={
                    "description": f"Analyze research findings on {topic}",
                    "required_capabilities": [AgentCapability.DATA_ANALYSIS.value],
                    "task_type": "pattern_recognition"
                },
                dependencies=["Initial Research"]
            ),
            WorkflowStep(
                name="Summary",
                task_template={
                    "description": f"Create comprehensive summary of {topic} research",
                    "required_capabilities": [AgentCapability.INFORMATION_SYNTHESIS.value],
                    "task_type": "synthesize"
                },
                dependencies=["Initial Research", "Analysis"]
            )
        ]
        
        return Workflow(
            name=f"Comprehensive Research: {topic}",
            description=f"Multi-agent research workflow for {topic}",
            steps=steps
        )
