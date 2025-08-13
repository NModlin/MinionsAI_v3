"""
MinionsAI v3.1 - Multi-Agent Workflows
Pre-defined collaboration patterns and workflows for multi-agent tasks.
"""

from .research_workflow import ResearchWorkflow
from .coding_workflow import CodingWorkflow
from .general_workflow import GeneralWorkflow

__all__ = [
    "ResearchWorkflow",
    "CodingWorkflow", 
    "GeneralWorkflow"
]
