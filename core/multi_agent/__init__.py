"""
MinionsAI v3.1 - Multi-Agent System
Core multi-agent system for specialized agent collaboration.
"""

__version__ = "3.1.0"
__author__ = "MinionsAI Team"

# Core multi-agent imports
from .agent_registry import AgentRegistry, AgentCapability, AgentStatus, AgentInfo
from .coordinator import MultiAgentCoordinator, TaskPriority, TaskStatus, Task, Workflow, WorkflowStep
from .communication import AgentCommunication, Message, MessageType

# Agent imports
from .agents import (
    BaseAgent,
    ResearchAgent,
    AnalysisAgent,
    CodeAgent,
    PlanningAgent,
    SummaryAgent
)

# Workflow imports
from .workflows import (
    ResearchWorkflow,
    CodingWorkflow,
    GeneralWorkflow
)

__all__ = [
    # Core system
    "AgentRegistry",
    "AgentCapability",
    "AgentStatus",
    "AgentInfo",
    "MultiAgentCoordinator",
    "TaskPriority",
    "TaskStatus",
    "Task",
    "Workflow",
    "WorkflowStep",
    "AgentCommunication",
    "Message",
    "MessageType",

    # Agents
    "BaseAgent",
    "ResearchAgent",
    "AnalysisAgent",
    "CodeAgent",
    "PlanningAgent",
    "SummaryAgent",

    # Workflows
    "ResearchWorkflow",
    "CodingWorkflow",
    "GeneralWorkflow"
]
