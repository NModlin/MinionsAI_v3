"""
MinionsAI v3.1 - Core Module
Core functionality for the MinionsAI agent system.
"""

__version__ = "3.1.0"
__author__ = "MinionsAI Team"

# Core module imports
from .agent_manager import AgentManager
from .chat_handler import ChatHandler
from .system_monitor import SystemMonitor

__all__ = [
    "AgentManager",
    "ChatHandler", 
    "SystemMonitor"
]
