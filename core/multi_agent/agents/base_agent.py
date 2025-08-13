"""
MinionsAI v3.1 - Base Agent
Abstract base class for all specialized agents in the multi-agent system.
"""

import uuid
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import logging

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from ..communication import AgentCommunication, Message, MessageType, Priority
from ..agent_registry import AgentCapability, AgentStatus

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        capabilities: List[AgentCapability],
        model_name: str = "llama3:8b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name
            description: Agent description
            capabilities: List of agent capabilities
            model_name: LLM model name
            base_url: Ollama base URL
        """
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.capabilities = set(capabilities)
        self.agent_type = self.__class__.__name__
        self.status = AgentStatus.INACTIVE
        
        # LLM configuration
        self.model_name = model_name
        self.base_url = base_url
        self.llm: Optional[OllamaLLM] = None
        
        # Communication
        self.communication: Optional[AgentCommunication] = None
        self.message_handlers: Dict[MessageType, callable] = {}
        
        # Task tracking
        self.current_task_id: Optional[str] = None
        self.task_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_response_time": 0.0,
            "total_response_time": 0.0
        }
        
        # Agent state
        self.memory: Dict[str, Any] = {}
        self.context: List[Dict[str, Any]] = []
        self.is_initialized = False
        
        # Register default message handlers
        self._register_default_handlers()
    
    async def initialize(self, communication: AgentCommunication) -> bool:
        """
        Initialize the agent.
        
        Args:
            communication: Communication system instance
            
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize LLM
            self.llm = OllamaLLM(model=self.model_name, base_url=self.base_url)
            
            # Set up communication
            self.communication = communication
            self.communication.register_agent(self.agent_id)
            
            # Register message handlers
            for msg_type, handler in self.message_handlers.items():
                self.communication.register_message_handler(self.agent_id, msg_type, handler)
            
            # Perform agent-specific initialization
            await self._initialize_agent()
            
            self.status = AgentStatus.ACTIVE
            self.is_initialized = True
            
            logger.info(f"Agent {self.name} ({self.agent_id}) initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing agent {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the agent."""
        try:
            self.status = AgentStatus.SHUTTING_DOWN
            
            # Perform agent-specific cleanup
            await self._cleanup_agent()
            
            # Unregister from communication
            if self.communication:
                self.communication.unregister_agent(self.agent_id)
            
            self.status = AgentStatus.INACTIVE
            self.is_initialized = False
            
            logger.info(f"Agent {self.name} ({self.agent_id}) shutdown successfully")
            
        except Exception as e:
            logger.error(f"Error shutting down agent {self.name}: {e}")
    
    async def process_messages(self) -> None:
        """Process pending messages."""
        if not self.communication:
            return
        
        try:
            await self.communication.process_messages(self.agent_id)
        except Exception as e:
            logger.error(f"Error processing messages for agent {self.name}: {e}")
    
    async def send_message(self, message: Message) -> bool:
        """
        Send a message to another agent.
        
        Args:
            message: Message to send
            
        Returns:
            bool: True if message sent successfully
        """
        if not self.communication:
            logger.error(f"Agent {self.name} not connected to communication system")
            return False
        
        message.sender_id = self.agent_id
        return await self.communication.send_message(message)
    
    async def broadcast_message(self, message: Message) -> int:
        """
        Broadcast a message to all agents.
        
        Args:
            message: Message to broadcast
            
        Returns:
            int: Number of agents message was sent to
        """
        if not self.communication:
            logger.error(f"Agent {self.name} not connected to communication system")
            return 0
        
        message.sender_id = self.agent_id
        return await self.communication.broadcast_message(message)
    
    @abstractmethod
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.
        
        Args:
            task_data: Task data and parameters
            
        Returns:
            Dict containing task results
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        
        Returns:
            str: System prompt
        """
        pass
    
    async def generate_response(self, prompt: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a response using the agent's LLM.
        
        Args:
            prompt: Input prompt
            context: Optional conversation context
            
        Returns:
            str: Generated response
        """
        if not self.llm:
            raise RuntimeError("Agent LLM not initialized")
        
        try:
            # Prepare messages
            messages = [SystemMessage(content=self.get_system_prompt())]
            
            # Add context if provided
            if context:
                for ctx in context:
                    if ctx.get("role") == "user":
                        messages.append(HumanMessage(content=ctx["content"]))
                    elif ctx.get("role") == "assistant":
                        messages.append(AIMessage(content=ctx["content"]))
            
            # Add current prompt
            messages.append(HumanMessage(content=prompt))
            
            # Generate response
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Error generating response for agent {self.name}: {e}")
            return f"Error generating response: {str(e)}"
    
    def add_to_memory(self, key: str, value: Any) -> None:
        """Add information to agent memory."""
        self.memory[key] = value
    
    def get_from_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve information from agent memory."""
        return self.memory.get(key, default)
    
    def add_to_context(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation context."""
        self.context.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        
        # Keep context size manageable
        if len(self.context) > 50:
            self.context = self.context[-50:]
    
    def clear_context(self) -> None:
        """Clear the conversation context."""
        self.context.clear()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Calculate success rate
        total_tasks = metrics["tasks_completed"] + metrics["tasks_failed"]
        if total_tasks > 0:
            metrics["success_rate"] = metrics["tasks_completed"] / total_tasks
        else:
            metrics["success_rate"] = 0.0
        
        return metrics
    
    def update_performance_metrics(self, task_successful: bool, response_time: float) -> None:
        """Update performance metrics after task completion."""
        if task_successful:
            self.performance_metrics["tasks_completed"] += 1
        else:
            self.performance_metrics["tasks_failed"] += 1
        
        # Update average response time
        total_time = self.performance_metrics["total_response_time"] + response_time
        total_tasks = self.performance_metrics["tasks_completed"] + self.performance_metrics["tasks_failed"]
        
        self.performance_metrics["total_response_time"] = total_time
        self.performance_metrics["average_response_time"] = total_time / total_tasks if total_tasks > 0 else 0.0
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information."""
        return {
            "id": self.agent_id,
            "name": self.name,
            "type": self.agent_type,
            "description": self.description,
            "capabilities": [cap.value for cap in self.capabilities],
            "status": self.status.value,
            "current_task": self.current_task_id,
            "performance": self.get_performance_metrics(),
            "memory_size": len(self.memory),
            "context_size": len(self.context),
            "is_initialized": self.is_initialized
        }
    
    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.message_handlers[MessageType.TASK_REQUEST] = self._handle_task_request
        self.message_handlers[MessageType.STATUS_UPDATE] = self._handle_status_update
        self.message_handlers[MessageType.COORDINATION] = self._handle_coordination
    
    async def _handle_task_request(self, message: Message) -> None:
        """Handle task request messages."""
        try:
            self.status = AgentStatus.BUSY
            self.current_task_id = message.content.get("task_id")
            
            start_time = datetime.now()
            
            # Execute the task
            result = await self.execute_task(message.content)
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Update performance metrics
            task_successful = result.get("success", False)
            self.update_performance_metrics(task_successful, response_time)
            
            # Send response
            response_message = Message(
                recipient_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    "task_id": self.current_task_id,
                    "result": result,
                    "agent_id": self.agent_id,
                    "response_time": response_time
                },
                correlation_id=message.id
            )
            
            await self.send_message(response_message)
            
            self.current_task_id = None
            self.status = AgentStatus.ACTIVE
            
        except Exception as e:
            logger.error(f"Error handling task request in agent {self.name}: {e}")
            
            # Send error response
            error_message = Message(
                recipient_id=message.sender_id,
                message_type=MessageType.ERROR,
                content={
                    "task_id": self.current_task_id,
                    "error": str(e),
                    "agent_id": self.agent_id
                },
                correlation_id=message.id
            )
            
            await self.send_message(error_message)
            
            self.current_task_id = None
            self.status = AgentStatus.ACTIVE
    
    async def _handle_status_update(self, message: Message) -> None:
        """Handle status update messages."""
        # Default implementation - can be overridden by subclasses
        logger.debug(f"Agent {self.name} received status update: {message.content}")
    
    async def _handle_coordination(self, message: Message) -> None:
        """Handle coordination messages."""
        # Default implementation - can be overridden by subclasses
        logger.debug(f"Agent {self.name} received coordination message: {message.content}")
    
    async def _initialize_agent(self) -> None:
        """Agent-specific initialization. Override in subclasses."""
        pass
    
    async def _cleanup_agent(self) -> None:
        """Agent-specific cleanup. Override in subclasses."""
        pass
