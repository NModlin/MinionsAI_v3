"""
MinionsAI v3.1 - Agent Communication System
Handles message passing and communication between agents.
"""

import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages that can be sent between agents."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    INFORMATION_SHARE = "information_share"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    WORKFLOW_CONTROL = "workflow_control"


class Priority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Message:
    """
    Message structure for inter-agent communication.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: MessageType = MessageType.INFORMATION_SHARE
    priority: Priority = Priority.NORMAL
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False
    response_timeout: Optional[float] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "requires_response": self.requires_response,
            "response_timeout": self.response_timeout,
            "correlation_id": self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            sender_id=data.get("sender_id", ""),
            recipient_id=data.get("recipient_id", ""),
            message_type=MessageType(data.get("message_type", "information_share")),
            priority=Priority(data.get("priority", 2)),
            content=data.get("content", {}),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            requires_response=data.get("requires_response", False),
            response_timeout=data.get("response_timeout"),
            correlation_id=data.get("correlation_id")
        )


class AgentCommunication:
    """
    Central communication hub for agent message passing.
    """
    
    def __init__(self):
        """Initialize the communication system."""
        self.message_queue: Dict[str, List[Message]] = {}
        self.message_handlers: Dict[str, Dict[MessageType, Callable]] = {}
        self.message_history: List[Message] = []
        self.active_agents: Dict[str, bool] = {}
        self.pending_responses: Dict[str, Message] = {}
        self._lock = asyncio.Lock()
        
    def register_agent(self, agent_id: str) -> None:
        """
        Register an agent with the communication system.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        if agent_id not in self.message_queue:
            self.message_queue[agent_id] = []
            self.message_handlers[agent_id] = {}
            self.active_agents[agent_id] = True
            logger.info(f"Agent {agent_id} registered with communication system")
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the communication system.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        if agent_id in self.active_agents:
            self.active_agents[agent_id] = False
            logger.info(f"Agent {agent_id} unregistered from communication system")
    
    def register_message_handler(
        self, 
        agent_id: str, 
        message_type: MessageType, 
        handler: Callable[[Message], Any]
    ) -> None:
        """
        Register a message handler for an agent.
        
        Args:
            agent_id: Agent identifier
            message_type: Type of message to handle
            handler: Function to handle the message
        """
        if agent_id not in self.message_handlers:
            self.message_handlers[agent_id] = {}
        
        self.message_handlers[agent_id][message_type] = handler
        logger.debug(f"Registered handler for {message_type} on agent {agent_id}")
    
    async def send_message(self, message: Message) -> bool:
        """
        Send a message to an agent.
        
        Args:
            message: Message to send
            
        Returns:
            bool: True if message was sent successfully
        """
        async with self._lock:
            try:
                # Check if recipient is active
                if message.recipient_id not in self.active_agents:
                    logger.error(f"Recipient {message.recipient_id} not registered")
                    return False
                
                if not self.active_agents[message.recipient_id]:
                    logger.error(f"Recipient {message.recipient_id} is not active")
                    return False
                
                # Add to recipient's queue
                self.message_queue[message.recipient_id].append(message)
                
                # Add to history
                self.message_history.append(message)
                
                # If response is required, track it
                if message.requires_response:
                    self.pending_responses[message.id] = message
                
                logger.debug(f"Message {message.id} sent from {message.sender_id} to {message.recipient_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                return False
    
    async def receive_messages(self, agent_id: str) -> List[Message]:
        """
        Retrieve messages for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of messages for the agent
        """
        async with self._lock:
            if agent_id not in self.message_queue:
                return []
            
            messages = self.message_queue[agent_id].copy()
            self.message_queue[agent_id].clear()
            
            return messages
    
    async def broadcast_message(self, message: Message, exclude_sender: bool = True) -> int:
        """
        Broadcast a message to all active agents.
        
        Args:
            message: Message to broadcast
            exclude_sender: Whether to exclude the sender from broadcast
            
        Returns:
            Number of agents the message was sent to
        """
        sent_count = 0
        
        for agent_id in self.active_agents:
            if not self.active_agents[agent_id]:
                continue
                
            if exclude_sender and agent_id == message.sender_id:
                continue
            
            # Create a copy of the message for each recipient
            broadcast_message = Message(
                sender_id=message.sender_id,
                recipient_id=agent_id,
                message_type=message.message_type,
                priority=message.priority,
                content=message.content.copy(),
                metadata=message.metadata.copy(),
                requires_response=message.requires_response,
                response_timeout=message.response_timeout,
                correlation_id=message.correlation_id
            )
            
            if await self.send_message(broadcast_message):
                sent_count += 1
        
        return sent_count
    
    async def process_messages(self, agent_id: str) -> None:
        """
        Process pending messages for an agent using registered handlers.
        
        Args:
            agent_id: Agent identifier
        """
        messages = await self.receive_messages(agent_id)
        
        for message in messages:
            try:
                # Check if there's a handler for this message type
                if (agent_id in self.message_handlers and 
                    message.message_type in self.message_handlers[agent_id]):
                    
                    handler = self.message_handlers[agent_id][message.message_type]
                    await handler(message)
                    
                else:
                    logger.warning(f"No handler for {message.message_type} on agent {agent_id}")
                    
            except Exception as e:
                logger.error(f"Error processing message {message.id}: {e}")
    
    def get_message_history(
        self, 
        agent_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        limit: int = 100
    ) -> List[Message]:
        """
        Get message history with optional filtering.
        
        Args:
            agent_id: Filter by agent (sender or recipient)
            message_type: Filter by message type
            limit: Maximum number of messages to return
            
        Returns:
            List of messages matching the criteria
        """
        filtered_messages = self.message_history
        
        if agent_id:
            filtered_messages = [
                msg for msg in filtered_messages 
                if msg.sender_id == agent_id or msg.recipient_id == agent_id
            ]
        
        if message_type:
            filtered_messages = [
                msg for msg in filtered_messages 
                if msg.message_type == message_type
            ]
        
        # Sort by timestamp (newest first) and limit
        filtered_messages.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_messages[:limit]
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """
        Get communication system statistics.
        
        Returns:
            Dictionary containing system statistics
        """
        total_messages = len(self.message_history)
        active_agent_count = sum(1 for active in self.active_agents.values() if active)
        pending_responses = len(self.pending_responses)
        
        message_type_counts = {}
        for msg in self.message_history:
            msg_type = msg.message_type.value
            message_type_counts[msg_type] = message_type_counts.get(msg_type, 0) + 1
        
        return {
            "total_messages": total_messages,
            "active_agents": active_agent_count,
            "pending_responses": pending_responses,
            "message_type_distribution": message_type_counts,
            "registered_agents": list(self.active_agents.keys())
        }
    
    def clear_history(self) -> None:
        """Clear message history."""
        self.message_history.clear()
        self.pending_responses.clear()
        logger.info("Message history cleared")
