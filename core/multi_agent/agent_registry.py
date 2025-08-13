"""
MinionsAI v3.1 - Agent Registry
Central registry for managing multiple agents and their capabilities.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Type, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent status states."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class AgentCapability(Enum):
    """Agent capability types."""
    WEB_SEARCH = "web_search"
    DATA_ANALYSIS = "data_analysis"
    CODE_GENERATION = "code_generation"
    CODE_EXECUTION = "code_execution"
    FILE_OPERATIONS = "file_operations"
    TASK_PLANNING = "task_planning"
    INFORMATION_SYNTHESIS = "information_synthesis"
    NATURAL_LANGUAGE = "natural_language"
    MATHEMATICAL_COMPUTATION = "mathematical_computation"
    IMAGE_PROCESSING = "image_processing"
    DOCUMENT_PROCESSING = "document_processing"


@dataclass
class AgentInfo:
    """
    Information about a registered agent.
    """
    id: str
    name: str
    agent_type: str
    capabilities: Set[AgentCapability] = field(default_factory=set)
    status: AgentStatus = AgentStatus.INACTIVE
    description: str = ""
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    task_count: int = 0
    success_rate: float = 0.0
    average_response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent info to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "agent_type": self.agent_type,
            "capabilities": [cap.value for cap in self.capabilities],
            "status": self.status.value,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "task_count": self.task_count,
            "success_rate": self.success_rate,
            "average_response_time": self.average_response_time,
            "metadata": self.metadata
        }


class AgentRegistry:
    """
    Central registry for managing multiple agents.
    """
    
    def __init__(self):
        """Initialize the agent registry."""
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_instances: Dict[str, Any] = {}
        self.capability_index: Dict[AgentCapability, Set[str]] = {}
        self.type_index: Dict[str, Set[str]] = {}
        
        # Initialize capability index
        for capability in AgentCapability:
            self.capability_index[capability] = set()
    
    def register_agent(
        self,
        agent_instance: Any,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        agent_type: Optional[str] = None,
        capabilities: Optional[List[AgentCapability]] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register an agent with the registry.
        
        Args:
            agent_instance: The actual agent instance
            agent_id: Unique identifier (generated if not provided)
            name: Human-readable name
            agent_type: Type/class of the agent
            capabilities: List of agent capabilities
            description: Agent description
            metadata: Additional metadata
            
        Returns:
            str: The agent ID
        """
        # Generate ID if not provided
        if agent_id is None:
            agent_id = str(uuid.uuid4())
        
        # Extract information from agent instance if available
        if hasattr(agent_instance, 'agent_id'):
            agent_id = agent_instance.agent_id
        if hasattr(agent_instance, 'name') and name is None:
            name = agent_instance.name
        if hasattr(agent_instance, 'agent_type') and agent_type is None:
            agent_type = agent_instance.agent_type
        if hasattr(agent_instance, 'capabilities') and capabilities is None:
            capabilities = agent_instance.capabilities
        if hasattr(agent_instance, 'description') and not description:
            description = agent_instance.description
        
        # Set defaults
        if name is None:
            name = f"Agent_{agent_id[:8]}"
        if agent_type is None:
            agent_type = type(agent_instance).__name__
        if capabilities is None:
            capabilities = []
        if metadata is None:
            metadata = {}
        
        # Create agent info
        agent_info = AgentInfo(
            id=agent_id,
            name=name,
            agent_type=agent_type,
            capabilities=set(capabilities),
            description=description,
            metadata=metadata
        )
        
        # Store agent
        self.agents[agent_id] = agent_info
        self.agent_instances[agent_id] = agent_instance
        
        # Update indexes
        self._update_capability_index(agent_id, capabilities)
        self._update_type_index(agent_id, agent_type)
        
        logger.info(f"Registered agent: {name} ({agent_id}) with capabilities: {[c.value for c in capabilities]}")
        return agent_id
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the registry.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            bool: True if agent was unregistered successfully
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found in registry")
            return False
        
        agent_info = self.agents[agent_id]
        
        # Remove from indexes
        self._remove_from_capability_index(agent_id, agent_info.capabilities)
        self._remove_from_type_index(agent_id, agent_info.agent_type)
        
        # Remove from registry
        del self.agents[agent_id]
        if agent_id in self.agent_instances:
            del self.agent_instances[agent_id]
        
        logger.info(f"Unregistered agent: {agent_info.name} ({agent_id})")
        return True
    
    def get_agent(self, agent_id: str) -> Optional[Any]:
        """
        Get an agent instance by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent instance or None if not found
        """
        return self.agent_instances.get(agent_id)
    
    def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]:
        """
        Get agent information by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            AgentInfo or None if not found
        """
        return self.agents.get(agent_id)
    
    def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """
        Update an agent's status.
        
        Args:
            agent_id: Agent identifier
            status: New status
            
        Returns:
            bool: True if updated successfully
        """
        if agent_id not in self.agents:
            return False
        
        self.agents[agent_id].status = status
        self.agents[agent_id].last_active = datetime.now()
        
        logger.debug(f"Updated agent {agent_id} status to {status.value}")
        return True
    
    def find_agents_by_capability(self, capability: AgentCapability) -> List[str]:
        """
        Find agents that have a specific capability.
        
        Args:
            capability: Required capability
            
        Returns:
            List of agent IDs
        """
        return list(self.capability_index.get(capability, set()))
    
    def find_agents_by_type(self, agent_type: str) -> List[str]:
        """
        Find agents of a specific type.
        
        Args:
            agent_type: Agent type to search for
            
        Returns:
            List of agent IDs
        """
        return list(self.type_index.get(agent_type, set()))
    
    def find_available_agents(
        self,
        capabilities: Optional[List[AgentCapability]] = None,
        agent_type: Optional[str] = None,
        exclude_busy: bool = True
    ) -> List[str]:
        """
        Find available agents matching criteria.
        
        Args:
            capabilities: Required capabilities
            agent_type: Required agent type
            exclude_busy: Whether to exclude busy agents
            
        Returns:
            List of agent IDs
        """
        candidate_agents = set(self.agents.keys())
        
        # Filter by capabilities
        if capabilities:
            for capability in capabilities:
                capable_agents = self.capability_index.get(capability, set())
                candidate_agents &= capable_agents
        
        # Filter by type
        if agent_type:
            type_agents = self.type_index.get(agent_type, set())
            candidate_agents &= type_agents
        
        # Filter by status
        if exclude_busy:
            available_agents = []
            for agent_id in candidate_agents:
                agent_info = self.agents[agent_id]
                if agent_info.status in [AgentStatus.ACTIVE, AgentStatus.INACTIVE]:
                    available_agents.append(agent_id)
            return available_agents
        
        return list(candidate_agents)
    
    def get_best_agent_for_task(
        self,
        capabilities: List[AgentCapability],
        agent_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Find the best agent for a specific task based on capabilities and performance.
        
        Args:
            capabilities: Required capabilities
            agent_type: Preferred agent type
            
        Returns:
            Agent ID of the best match or None
        """
        available_agents = self.find_available_agents(capabilities, agent_type)
        
        if not available_agents:
            return None
        
        # Score agents based on success rate and response time
        best_agent = None
        best_score = -1
        
        for agent_id in available_agents:
            agent_info = self.agents[agent_id]
            
            # Calculate score (higher is better)
            score = agent_info.success_rate
            
            # Bonus for having more relevant capabilities
            capability_match = len(set(capabilities) & agent_info.capabilities) / len(capabilities)
            score += capability_match * 0.2
            
            # Penalty for slow response time (if we have data)
            if agent_info.average_response_time > 0:
                time_penalty = min(agent_info.average_response_time / 10.0, 0.3)
                score -= time_penalty
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    def get_all_agents(self) -> Dict[str, AgentInfo]:
        """Get all registered agents."""
        return self.agents.copy()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary containing registry statistics
        """
        total_agents = len(self.agents)
        status_counts = {}
        type_counts = {}
        capability_counts = {}
        
        for agent_info in self.agents.values():
            # Count by status
            status = agent_info.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by type
            agent_type = agent_info.agent_type
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1
            
            # Count capabilities
            for capability in agent_info.capabilities:
                cap_name = capability.value
                capability_counts[cap_name] = capability_counts.get(cap_name, 0) + 1
        
        return {
            "total_agents": total_agents,
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "capability_distribution": capability_counts,
            "active_agents": len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE])
        }
    
    def _update_capability_index(self, agent_id: str, capabilities: List[AgentCapability]) -> None:
        """Update the capability index for an agent."""
        for capability in capabilities:
            self.capability_index[capability].add(agent_id)
    
    def _remove_from_capability_index(self, agent_id: str, capabilities: Set[AgentCapability]) -> None:
        """Remove agent from capability index."""
        for capability in capabilities:
            self.capability_index[capability].discard(agent_id)
    
    def _update_type_index(self, agent_id: str, agent_type: str) -> None:
        """Update the type index for an agent."""
        if agent_type not in self.type_index:
            self.type_index[agent_type] = set()
        self.type_index[agent_type].add(agent_id)
    
    def _remove_from_type_index(self, agent_id: str, agent_type: str) -> None:
        """Remove agent from type index."""
        if agent_type in self.type_index:
            self.type_index[agent_type].discard(agent_id)
