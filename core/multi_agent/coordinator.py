"""
MinionsAI v3.1 - Multi-Agent Coordinator
Central coordination system for managing multi-agent workflows and task distribution.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from .communication import AgentCommunication, Message, MessageType, Priority
from .agent_registry import AgentRegistry, AgentCapability, AgentStatus

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """
    Represents a task that can be assigned to agents.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    required_capabilities: List[AgentCapability] = field(default_factory=list)
    preferred_agent_type: Optional[str] = None
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout: Optional[timedelta] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Task IDs this task depends on
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "required_capabilities": [cap.value for cap in self.required_capabilities],
            "preferred_agent_type": self.preferred_agent_type,
            "priority": self.priority.value,
            "status": self.status.value,
            "assigned_agent_id": self.assigned_agent_id,
            "created_at": self.created_at.isoformat(),
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "timeout": self.timeout.total_seconds() if self.timeout else None,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "dependencies": self.dependencies
        }


@dataclass
class WorkflowStep:
    """
    Represents a step in a workflow.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    task_template: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    parallel_execution: bool = False
    optional: bool = False
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class Workflow:
    """
    Represents a multi-step workflow.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiAgentCoordinator:
    """
    Central coordinator for managing multi-agent workflows and task distribution.
    """
    
    def __init__(self, agent_registry: AgentRegistry, communication: AgentCommunication):
        """
        Initialize the coordinator.
        
        Args:
            agent_registry: Agent registry instance
            communication: Communication system instance
        """
        self.agent_registry = agent_registry
        self.communication = communication
        self.tasks: Dict[str, Task] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.task_queue: List[str] = []  # Task IDs in priority order
        self.active_tasks: Dict[str, str] = {}  # task_id -> agent_id
        self.task_callbacks: Dict[str, Callable] = {}
        self.is_running = False
        self._coordinator_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the coordinator."""
        if self.is_running:
            logger.warning("Coordinator is already running")
            return
        
        self.is_running = True
        self._coordinator_task = asyncio.create_task(self._coordination_loop())
        logger.info("Multi-agent coordinator started")
    
    async def stop(self) -> None:
        """Stop the coordinator."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self._coordinator_task:
            self._coordinator_task.cancel()
            try:
                await self._coordinator_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Multi-agent coordinator stopped")
    
    def create_task(
        self,
        name: str,
        description: str,
        required_capabilities: List[AgentCapability],
        input_data: Dict[str, Any],
        preferred_agent_type: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[timedelta] = None,
        dependencies: Optional[List[str]] = None,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Create a new task.
        
        Args:
            name: Task name
            description: Task description
            required_capabilities: Required agent capabilities
            input_data: Input data for the task
            preferred_agent_type: Preferred agent type
            priority: Task priority
            timeout: Task timeout
            dependencies: Task dependencies (task IDs)
            callback: Callback function for task completion
            
        Returns:
            str: Task ID
        """
        task = Task(
            name=name,
            description=description,
            required_capabilities=required_capabilities,
            preferred_agent_type=preferred_agent_type,
            priority=priority,
            timeout=timeout,
            input_data=input_data,
            dependencies=dependencies or []
        )
        
        self.tasks[task.id] = task
        
        if callback:
            self.task_callbacks[task.id] = callback
        
        # Add to queue in priority order
        self._add_task_to_queue(task.id)
        
        logger.info(f"Created task: {name} ({task.id}) with priority {priority.value}")
        return task.id
    
    def create_workflow(
        self,
        name: str,
        description: str,
        steps: List[WorkflowStep]
    ) -> str:
        """
        Create a new workflow.
        
        Args:
            name: Workflow name
            description: Workflow description
            steps: Workflow steps
            
        Returns:
            str: Workflow ID
        """
        workflow = Workflow(
            name=name,
            description=description,
            steps=steps
        )
        
        self.workflows[workflow.id] = workflow
        
        logger.info(f"Created workflow: {name} ({workflow.id}) with {len(steps)} steps")
        return workflow.id
    
    async def execute_workflow(self, workflow_id: str) -> bool:
        """
        Execute a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            bool: True if workflow started successfully
        """
        if workflow_id not in self.workflows:
            logger.error(f"Workflow {workflow_id} not found")
            return False
        
        workflow = self.workflows[workflow_id]
        workflow.status = TaskStatus.IN_PROGRESS
        workflow.started_at = datetime.now()
        
        # Create tasks for each workflow step
        step_tasks = {}
        
        for step in workflow.steps:
            task_id = self.create_task(
                name=f"{workflow.name} - {step.name}",
                description=step.task_template.get("description", ""),
                required_capabilities=[
                    AgentCapability(cap) for cap in step.task_template.get("required_capabilities", [])
                ],
                input_data=step.task_template.get("input_data", {}),
                preferred_agent_type=step.task_template.get("preferred_agent_type"),
                priority=TaskPriority(step.task_template.get("priority", TaskPriority.NORMAL.value)),
                dependencies=[step_tasks[dep] for dep in step.dependencies if dep in step_tasks]
            )
            step_tasks[step.id] = task_id
        
        logger.info(f"Started workflow: {workflow.name} ({workflow_id})")
        return True
    
    async def assign_task(self, task_id: str) -> bool:
        """
        Assign a task to an appropriate agent.
        
        Args:
            task_id: Task identifier
            
        Returns:
            bool: True if task was assigned successfully
        """
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return False
        
        task = self.tasks[task_id]
        
        # Check if dependencies are completed
        if not self._are_dependencies_completed(task):
            logger.debug(f"Task {task_id} dependencies not yet completed")
            return False
        
        # Find the best agent for this task
        agent_id = self.agent_registry.get_best_agent_for_task(
            task.required_capabilities,
            task.preferred_agent_type
        )
        
        if not agent_id:
            logger.warning(f"No suitable agent found for task {task_id}")
            return False
        
        # Update agent status
        self.agent_registry.update_agent_status(agent_id, AgentStatus.BUSY)
        
        # Update task
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent_id = agent_id
        task.assigned_at = datetime.now()
        
        # Track active task
        self.active_tasks[task_id] = agent_id
        
        # Send task to agent
        message = Message(
            sender_id="coordinator",
            recipient_id=agent_id,
            message_type=MessageType.TASK_REQUEST,
            priority=Priority(task.priority.value),
            content={
                "task_id": task_id,
                "task_name": task.name,
                "task_description": task.description,
                "input_data": task.input_data,
                "required_capabilities": [cap.value for cap in task.required_capabilities]
            },
            requires_response=True,
            response_timeout=task.timeout.total_seconds() if task.timeout else None
        )
        
        await self.communication.send_message(message)
        
        logger.info(f"Assigned task {task_id} to agent {agent_id}")
        return True
    
    async def complete_task(self, task_id: str, output_data: Dict[str, Any], success: bool = True) -> None:
        """
        Mark a task as completed.
        
        Args:
            task_id: Task identifier
            output_data: Task output data
            success: Whether the task completed successfully
        """
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return
        
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        task.completed_at = datetime.now()
        task.output_data = output_data
        
        # Free up the agent
        if task.assigned_agent_id:
            self.agent_registry.update_agent_status(task.assigned_agent_id, AgentStatus.ACTIVE)
        
        # Remove from active tasks
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        
        # Call callback if registered
        if task_id in self.task_callbacks:
            try:
                await self.task_callbacks[task_id](task)
            except Exception as e:
                logger.error(f"Error in task callback for {task_id}: {e}")
            finally:
                del self.task_callbacks[task_id]
        
        logger.info(f"Task {task_id} completed with status: {task.status.value}")
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self.workflows.get(workflow_id)
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks."""
        return [task for task in self.tasks.values() if task.status == TaskStatus.PENDING]
    
    def get_active_tasks(self) -> List[Task]:
        """Get all active tasks."""
        return [task for task in self.tasks.values() if task.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]]
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        total_tasks = len(self.tasks)
        status_counts = {}
        
        for task in self.tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_tasks": total_tasks,
            "active_tasks": len(self.active_tasks),
            "pending_tasks": len(self.task_queue),
            "total_workflows": len(self.workflows),
            "task_status_distribution": status_counts,
            "is_running": self.is_running
        }
    
    async def _coordination_loop(self) -> None:
        """Main coordination loop."""
        while self.is_running:
            try:
                # Process pending tasks
                await self._process_task_queue()
                
                # Check for timeouts
                await self._check_task_timeouts()
                
                # Brief pause to prevent busy waiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(5)  # Longer pause on error
    
    async def _process_task_queue(self) -> None:
        """Process the task queue and assign tasks to agents."""
        tasks_to_remove = []
        
        for task_id in self.task_queue:
            if task_id not in self.tasks:
                tasks_to_remove.append(task_id)
                continue
            
            task = self.tasks[task_id]
            
            if task.status != TaskStatus.PENDING:
                tasks_to_remove.append(task_id)
                continue
            
            # Try to assign the task
            if await self.assign_task(task_id):
                tasks_to_remove.append(task_id)
        
        # Remove processed tasks from queue
        for task_id in tasks_to_remove:
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
    
    async def _check_task_timeouts(self) -> None:
        """Check for task timeouts and handle them."""
        current_time = datetime.now()
        
        for task_id, task in self.tasks.items():
            if (task.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS] and
                task.timeout and task.assigned_at and
                current_time - task.assigned_at > task.timeout):
                
                logger.warning(f"Task {task_id} timed out")
                task.status = TaskStatus.FAILED
                task.error_message = "Task timed out"
                
                # Free up the agent
                if task.assigned_agent_id:
                    self.agent_registry.update_agent_status(task.assigned_agent_id, AgentStatus.ACTIVE)
                
                # Remove from active tasks
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
    
    def _add_task_to_queue(self, task_id: str) -> None:
        """Add a task to the queue in priority order."""
        task = self.tasks[task_id]
        
        # Find the correct position based on priority
        insert_index = len(self.task_queue)
        for i, existing_task_id in enumerate(self.task_queue):
            existing_task = self.tasks[existing_task_id]
            if task.priority.value > existing_task.priority.value:
                insert_index = i
                break
        
        self.task_queue.insert(insert_index, task_id)
    
    def _are_dependencies_completed(self, task: Task) -> bool:
        """Check if all task dependencies are completed."""
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.tasks:
                return False
            
            dep_task = self.tasks[dep_task_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
