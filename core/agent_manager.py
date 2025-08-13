"""
MinionsAI v3.1 - Enhanced Agent Manager
Manages both single agent and multi-agent system lifecycle with GUI integration.
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional, Generator, Union
from datetime import datetime

# Core LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM

# DuckDuckGo search import
from duckduckgo_search import DDGS

# Multi-agent system imports
from .multi_agent import (
    AgentRegistry, MultiAgentCoordinator, AgentCommunication,
    ResearchAgent, AnalysisAgent, CodeAgent, PlanningAgent, SummaryAgent,
    AgentCapability, TaskPriority, Message, MessageType
)

# Advanced features imports
from .advanced import (
    PerformanceMonitor, CacheManager, AsyncTaskManager,
    SecurityManager, AuthenticationManager,
    ModelManager, ModelOptimizer,
    AdvancedToolManager, CustomToolFramework,
    AnalyticsEngine, MetricsCollector
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")


class AgentState:
    """
    Enhanced state management for the agent with better encapsulation.
    """
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.search_results: List[Dict[str, Any]] = []
        self.iteration_count: int = 0
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.is_processing: bool = False
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the conversation."""
        return self.messages.copy()
    
    def clear_messages(self) -> None:
        """Clear all messages and reset state."""
        self.messages.clear()
        self.search_results.clear()
        self.iteration_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for LangGraph compatibility."""
        return {
            "messages": self.messages,
            "search_results": self.search_results,
            "iteration_count": self.iteration_count
        }


@tool
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Formatted string containing search results
    """
    try:
        logger.info(f"Performing DuckDuckGo search for: {query}")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
        if not results:
            return f"No search results found for query: '{query}'"
        
        formatted_results = f"Search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            href = result.get('href', 'No URL')
            
            formatted_results += f"{i}. **{title}**\n"
            formatted_results += f"   {body}\n"
            formatted_results += f"   URL: {href}\n\n"
        
        logger.info(f"Search completed successfully with {len(results)} results")
        return formatted_results
    
    except Exception as e:
        error_msg = f"Error performing search: {str(e)}"
        logger.error(error_msg)
        return error_msg


class AgentManager:
    """
    Enhanced Agent Manager that supports both single agent and multi-agent modes with advanced features.
    """

    def __init__(self, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434",
                 multi_agent_mode: bool = False, enable_advanced_features: bool = True):
        """
        Initialize the Agent Manager.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for the Ollama service
            multi_agent_mode: Whether to enable multi-agent system
            enable_advanced_features: Whether to enable advanced features
        """
        self.model_name = model_name
        self.base_url = base_url
        self.multi_agent_mode = multi_agent_mode
        self.enable_advanced_features = enable_advanced_features
        self.llm: Optional[OllamaLLM] = None
        self.agent_app = None
        self.tools = [duckduckgo_search]
        self.is_initialized = False
        self.state = AgentState()

        # Multi-agent system components
        self.agent_registry: Optional[AgentRegistry] = None
        self.coordinator: Optional[MultiAgentCoordinator] = None
        self.communication: Optional[AgentCommunication] = None
        self.specialized_agents: Dict[str, Any] = {}

        # Advanced features components
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.cache_manager: Optional[CacheManager] = None
        self.task_manager: Optional[AsyncTaskManager] = None
        self.security_manager: Optional[SecurityManager] = None
        self.auth_manager: Optional[AuthenticationManager] = None
        self.model_manager: Optional[ModelManager] = None
        self.tool_manager: Optional[AdvancedToolManager] = None
        self.analytics_engine: Optional[AnalyticsEngine] = None
        self.metrics_collector: Optional[MetricsCollector] = None

        # System prompt for the agent
        self.system_prompt = """You are MinionsAI, a helpful AI assistant with access to web search capabilities.

Your role:
1. Analyze user queries and determine if you need to search for current information
2. Use the duckduckgo_search tool when you need up-to-date information or facts
3. Provide comprehensive, helpful responses based on your knowledge and search results
4. Be conversational and engaging while being accurate and informative

When to use search:
- For current events, news, or recent information
- For specific facts, statistics, or data you're unsure about
- For information that might have changed recently
- When the user explicitly asks you to search for something

Always explain your reasoning and cite sources when using search results."""
    
    def initialize(self) -> bool:
        """
        Initialize the agent system (single or multi-agent) with advanced features.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info(f"Initializing MinionsAI Agent Manager (Multi-agent: {self.multi_agent_mode}, Advanced: {self.enable_advanced_features})...")

            # Initialize advanced features first
            if self.enable_advanced_features:
                success = asyncio.run(self._initialize_advanced_features())
                if not success:
                    logger.warning("Advanced features initialization failed, continuing with basic features")

            # Initialize the LLM
            self.llm = OllamaLLM(model=self.model_name, base_url=self.base_url)
            logger.info("✅ Connected to Ollama LLM")

            if self.multi_agent_mode:
                # Initialize multi-agent system
                success = asyncio.run(self._initialize_multi_agent_system())
                if not success:
                    return False
            else:
                # Initialize single agent
                self._create_agent_graph()
                logger.info("✅ Single agent graph compiled successfully")

            self.is_initialized = True
            return True

        except Exception as e:
            error_msg = f"❌ Error initializing agent: {e}"
            logger.error(error_msg)
            return False
    
    def _create_agent_graph(self) -> None:
        """Create and compile the LangGraph agent."""
        from typing import TypedDict, Annotated
        
        # Define state schema for LangGraph
        class LangGraphState(TypedDict):
            messages: Annotated[List[Dict[str, Any]], "The conversation messages"]
            search_results: Annotated[List[Dict[str, Any]], "Search results from tools"]
            iteration_count: Annotated[int, "Number of iterations"]
        
        def agent_node(state: LangGraphState) -> LangGraphState:
            """Agent node that processes the current state."""
            messages = state.get("messages", [])
            iteration_count = state.get("iteration_count", 0)
            
            # Prepare messages for the LLM
            llm_messages = [SystemMessage(content=self.system_prompt)]
            
            # Add conversation history
            for msg in messages:
                if msg["role"] == "user":
                    llm_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    llm_messages.append(AIMessage(content=msg["content"]))
            
            # Get response from LLM
            try:
                response = self.llm.invoke(llm_messages)
                response_content = response.content if hasattr(response, 'content') else str(response)
                
                # Update state
                new_messages = messages + [{"role": "assistant", "content": response_content}]
                
                return {
                    "messages": new_messages,
                    "search_results": state.get("search_results", []),
                    "iteration_count": iteration_count + 1
                }
                
            except Exception as e:
                error_msg = f"Error in agent processing: {str(e)}"
                new_messages = messages + [{"role": "assistant", "content": error_msg}]
                
                return {
                    "messages": new_messages,
                    "search_results": state.get("search_results", []),
                    "iteration_count": iteration_count + 1
                }
        
        # Create the state graph
        workflow = StateGraph(LangGraphState)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Define the graph flow
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)
        workflow.add_edge("tools", "agent")
        
        # Compile the graph
        self.agent_app = workflow.compile()

    async def _initialize_advanced_features(self) -> bool:
        """Initialize advanced features."""
        try:
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector()
            logger.info("✅ Metrics collector initialized")

            # Initialize cache manager
            self.cache_manager = CacheManager(max_size=1000, default_ttl=3600)
            logger.info("✅ Cache manager initialized")

            # Initialize async task manager
            self.task_manager = AsyncTaskManager(max_concurrent_tasks=10)
            await self.task_manager.start()
            logger.info("✅ Async task manager started")

            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor()
            self.performance_monitor.set_cache_manager(self.cache_manager)
            self.performance_monitor.set_task_manager(self.task_manager)
            await self.performance_monitor.start_monitoring()
            logger.info("✅ Performance monitor started")

            # Initialize security manager
            self.security_manager = SecurityManager()
            self.auth_manager = AuthenticationManager(self.security_manager)
            logger.info("✅ Security system initialized")

            # Initialize model manager
            self.model_manager = ModelManager()
            # Register default Ollama model
            from .advanced.models import ModelType, ModelCapability
            self.model_manager.register_model(
                name=self.model_name,
                model_type=ModelType.OLLAMA,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.REASONING,
                    ModelCapability.ANALYSIS
                ],
                base_url=self.base_url
            )
            await self.model_manager.initialize_models()
            logger.info("✅ Model manager initialized")

            # Initialize advanced tool manager
            self.tool_manager = AdvancedToolManager()
            logger.info("✅ Advanced tool manager initialized")

            # Initialize analytics engine
            self.analytics_engine = AnalyticsEngine(self.metrics_collector)
            logger.info("✅ Analytics engine initialized")

            return True

        except Exception as e:
            logger.error(f"Error initializing advanced features: {e}")
            return False

    async def _initialize_multi_agent_system(self) -> bool:
        """Initialize the multi-agent system."""
        try:
            # Initialize communication system
            self.communication = AgentCommunication()
            logger.info("✅ Communication system initialized")

            # Initialize agent registry
            self.agent_registry = AgentRegistry()
            logger.info("✅ Agent registry initialized")

            # Initialize coordinator
            self.coordinator = MultiAgentCoordinator(self.agent_registry, self.communication)
            await self.coordinator.start()
            logger.info("✅ Multi-agent coordinator started")

            # Create and register specialized agents
            await self._create_specialized_agents()
            logger.info("✅ Specialized agents created and registered")

            return True

        except Exception as e:
            logger.error(f"Error initializing multi-agent system: {e}")
            return False

    async def _create_specialized_agents(self) -> None:
        """Create and register all specialized agents."""
        # Create agents
        agents_to_create = [
            ("research", ResearchAgent(self.model_name, self.base_url)),
            ("analysis", AnalysisAgent(self.model_name, self.base_url)),
            ("code", CodeAgent(self.model_name, self.base_url)),
            ("planning", PlanningAgent(self.model_name, self.base_url)),
            ("summary", SummaryAgent(self.model_name, self.base_url))
        ]

        for agent_key, agent_instance in agents_to_create:
            # Initialize the agent
            success = await agent_instance.initialize(self.communication)
            if success:
                # Register with registry
                agent_id = self.agent_registry.register_agent(agent_instance)
                self.specialized_agents[agent_key] = agent_id
                logger.info(f"✅ {agent_instance.name} registered with ID: {agent_id}")
            else:
                logger.error(f"❌ Failed to initialize {agent_instance.name}")

    def enable_multi_agent_mode(self) -> bool:
        """Enable multi-agent mode if not already enabled."""
        if self.multi_agent_mode:
            return True

        self.multi_agent_mode = True
        if self.is_initialized:
            # Re-initialize with multi-agent mode
            return self.initialize()
        return True

    def disable_multi_agent_mode(self) -> bool:
        """Disable multi-agent mode and return to single agent."""
        if not self.multi_agent_mode:
            return True

        # Shutdown multi-agent components
        if self.coordinator:
            asyncio.run(self.coordinator.stop())

        self.multi_agent_mode = False
        self.agent_registry = None
        self.coordinator = None
        self.communication = None
        self.specialized_agents.clear()

        if self.is_initialized:
            # Re-initialize in single agent mode
            return self.initialize()
        return True
    
    def process_message(self, user_message: str, agent_type: Optional[str] = None,
                       user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """
        Process a user message and return the agent's response with advanced features.

        Args:
            user_message: The user's input message
            agent_type: Specific agent type to use (for multi-agent mode)
            user_id: User identifier for analytics
            session_id: Session identifier for analytics

        Returns:
            str: The agent's response
        """
        if not self.is_initialized:
            return "❌ Agent not initialized. Please check system status."

        start_time = time.time()
        success = True

        try:
            self.state.is_processing = True
            self.state.add_message("user", user_message)

            # Record analytics event
            if self.metrics_collector:
                from .advanced.analytics import EventType
                self.metrics_collector.record_event(
                    EventType.MESSAGE_SENT,
                    user_id=user_id,
                    session_id=session_id,
                    data={"message_length": len(user_message), "agent_type": agent_type}
                )

            # Check cache first
            response = None
            if self.cache_manager:
                cache_key = f"response:{hash(user_message)}"
                response = self.cache_manager.get(cache_key)
                if response:
                    logger.debug("Response served from cache")

            if response is None:
                if self.multi_agent_mode and self.coordinator:
                    # Use multi-agent system
                    response = asyncio.run(self._process_with_multi_agent(user_message, agent_type))
                else:
                    # Use single agent system
                    response = self._process_with_single_agent(user_message)

                # Cache the response
                if self.cache_manager and response:
                    cache_key = f"response:{hash(user_message)}"
                    self.cache_manager.set(cache_key, response, ttl=1800)  # 30 minutes

            return response

        except Exception as e:
            success = False
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)

            # Record error event
            if self.metrics_collector:
                from .advanced.analytics import EventType
                self.metrics_collector.record_event(
                    EventType.ERROR_OCCURRED,
                    user_id=user_id,
                    session_id=session_id,
                    data={"error": str(e), "message": user_message}
                )

            return error_msg

        finally:
            self.state.is_processing = False

            # Record performance metrics
            response_time = time.time() - start_time
            if self.performance_monitor:
                self.performance_monitor.record_request(response_time, success)

            if self.metrics_collector:
                from .advanced.analytics import MetricType
                self.metrics_collector.record_metric(
                    "response_time",
                    response_time,
                    MetricType.TIMER,
                    tags={"agent_type": agent_type or "single", "success": str(success)}
                )

    def _process_with_single_agent(self, user_message: str) -> str:
        """Process message with single agent system."""
        # Create initial state for LangGraph
        initial_state = self.state.to_dict()

        # Run the agent
        result = self.agent_app.invoke(initial_state)

        # Update our state with the result
        self.state.messages = result.get("messages", [])
        self.state.search_results = result.get("search_results", [])
        self.state.iteration_count = result.get("iteration_count", 0)

        # Get the last assistant message
        messages = self.state.get_messages()
        if messages:
            last_message = messages[-1]
            if last_message["role"] == "assistant":
                return last_message["content"]

        return "No response generated."

    async def _process_with_multi_agent(self, user_message: str, agent_type: Optional[str] = None) -> str:
        """Process message with multi-agent system."""
        # Determine which agent(s) to use
        if agent_type and agent_type in self.specialized_agents:
            # Use specific agent
            agent_id = self.specialized_agents[agent_type]
            task_id = self.coordinator.create_task(
                name=f"User Query - {agent_type}",
                description=user_message,
                required_capabilities=self._get_capabilities_for_agent_type(agent_type),
                input_data={"query": user_message, "task_type": "user_query"},
                preferred_agent_type=agent_type.title() + "Agent"
            )
        else:
            # Auto-route based on message content
            agent_type = self._determine_best_agent_type(user_message)
            required_capabilities = self._get_capabilities_for_agent_type(agent_type)

            task_id = self.coordinator.create_task(
                name="User Query - Auto-routed",
                description=user_message,
                required_capabilities=required_capabilities,
                input_data={"query": user_message, "task_type": "user_query"}
            )

        # Wait for task completion (simplified for now)
        max_wait = 30  # seconds
        wait_time = 0
        while wait_time < max_wait:
            await asyncio.sleep(1)
            wait_time += 1

            task = self.coordinator.get_task(task_id)
            if task and task.status.value in ["completed", "failed"]:
                if task.status.value == "completed":
                    result = task.output_data.get("result", {})
                    if isinstance(result, dict):
                        return result.get("response", result.get("analysis", result.get("summary", str(result))))
                    return str(result)
                else:
                    return f"Task failed: {task.error_message}"

        return "Response timeout - the agent is still processing your request."
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return self.state.get_messages()
    
    def clear_conversation(self) -> None:
        """Clear the current conversation history."""
        self.state.clear_messages()
        logger.info("Conversation history cleared")
    
    def _determine_best_agent_type(self, message: str) -> str:
        """Determine the best agent type for a message."""
        message_lower = message.lower()

        # Simple keyword-based routing
        if any(word in message_lower for word in ["search", "research", "find", "information", "fact"]):
            return "research"
        elif any(word in message_lower for word in ["analyze", "data", "statistics", "pattern", "trend"]):
            return "analysis"
        elif any(word in message_lower for word in ["code", "program", "debug", "function", "script"]):
            return "code"
        elif any(word in message_lower for word in ["plan", "strategy", "organize", "workflow", "steps"]):
            return "planning"
        elif any(word in message_lower for word in ["summarize", "summary", "report", "synthesis"]):
            return "summary"
        else:
            return "research"  # Default to research agent

    def _get_capabilities_for_agent_type(self, agent_type: str) -> List[AgentCapability]:
        """Get required capabilities for an agent type."""
        capability_map = {
            "research": [AgentCapability.WEB_SEARCH, AgentCapability.INFORMATION_SYNTHESIS],
            "analysis": [AgentCapability.DATA_ANALYSIS, AgentCapability.MATHEMATICAL_COMPUTATION],
            "code": [AgentCapability.CODE_GENERATION, AgentCapability.CODE_EXECUTION],
            "planning": [AgentCapability.TASK_PLANNING, AgentCapability.INFORMATION_SYNTHESIS],
            "summary": [AgentCapability.INFORMATION_SYNTHESIS, AgentCapability.DOCUMENT_PROCESSING]
        }
        return capability_map.get(agent_type, [AgentCapability.NATURAL_LANGUAGE])

    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents."""
        if not self.multi_agent_mode or not self.agent_registry:
            return [{"name": "MinionsAI", "type": "general", "status": "active"}]

        agents = []
        for agent_key, agent_id in self.specialized_agents.items():
            agent_info = self.agent_registry.get_agent_info(agent_id)
            if agent_info:
                agents.append({
                    "name": agent_info.name,
                    "type": agent_key,
                    "status": agent_info.status.value,
                    "capabilities": [cap.value for cap in agent_info.capabilities],
                    "description": agent_info.description
                })
        return agents

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent system.

        Returns:
            Dict containing status information
        """
        status = {
            "initialized": self.is_initialized,
            "processing": self.state.is_processing,
            "message_count": len(self.state.messages),
            "session_id": self.state.session_id,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "multi_agent_mode": self.multi_agent_mode
        }

        if self.multi_agent_mode and self.agent_registry:
            status["agent_registry_stats"] = self.agent_registry.get_registry_stats()
            status["available_agents"] = self.get_available_agents()

        if self.coordinator:
            status["coordinator_stats"] = self.coordinator.get_coordinator_stats()

        return status
    
    def shutdown(self) -> None:
        """Shutdown the agent system and cleanup resources."""
        logger.info("Shutting down Agent Manager...")

        if self.multi_agent_mode:
            # Shutdown multi-agent components
            if self.coordinator:
                asyncio.run(self.coordinator.stop())

            # Shutdown specialized agents
            for agent_key, agent_id in self.specialized_agents.items():
                agent_instance = self.agent_registry.get_agent(agent_id)
                if agent_instance:
                    asyncio.run(agent_instance.shutdown())

            self.specialized_agents.clear()
            self.agent_registry = None
            self.coordinator = None
            self.communication = None

        self.is_initialized = False
        self.llm = None
        self.agent_app = None
        logger.info("Agent Manager shutdown complete")
