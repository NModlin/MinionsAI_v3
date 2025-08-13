#!/usr/bin/env python3
"""
MinionsAI v3.1 - Phase 1, Task 1.2: First Agent Graph
A LangGraph-based agent with DuckDuckGo search tool capability.

This script creates a simple agent that can:
1. Think and reason about queries
2. Use DuckDuckGo search tool to find information
3. Provide comprehensive responses based on search results
"""

import os
import json
from typing import TypedDict, Annotated, List, Dict, Any
from datetime import datetime

# Core LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM

# DuckDuckGo search import
from duckduckgo_search import DDGS

# Environment setup
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")


class AgentState(TypedDict):
    """
    State schema for the agent graph.
    Contains the conversation messages and any additional metadata.
    """
    messages: Annotated[List[Dict[str, Any]], "The conversation messages"]
    search_results: Annotated[List[Dict[str, Any]], "Search results from tools"]
    iteration_count: Annotated[int, "Number of iterations in the current conversation"]


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
        
        return formatted_results
    
    except Exception as e:
        return f"Error performing search: {str(e)}"


def create_agent_node(llm):
    """
    Create the agent node function that handles reasoning and decision making.
    
    Args:
        llm: The language model instance
    
    Returns:
        Function that processes agent state and returns updated state
    """
    def agent_node(state: AgentState) -> AgentState:
        """
        Agent node that processes the current state and decides on next actions.
        """
        messages = state.get("messages", [])
        iteration_count = state.get("iteration_count", 0)
        
        # Create system message for the agent
        system_prompt = """You are MinionsAI, a helpful AI assistant with access to web search capabilities.

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

        # Prepare messages for the LLM
        llm_messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        for msg in messages:
            if msg["role"] == "user":
                llm_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                llm_messages.append(AIMessage(content=msg["content"]))
        
        # Get response from LLM
        try:
            response = llm.invoke(llm_messages)
            
            # Check if the agent wants to use tools
            # For this simple implementation, we'll use keyword detection
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Simple heuristic to determine if search is needed
            search_keywords = ["search for", "look up", "find information", "current", "recent", "latest"]
            needs_search = any(keyword in response_content.lower() for keyword in search_keywords)
            
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
    
    return agent_node


def create_agent_graph():
    """
    Create and compile the LangGraph agent with tool capabilities.
    
    Returns:
        Compiled LangGraph application
    """
    # Initialize the LLM (Ollama)
    try:
        llm = OllamaLLM(model="llama3:8b", base_url="http://localhost:11434")
        print("✅ Connected to Ollama LLM")
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("Please ensure Ollama is running and the llama3:8b model is available")
        return None
    
    # Create tools list
    tools = [duckduckgo_search]
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    agent_node = create_agent_node(llm)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    
    # Define the graph flow
    workflow.set_entry_point("agent")
    
    # Add conditional edges (simplified for this version)
    workflow.add_edge("agent", END)
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    print("✅ Agent graph compiled successfully")
    return app


def run_agent_test():
    """
    Test the agent with sample queries to verify functionality.
    """
    print("🚀 Starting MinionsAI Agent Test...")
    print("=" * 50)
    
    # Create the agent
    agent_app = create_agent_graph()
    if not agent_app:
        print("❌ Failed to create agent. Exiting.")
        return
    
    # Test queries
    test_queries = [
        "Hello! Can you introduce yourself?",
        "What is the current weather like? Please search for weather information.",
        "Tell me about the latest developments in AI technology."
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: {query}")
        print("-" * 40)
        
        # Create initial state
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "search_results": [],
            "iteration_count": 0
        }
        
        try:
            # Run the agent
            result = agent_app.invoke(initial_state)
            
            # Display results
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if last_message["role"] == "assistant":
                    print(f"🤖 Agent Response: {last_message['content']}")
                else:
                    print("🤖 Agent Response: No response generated")
            
            print(f"📊 Iterations: {result.get('iteration_count', 0)}")
            
        except Exception as e:
            print(f"❌ Error running agent: {e}")
        
        print("-" * 40)
    
    print("\n✅ Agent testing completed!")


if __name__ == "__main__":
    """
    Main execution block - runs the agent test when script is executed directly.
    """
    print("MinionsAI v3.1 - Phase 1: Core Agent Engine")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run the test
    run_agent_test()
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
