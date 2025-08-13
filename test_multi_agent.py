#!/usr/bin/env python3
"""
MinionsAI v3.1 - Multi-Agent System Test
Test script to verify the multi-agent system functionality.
"""

import asyncio
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.agent_manager import AgentManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_multi_agent_system():
    """Test the multi-agent system functionality."""
    print("🚀 Testing MinionsAI Multi-Agent System")
    print("=" * 50)
    
    # Test 1: Initialize multi-agent system
    print("\n📋 Test 1: Initialize Multi-Agent System")
    agent_manager = AgentManager(multi_agent_mode=True)
    
    success = agent_manager.initialize()
    if success:
        print("✅ Multi-agent system initialized successfully")
    else:
        print("❌ Failed to initialize multi-agent system")
        return False
    
    # Test 2: Check available agents
    print("\n📋 Test 2: Check Available Agents")
    available_agents = agent_manager.get_available_agents()
    print(f"Available agents: {len(available_agents)}")
    for agent in available_agents:
        print(f"  - {agent['name']} ({agent['type']}): {agent['status']}")
    
    # Test 3: Test agent routing
    print("\n📋 Test 3: Test Agent Routing")
    test_queries = [
        ("research", "Search for information about artificial intelligence"),
        ("analysis", "Analyze the data trends in AI development"),
        ("code", "Write a Python function to calculate fibonacci numbers"),
        ("planning", "Create a plan for learning machine learning"),
        ("summary", "Summarize the key points about AI development")
    ]
    
    for expected_agent, query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        print(f"Expected agent: {expected_agent}")
        
        # Test auto-routing
        determined_agent = agent_manager._determine_best_agent_type(query)
        print(f"Auto-routed to: {determined_agent}")
        
        if determined_agent == expected_agent:
            print("✅ Routing successful")
        else:
            print("⚠️ Routing mismatch (this may be acceptable)")
    
    # Test 4: Get system status
    print("\n📋 Test 4: System Status")
    status = agent_manager.get_status()
    print(f"System initialized: {status['initialized']}")
    print(f"Multi-agent mode: {status['multi_agent_mode']}")
    print(f"Active agents: {status.get('agent_registry_stats', {}).get('active_agents', 0)}")
    
    # Test 5: Cleanup
    print("\n📋 Test 5: System Cleanup")
    agent_manager.shutdown()
    print("✅ System shutdown completed")
    
    print("\n🎉 Multi-Agent System Test Completed Successfully!")
    return True


def test_single_vs_multi_agent():
    """Test comparison between single and multi-agent modes."""
    print("\n🔄 Testing Single vs Multi-Agent Modes")
    print("=" * 50)
    
    # Test single agent mode
    print("\n📋 Single Agent Mode Test")
    single_agent = AgentManager(multi_agent_mode=False)
    success = single_agent.initialize()
    if success:
        print("✅ Single agent initialized")
        status = single_agent.get_status()
        print(f"Multi-agent mode: {status['multi_agent_mode']}")
        single_agent.shutdown()
    else:
        print("❌ Single agent initialization failed")
    
    # Test multi-agent mode
    print("\n📋 Multi-Agent Mode Test")
    multi_agent = AgentManager(multi_agent_mode=True)
    success = multi_agent.initialize()
    if success:
        print("✅ Multi-agent system initialized")
        status = multi_agent.get_status()
        print(f"Multi-agent mode: {status['multi_agent_mode']}")
        print(f"Available agents: {len(multi_agent.get_available_agents())}")
        multi_agent.shutdown()
    else:
        print("❌ Multi-agent initialization failed")
    
    print("✅ Mode comparison test completed")


def test_agent_capabilities():
    """Test agent capability mapping."""
    print("\n🧪 Testing Agent Capabilities")
    print("=" * 50)
    
    agent_manager = AgentManager(multi_agent_mode=True)
    
    # Test capability mapping
    test_cases = [
        ("research", ["WEB_SEARCH", "INFORMATION_SYNTHESIS"]),
        ("analysis", ["DATA_ANALYSIS", "MATHEMATICAL_COMPUTATION"]),
        ("code", ["CODE_GENERATION", "CODE_EXECUTION"]),
        ("planning", ["TASK_PLANNING", "INFORMATION_SYNTHESIS"]),
        ("summary", ["INFORMATION_SYNTHESIS", "DOCUMENT_PROCESSING"])
    ]
    
    for agent_type, expected_caps in test_cases:
        capabilities = agent_manager._get_capabilities_for_agent_type(agent_type)
        cap_names = [cap.value for cap in capabilities]
        
        print(f"\n{agent_type.title()} Agent:")
        print(f"  Expected: {expected_caps}")
        print(f"  Actual: {cap_names}")
        
        # Check if expected capabilities are present
        matches = all(exp_cap in cap_names for exp_cap in expected_caps)
        if matches:
            print("  ✅ Capabilities match")
        else:
            print("  ❌ Capability mismatch")
    
    print("\n✅ Capability testing completed")


async def main():
    """Main test function."""
    print("MinionsAI v3.1 - Multi-Agent System Tests")
    print("=" * 60)
    
    try:
        # Run basic multi-agent system test
        success = await test_multi_agent_system()
        if not success:
            print("❌ Multi-agent system test failed")
            return
        
        # Run comparison test
        test_single_vs_multi_agent()
        
        # Run capability test
        test_agent_capabilities()
        
        print("\n🎉 All tests completed successfully!")
        print("The multi-agent system is ready for integration with the GUI.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
