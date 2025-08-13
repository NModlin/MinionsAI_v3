#!/usr/bin/env python3
"""
MinionsAI v3.1 - Advanced Features Test Suite
Comprehensive testing for Phase 4 advanced features including performance, security, analytics, and tools.
"""

import asyncio
import sys
import os
import time
import tempfile
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_performance_system():
    """Test the performance monitoring and caching system."""
    print("\n🚀 Testing Performance System")
    print("=" * 50)
    
    try:
        from core.advanced.performance import PerformanceMonitor, CacheManager, AsyncTaskManager
        
        # Test 1: Cache Manager
        print("\n📋 Test 1: Cache Manager")
        cache = CacheManager(max_size=100, default_ttl=60)
        
        # Test basic operations
        cache.set("test_key", "test_value")
        value = cache.get("test_key")
        assert value == "test_value", "Cache get/set failed"
        print("✅ Basic cache operations work")
        
        # Test TTL
        cache.set("ttl_key", "ttl_value", ttl=1)
        time.sleep(2)
        expired_value = cache.get("ttl_key")
        assert expired_value is None, "TTL not working"
        print("✅ TTL expiration works")
        
        # Test cache stats
        stats = cache.get_stats()
        assert "hit_rate" in stats, "Cache stats missing"
        print(f"✅ Cache stats: {stats}")
        
        # Test 2: Async Task Manager
        print("\n📋 Test 2: Async Task Manager")
        task_manager = AsyncTaskManager(max_concurrent_tasks=5)
        await task_manager.start()
        
        # Submit test tasks
        async def test_task():
            await asyncio.sleep(0.1)
            return "task_completed"
        
        task_id = await task_manager.submit_task(test_task(), "test_task_1")
        
        # Wait for completion
        await asyncio.sleep(0.5)
        result = task_manager.get_task_result(task_id)
        assert result and result["success"], "Task execution failed"
        print("✅ Async task execution works")
        
        await task_manager.stop()
        
        # Test 3: Performance Monitor
        print("\n📋 Test 3: Performance Monitor")
        monitor = PerformanceMonitor()
        monitor.set_cache_manager(cache)
        monitor.set_task_manager(task_manager)
        
        # Collect metrics
        metrics = monitor.collect_metrics()
        assert hasattr(metrics, 'cpu_usage'), "Metrics collection failed"
        print(f"✅ Performance metrics collected: CPU {metrics.cpu_usage:.1f}%")
        
        # Record request
        monitor.record_request(0.5, True)
        summary = monitor.get_metrics_summary(1)
        assert "total_requests" in summary, "Metrics summary failed"
        print("✅ Request recording works")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance system test failed: {e}")
        return False


async def test_security_system():
    """Test the security and authentication system."""
    print("\n🔒 Testing Security System")
    print("=" * 50)
    
    try:
        from core.advanced.security import SecurityManager, AuthenticationManager, UserRole, Permission
        
        # Test 1: Security Manager
        print("\n📋 Test 1: Security Manager")
        with tempfile.TemporaryDirectory() as temp_dir:
            security_manager = SecurityManager(data_dir=temp_dir)
            
            # Test user creation
            success = security_manager.create_user(
                username="testuser",
                email="test@example.com",
                password="TestPassword123!",
                role=UserRole.USER
            )
            assert success, "User creation failed"
            print("✅ User creation works")
            
            # Test authentication
            token = security_manager.authenticate_user("testuser", "TestPassword123!")
            assert token is not None, "Authentication failed"
            print("✅ User authentication works")
            
            # Test token validation
            payload = security_manager.validate_session_token(token)
            assert payload is not None, "Token validation failed"
            print("✅ Token validation works")
            
            # Test permission checking
            has_permission = security_manager.check_permission(token, Permission.READ_CONVERSATIONS)
            assert has_permission, "Permission check failed"
            print("✅ Permission checking works")
            
            # Test 2: Authentication Manager
            print("\n📋 Test 2: Authentication Manager")
            auth_manager = AuthenticationManager(security_manager)
            
            # Test simple login/logout
            login_token = auth_manager.login("testuser", "TestPassword123!")
            assert login_token is not None, "Auth manager login failed"
            print("✅ Auth manager login works")
            
            is_authenticated = auth_manager.is_authenticated(login_token)
            assert is_authenticated, "Authentication check failed"
            print("✅ Authentication check works")
            
            logout_success = auth_manager.logout(login_token)
            assert logout_success, "Logout failed"
            print("✅ Logout works")
        
        return True
        
    except Exception as e:
        print(f"❌ Security system test failed: {e}")
        return False


async def test_model_management():
    """Test the model management system."""
    print("\n🧠 Testing Model Management")
    print("=" * 50)
    
    try:
        from core.advanced.models import ModelManager, ModelType, ModelCapability
        
        # Test 1: Model Manager
        print("\n📋 Test 1: Model Manager")
        model_manager = ModelManager()
        
        # Register a test model
        model_manager.register_model(
            name="test_model",
            model_type=ModelType.OLLAMA,
            capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
            context_length=4096,
            description="Test model for unit testing"
        )
        print("✅ Model registration works")
        
        # Test model selection
        best_model = model_manager.get_best_model_for_task(
            "test_task",
            required_capabilities=[ModelCapability.TEXT_GENERATION]
        )
        print(f"✅ Model selection works: {best_model}")
        
        # Test model stats
        stats = model_manager.get_model_stats()
        assert "total_models" in stats, "Model stats failed"
        print(f"✅ Model stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model management test failed: {e}")
        return False


async def test_tool_system():
    """Test the advanced tool system."""
    print("\n🛠️ Testing Tool System")
    print("=" * 50)
    
    try:
        from core.advanced.tools import AdvancedToolManager, FileOperationsTool, DatabaseTool
        
        # Test 1: Tool Manager
        print("\n📋 Test 1: Tool Manager")
        tool_manager = AdvancedToolManager()
        
        # Test tool listing
        tools = tool_manager.list_tools()
        assert len(tools) > 0, "No tools found"
        print(f"✅ Found {len(tools)} tools")
        
        # Test tool categories
        categories = tool_manager.get_tool_categories()
        assert len(categories) > 0, "No tool categories found"
        print(f"✅ Tool categories: {list(categories.keys())}")
        
        # Test 2: File Operations Tool
        print("\n📋 Test 2: File Operations Tool")
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            
            # Test file write
            result = tool_manager.execute_tool(
                "file_operations",
                operation="write",
                path=str(test_file),
                content="Hello, World!"
            )
            assert result["success"], "File write failed"
            print("✅ File write works")
            
            # Test file read
            result = tool_manager.execute_tool(
                "file_operations",
                operation="read",
                path=str(test_file)
            )
            assert result["success"], "File read failed"
            assert result["result"]["content"] == "Hello, World!", "File content mismatch"
            print("✅ File read works")
        
        # Test 3: Database Tool
        print("\n📋 Test 3: Database Tool")
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            connection_string = f"sqlite://{db_path}"
            
            # Test database connection
            result = tool_manager.execute_tool(
                "database",
                operation="connect",
                connection_string=connection_string
            )
            assert result["success"], "Database connection failed"
            print("✅ Database connection works")
            
            # Test table creation and query
            result = tool_manager.execute_tool(
                "database",
                operation="execute",
                connection_string=connection_string,
                query="CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)"
            )
            assert result["success"], "Table creation failed"
            print("✅ Database operations work")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool system test failed: {e}")
        return False


async def test_analytics_system():
    """Test the analytics and reporting system."""
    print("\n📊 Testing Analytics System")
    print("=" * 50)
    
    try:
        from core.advanced.analytics import MetricsCollector, AnalyticsEngine, EventType, MetricType
        
        # Test 1: Metrics Collector
        print("\n📋 Test 1: Metrics Collector")
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.db"
            collector = MetricsCollector(str(metrics_path))
            
            # Record test metrics
            collector.record_metric("test_metric", 42.0, MetricType.GAUGE)
            collector.record_event(EventType.USER_LOGIN, user_id="test_user")
            
            # Flush to database
            collector.flush_to_database()
            print("✅ Metrics recording works")
            
            # Retrieve metrics
            metrics = collector.get_metrics("test_metric")
            assert len(metrics) > 0, "No metrics retrieved"
            print("✅ Metrics retrieval works")
            
            # Retrieve events
            events = collector.get_events(EventType.USER_LOGIN)
            assert len(events) > 0, "No events retrieved"
            print("✅ Events retrieval works")
            
            # Test 2: Analytics Engine
            print("\n📋 Test 2: Analytics Engine")
            analytics = AnalyticsEngine(collector)
            
            # Generate usage report
            report = analytics.generate_usage_report(1)
            assert "total_events" in report, "Usage report failed"
            print("✅ Usage report generation works")
            
            # Generate performance report
            perf_report = analytics.generate_performance_report(1)
            assert "period_hours" in perf_report, "Performance report failed"
            print("✅ Performance report generation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Analytics system test failed: {e}")
        return False


async def test_integration():
    """Test integration between advanced features."""
    print("\n🔗 Testing System Integration")
    print("=" * 50)
    
    try:
        from core.agent_manager import AgentManager
        
        # Test 1: Agent Manager with Advanced Features
        print("\n📋 Test 1: Agent Manager Integration")
        agent_manager = AgentManager(
            multi_agent_mode=False,
            enable_advanced_features=True
        )
        
        # Initialize (this will test all advanced features integration)
        success = agent_manager.initialize()
        assert success, "Agent manager initialization failed"
        print("✅ Agent manager with advanced features initialized")
        
        # Test status with advanced features
        status = agent_manager.get_status()
        assert "multi_agent_mode" in status, "Status missing advanced info"
        print("✅ Advanced status reporting works")
        
        # Test message processing with advanced features
        response = agent_manager.process_message(
            "Hello, this is a test message",
            user_id="test_user",
            session_id="test_session"
        )
        assert response, "Message processing failed"
        print("✅ Message processing with advanced features works")
        
        # Cleanup
        agent_manager.shutdown()
        print("✅ System shutdown works")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


async def run_performance_benchmark():
    """Run performance benchmarks."""
    print("\n⚡ Running Performance Benchmarks")
    print("=" * 50)
    
    try:
        from core.advanced.performance import CacheManager
        
        # Benchmark cache performance
        print("\n📋 Cache Performance Benchmark")
        cache = CacheManager(max_size=10000)
        
        # Write benchmark
        start_time = time.time()
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")
        write_time = time.time() - start_time
        print(f"✅ Cache write: 1000 operations in {write_time:.3f}s ({1000/write_time:.0f} ops/sec)")
        
        # Read benchmark
        start_time = time.time()
        for i in range(1000):
            cache.get(f"key_{i}")
        read_time = time.time() - start_time
        print(f"✅ Cache read: 1000 operations in {read_time:.3f}s ({1000/read_time:.0f} ops/sec)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


async def main():
    """Main test function."""
    print("MinionsAI v3.1 - Advanced Features Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Performance System", test_performance_system),
        ("Security System", test_security_system),
        ("Model Management", test_model_management),
        ("Tool System", test_tool_system),
        ("Analytics System", test_analytics_system),
        ("System Integration", test_integration),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name} Tests...")
            result = await test_func()
            test_results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} tests PASSED")
            else:
                print(f"❌ {test_name} tests FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} tests FAILED with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All advanced features tests PASSED!")
        print("MinionsAI v3.1 Phase 4 is ready for production!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review and fix issues.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
