"""
MinionsAI v3.1 - Advanced Features Module
Advanced features including performance optimization, enterprise capabilities, and production-ready enhancements.
"""

__version__ = "3.1.0"
__author__ = "MinionsAI Team"

# Performance and caching
from .performance import PerformanceMonitor, CacheManager, AsyncTaskManager
from .security import SecurityManager, AuthenticationManager
from .models import ModelManager, ModelOptimizer
from .tools import AdvancedToolManager, CustomToolFramework
from .analytics import AnalyticsEngine, MetricsCollector
from .deployment import DeploymentManager, ContainerManager

__all__ = [
    # Performance
    "PerformanceMonitor",
    "CacheManager", 
    "AsyncTaskManager",
    
    # Security
    "SecurityManager",
    "AuthenticationManager",
    
    # Models
    "ModelManager",
    "ModelOptimizer",
    
    # Tools
    "AdvancedToolManager",
    "CustomToolFramework",
    
    # Analytics
    "AnalyticsEngine",
    "MetricsCollector",
    
    # Deployment
    "DeploymentManager",
    "ContainerManager"
]
