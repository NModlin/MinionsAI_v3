"""
MinionsAI v3.1 - Performance Management
Advanced performance monitoring, caching, and optimization systems.
"""

import asyncio
import time
import hashlib
import json
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import psutil
import weakref

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time: float = 0.0
    cache_hit_rate: float = 0.0
    active_tasks: int = 0
    error_count: int = 0
    throughput: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": self.response_time,
            "cache_hit_rate": self.cache_hit_rate,
            "active_tasks": self.active_tasks,
            "error_count": self.error_count,
            "throughput": self.throughput
        }


class CacheManager:
    """
    Advanced caching system with TTL, LRU eviction, and intelligent cache warming.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize the cache manager.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {"args": args, "kwargs": kwargs}
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if datetime.now() > entry["expires"]:
                del self.cache[key]
                del self.access_times[key]
                self.miss_count += 1
                return None
            
            # Update access time
            self.access_times[key] = datetime.now()
            self.hit_count += 1
            return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            expires = datetime.now() + timedelta(seconds=ttl)
            self.cache[key] = {
                "value": value,
                "expires": expires,
                "created": datetime.now()
            }
            self.access_times[key] = datetime.now()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "memory_usage": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of cache in bytes."""
        # Rough estimation
        return len(json.dumps(self.cache, default=str).encode())
    
    def cached(self, ttl: Optional[int] = None):
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                cache_key = f"{func.__name__}:{self._generate_key(*args, **kwargs)}"
                
                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator


class AsyncTaskManager:
    """
    Advanced asynchronous task management with priority queues and resource limits.
    """
    
    def __init__(self, max_concurrent_tasks: int = 10, max_queue_size: int = 100):
        """
        Initialize the async task manager.
        
        Args:
            max_concurrent_tasks: Maximum number of concurrent tasks
            max_queue_size: Maximum queue size
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_queue_size = max_queue_size
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.task_results: Dict[str, Any] = {}
        self.task_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.is_running = False
        self._worker_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the task manager."""
        if self.is_running:
            return
        
        self.is_running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("AsyncTaskManager started")
    
    async def stop(self) -> None:
        """Stop the task manager."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all active tasks
        for task in self.active_tasks.values():
            task.cancel()
        
        # Wait for worker to finish
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AsyncTaskManager stopped")
    
    async def submit_task(
        self, 
        coro: Callable, 
        task_id: str, 
        priority: int = 5,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Submit a task for execution.
        
        Args:
            coro: Coroutine to execute
            task_id: Unique task identifier
            priority: Task priority (lower = higher priority)
            callback: Optional callback function
            
        Returns:
            str: Task ID
        """
        if callback:
            self.task_callbacks[task_id].append(callback)
        
        await self.task_queue.put((priority, task_id, coro))
        logger.debug(f"Task {task_id} submitted with priority {priority}")
        return task_id
    
    async def _worker_loop(self) -> None:
        """Main worker loop for processing tasks."""
        while self.is_running:
            try:
                # Wait for available slot
                while len(self.active_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.1)
                
                # Get next task
                try:
                    priority, task_id, coro = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Execute task
                task = asyncio.create_task(self._execute_task(task_id, coro))
                self.active_tasks[task_id] = task
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task_id: str, coro: Callable) -> None:
        """Execute a single task."""
        try:
            start_time = time.time()
            result = await coro
            execution_time = time.time() - start_time
            
            # Store result
            self.task_results[task_id] = {
                "result": result,
                "success": True,
                "execution_time": execution_time,
                "completed_at": datetime.now()
            }
            
            # Execute callbacks
            for callback in self.task_callbacks.get(task_id, []):
                try:
                    await callback(result)
                except Exception as e:
                    logger.error(f"Error in task callback: {e}")
            
            logger.debug(f"Task {task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            # Store error
            self.task_results[task_id] = {
                "result": None,
                "success": False,
                "error": str(e),
                "completed_at": datetime.now()
            }
            logger.error(f"Task {task_id} failed: {e}")
        
        finally:
            # Cleanup
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            if task_id in self.task_callbacks:
                del self.task_callbacks[task_id]
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result."""
        return self.task_results.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        return {
            "active_tasks": len(self.active_tasks),
            "queued_tasks": self.task_queue.qsize(),
            "completed_tasks": len(self.task_results),
            "max_concurrent": self.max_concurrent_tasks,
            "is_running": self.is_running
        }


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize the performance monitor.
        
        Args:
            history_size: Number of metrics to keep in history
        """
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.cache_manager: Optional[CacheManager] = None
        self.task_manager: Optional[AsyncTaskManager] = None
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    def set_cache_manager(self, cache_manager: CacheManager) -> None:
        """Set cache manager for monitoring."""
        self.cache_manager = cache_manager
    
    def set_task_manager(self, task_manager: AsyncTaskManager) -> None:
        """Set task manager for monitoring."""
        self.task_manager = task_manager
    
    async def start_monitoring(self, interval: int = 60) -> None:
        """Start performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self, interval: int) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Cache metrics
        cache_hit_rate = 0.0
        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
            cache_hit_rate = cache_stats["hit_rate"]
        
        # Task metrics
        active_tasks = 0
        if self.task_manager:
            task_stats = self.task_manager.get_stats()
            active_tasks = task_stats["active_tasks"]
        
        # Calculate throughput
        uptime = (datetime.now() - self.start_time).total_seconds()
        throughput = self.request_count / uptime if uptime > 0 else 0
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            cache_hit_rate=cache_hit_rate,
            active_tasks=active_tasks,
            error_count=self.error_count,
            throughput=throughput
        )
    
    def record_request(self, response_time: float, success: bool = True) -> None:
        """Record a request for metrics."""
        self.request_count += 1
        if not success:
            self.error_count += 1
    
    def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No metrics available"}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        
        return {
            "period_minutes": minutes,
            "sample_count": len(recent_metrics),
            "avg_cpu_usage": avg_cpu,
            "avg_memory_usage": avg_memory,
            "avg_cache_hit_rate": avg_cache_hit_rate,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": (self.error_count / self.request_count) if self.request_count > 0 else 0,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }
