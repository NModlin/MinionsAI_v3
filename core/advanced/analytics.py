"""
MinionsAI v3.1 - Advanced Analytics & Reporting
Comprehensive analytics engine with metrics collection, reporting, and business intelligence.
"""

import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
import logging
import statistics
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class EventType(Enum):
    """Types of events to track."""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    CONVERSATION_START = "conversation_start"
    CONVERSATION_END = "conversation_end"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    TOOL_USED = "tool_used"
    MODEL_SWITCHED = "model_switched"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_ALERT = "performance_alert"


@dataclass
class Metric:
    """A single metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "metric_type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


@dataclass
class Event:
    """An event record."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "data": self.data
        }


class MetricsCollector:
    """
    Collects and stores various system metrics.
    """
    
    def __init__(self, storage_path: str = "data/metrics.db"):
        """Initialize the metrics collector."""
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(exist_ok=True)
        
        self.metrics_buffer: List[Metric] = []
        self.events_buffer: List[Event] = []
        self.buffer_size = 1000
        self.flush_interval = 60  # seconds
        
        self._initialize_database()
        self._start_background_flush()
    
    def _initialize_database(self) -> None:
        """Initialize the SQLite database for metrics storage."""
        with sqlite3.connect(self.storage_path) as conn:
            # Create metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tags TEXT
                )
            """)
            
            # Create events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    data TEXT
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics(name, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type_timestamp ON events(event_type, timestamp)")
            
            conn.commit()
    
    def _start_background_flush(self) -> None:
        """Start background task to flush metrics to database."""
        asyncio.create_task(self._flush_loop())
    
    async def _flush_loop(self) -> None:
        """Background loop to flush metrics periodically."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                self.flush_to_database()
            except Exception as e:
                logger.error(f"Error in metrics flush loop: {e}")
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {}
        )
        
        self.metrics_buffer.append(metric)
        
        # Auto-flush if buffer is full
        if len(self.metrics_buffer) >= self.buffer_size:
            self.flush_to_database()
    
    def record_event(
        self,
        event_type: EventType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an event."""
        event = Event(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            data=data or {}
        )
        
        self.events_buffer.append(event)
        
        # Auto-flush if buffer is full
        if len(self.events_buffer) >= self.buffer_size:
            self.flush_to_database()
    
    def flush_to_database(self) -> None:
        """Flush buffered metrics and events to database."""
        if not self.metrics_buffer and not self.events_buffer:
            return
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                # Insert metrics
                if self.metrics_buffer:
                    metrics_data = [
                        (
                            metric.name,
                            metric.value,
                            metric.metric_type.value,
                            metric.timestamp.isoformat(),
                            json.dumps(metric.tags)
                        )
                        for metric in self.metrics_buffer
                    ]
                    
                    conn.executemany(
                        "INSERT INTO metrics (name, value, metric_type, timestamp, tags) VALUES (?, ?, ?, ?, ?)",
                        metrics_data
                    )
                
                # Insert events
                if self.events_buffer:
                    events_data = [
                        (
                            event.event_type.value,
                            event.timestamp.isoformat(),
                            event.user_id,
                            event.session_id,
                            json.dumps(event.data)
                        )
                        for event in self.events_buffer
                    ]
                    
                    conn.executemany(
                        "INSERT INTO events (event_type, timestamp, user_id, session_id, data) VALUES (?, ?, ?, ?, ?)",
                        events_data
                    )
                
                conn.commit()
            
            # Clear buffers
            metrics_count = len(self.metrics_buffer)
            events_count = len(self.events_buffer)
            self.metrics_buffer.clear()
            self.events_buffer.clear()
            
            logger.debug(f"Flushed {metrics_count} metrics and {events_count} events to database")
            
        except Exception as e:
            logger.error(f"Error flushing metrics to database: {e}")
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve metrics from database."""
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []
        
        if metric_name:
            query += " AND name = ?"
            params.append(metric_name)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                metrics = []
                for row in cursor.fetchall():
                    metric_data = dict(row)
                    metric_data["tags"] = json.loads(metric_data["tags"] or "{}")
                    
                    # Filter by tags if specified
                    if tags:
                        metric_tags = metric_data["tags"]
                        if not all(metric_tags.get(k) == v for k, v in tags.items()):
                            continue
                    
                    metrics.append(metric_data)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error retrieving metrics: {e}")
            return []
    
    def get_events(
        self,
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve events from database."""
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                events = []
                for row in cursor.fetchall():
                    event_data = dict(row)
                    event_data["data"] = json.loads(event_data["data"] or "{}")
                    events.append(event_data)
                
                return events
                
        except Exception as e:
            logger.error(f"Error retrieving events: {e}")
            return []


class AnalyticsEngine:
    """
    Advanced analytics engine for generating insights and reports.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize the analytics engine."""
        self.metrics_collector = metrics_collector
        self.report_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(minutes=15)
    
    def generate_usage_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate a comprehensive usage report."""
        cache_key = f"usage_report_{days}"
        
        # Check cache
        if cache_key in self.report_cache:
            cached_report = self.report_cache[cache_key]
            if datetime.now() - datetime.fromisoformat(cached_report["generated_at"]) < self.cache_ttl:
                return cached_report
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Get events for the period
        events = self.metrics_collector.get_events(start_time=start_time, end_time=end_time)
        
        # Analyze events
        event_counts = Counter(event["event_type"] for event in events)
        user_activity = defaultdict(int)
        daily_activity = defaultdict(int)
        
        for event in events:
            if event["user_id"]:
                user_activity[event["user_id"]] += 1
            
            event_date = datetime.fromisoformat(event["timestamp"]).date()
            daily_activity[event_date.isoformat()] += 1
        
        # Get metrics
        metrics = self.metrics_collector.get_metrics(start_time=start_time, end_time=end_time)
        
        # Calculate performance metrics
        response_times = [
            m["value"] for m in metrics 
            if m["name"] == "response_time"
        ]
        
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        report = {
            "period_days": days,
            "start_date": start_time.isoformat(),
            "end_date": end_time.isoformat(),
            "total_events": len(events),
            "event_breakdown": dict(event_counts),
            "unique_users": len(user_activity),
            "most_active_users": dict(Counter(user_activity).most_common(10)),
            "daily_activity": dict(daily_activity),
            "avg_response_time": avg_response_time,
            "total_metrics": len(metrics),
            "generated_at": datetime.now().isoformat()
        }
        
        # Cache the report
        self.report_cache[cache_key] = report
        
        return report
    
    def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate a performance analysis report."""
        cache_key = f"performance_report_{hours}"
        
        # Check cache
        if cache_key in self.report_cache:
            cached_report = self.report_cache[cache_key]
            if datetime.now() - datetime.fromisoformat(cached_report["generated_at"]) < self.cache_ttl:
                return cached_report
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get performance metrics
        metrics = self.metrics_collector.get_metrics(start_time=start_time, end_time=end_time)
        
        # Analyze different metric types
        response_times = [m["value"] for m in metrics if m["name"] == "response_time"]
        cpu_usage = [m["value"] for m in metrics if m["name"] == "cpu_usage"]
        memory_usage = [m["value"] for m in metrics if m["name"] == "memory_usage"]
        error_counts = [m["value"] for m in metrics if m["name"] == "error_count"]
        
        # Calculate statistics
        def calculate_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"min": 0, "max": 0, "avg": 0, "median": 0, "std": 0}
            
            return {
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0
            }
        
        report = {
            "period_hours": hours,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "response_time_stats": calculate_stats(response_times),
            "cpu_usage_stats": calculate_stats(cpu_usage),
            "memory_usage_stats": calculate_stats(memory_usage),
            "total_errors": sum(error_counts),
            "data_points": len(metrics),
            "generated_at": datetime.now().isoformat()
        }
        
        # Add performance alerts
        alerts = []
        if response_times and statistics.mean(response_times) > 10:
            alerts.append("High average response time detected")
        
        if cpu_usage and max(cpu_usage) > 90:
            alerts.append("High CPU usage detected")
        
        if memory_usage and max(memory_usage) > 90:
            alerts.append("High memory usage detected")
        
        report["alerts"] = alerts
        
        # Cache the report
        self.report_cache[cache_key] = report
        
        return report
    
    def generate_model_performance_report(self) -> Dict[str, Any]:
        """Generate model performance analysis."""
        # Get model-related events and metrics
        events = self.metrics_collector.get_events(
            start_time=datetime.now() - timedelta(days=7)
        )
        
        model_usage = defaultdict(int)
        model_errors = defaultdict(int)
        
        for event in events:
            if event["event_type"] == "tool_used" and "model" in event.get("data", {}):
                model_name = event["data"]["model"]
                model_usage[model_name] += 1
                
                if not event["data"].get("success", True):
                    model_errors[model_name] += 1
        
        # Calculate success rates
        model_stats = {}
        for model, usage_count in model_usage.items():
            error_count = model_errors.get(model, 0)
            success_rate = (usage_count - error_count) / usage_count if usage_count > 0 else 0
            
            model_stats[model] = {
                "usage_count": usage_count,
                "error_count": error_count,
                "success_rate": success_rate
            }
        
        return {
            "period": "7 days",
            "model_statistics": model_stats,
            "most_used_model": max(model_usage.items(), key=lambda x: x[1])[0] if model_usage else None,
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_user_insights(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Generate insights for a specific user."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Get user events
        events = self.metrics_collector.get_events(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )
        
        if not events:
            return {"error": "No data found for user"}
        
        # Analyze user behavior
        event_counts = Counter(event["event_type"] for event in events)
        
        # Calculate session patterns
        login_events = [e for e in events if e["event_type"] == "user_login"]
        logout_events = [e for e in events if e["event_type"] == "user_logout"]
        
        session_count = len(login_events)
        avg_session_duration = 0
        
        if len(login_events) == len(logout_events):
            session_durations = []
            for login, logout in zip(login_events, logout_events):
                login_time = datetime.fromisoformat(login["timestamp"])
                logout_time = datetime.fromisoformat(logout["timestamp"])
                duration = (logout_time - login_time).total_seconds() / 60  # minutes
                session_durations.append(duration)
            
            if session_durations:
                avg_session_duration = statistics.mean(session_durations)
        
        # Most active hours
        hour_activity = defaultdict(int)
        for event in events:
            hour = datetime.fromisoformat(event["timestamp"]).hour
            hour_activity[hour] += 1
        
        most_active_hour = max(hour_activity.items(), key=lambda x: x[1])[0] if hour_activity else 0
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_events": len(events),
            "event_breakdown": dict(event_counts),
            "session_count": session_count,
            "avg_session_duration_minutes": avg_session_duration,
            "most_active_hour": most_active_hour,
            "activity_by_hour": dict(hour_activity),
            "generated_at": datetime.now().isoformat()
        }
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time system metrics."""
        # Get recent metrics (last 5 minutes)
        recent_time = datetime.now() - timedelta(minutes=5)
        recent_metrics = self.metrics_collector.get_metrics(start_time=recent_time)
        
        # Get current values for key metrics
        current_metrics = {}
        for metric in recent_metrics:
            metric_name = metric["name"]
            if metric_name not in current_metrics or metric["timestamp"] > current_metrics[metric_name]["timestamp"]:
                current_metrics[metric_name] = metric
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {name: data["value"] for name, data in current_metrics.items()},
            "active_sessions": len([m for m in recent_metrics if m["name"] == "active_sessions"]),
            "system_status": "healthy"  # This could be more sophisticated
        }
    
    def clear_cache(self) -> None:
        """Clear the report cache."""
        self.report_cache.clear()
        logger.info("Analytics cache cleared")
