"""
MinionsAI v3.1 - System Monitor
Monitors system health, Ollama status, and provides diagnostic information.
"""

import subprocess
import requests
import psutil
import streamlit as st
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Monitors system health and provides diagnostic information.
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize the System Monitor.
        
        Args:
            ollama_url: URL for the Ollama service
        """
        self.ollama_url = ollama_url
        self.last_check_time: Optional[datetime] = None
        self.cached_status: Dict[str, Any] = {}
        self.cache_duration = 30  # seconds
    
    def check_ollama_status(self) -> Tuple[bool, str]:
        """
        Check if Ollama service is running and accessible.
        
        Returns:
            Tuple of (is_running, status_message)
        """
        try:
            # Check if Ollama process is running
            ollama_running = self._is_ollama_process_running()
            
            if not ollama_running:
                return False, "Ollama process not found"
            
            # Check if Ollama API is accessible
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                return True, "Ollama service is running and accessible"
            else:
                return False, f"Ollama API returned status code: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to Ollama API"
        except requests.exceptions.Timeout:
            return False, "Ollama API request timed out"
        except Exception as e:
            return False, f"Error checking Ollama status: {str(e)}"
    
    def _is_ollama_process_running(self) -> bool:
        """Check if Ollama process is running."""
        try:
            for process in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'ollama' in process.info['name'].lower():
                    return True
                
                # Check command line for ollama
                cmdline = process.info.get('cmdline', [])
                if cmdline and any('ollama' in arg.lower() for arg in cmdline):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking Ollama process: {e}")
            return False
    
    def check_model_availability(self, model_name: str = "llama3:8b") -> Tuple[bool, str]:
        """
        Check if the specified model is available in Ollama.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Tuple of (is_available, status_message)
        """
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            
            if response.status_code != 200:
                return False, f"Cannot access Ollama models API (status: {response.status_code})"
            
            data = response.json()
            models = data.get('models', [])
            
            # Check if the model exists
            for model in models:
                if model.get('name', '').startswith(model_name):
                    return True, f"Model '{model_name}' is available"
            
            return False, f"Model '{model_name}' not found. Available models: {[m.get('name', 'Unknown') for m in models]}"
            
        except Exception as e:
            return False, f"Error checking model availability: {str(e)}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dict containing system information
        """
        try:
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk information
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "core_count": cpu_count
                },
                "memory": {
                    "usage_percent": memory_percent,
                    "available_gb": round(memory_available_gb, 2),
                    "total_gb": round(memory_total_gb, 2)
                },
                "disk": {
                    "usage_percent": round(disk_percent, 2),
                    "free_gb": round(disk_free_gb, 2),
                    "total_gb": round(disk_total_gb, 2)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {"error": str(e)}
    
    def get_comprehensive_status(self, model_name: str = "llama3:8b") -> Dict[str, Any]:
        """
        Get comprehensive system and service status.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Dict containing comprehensive status information
        """
        # Check if we can use cached status
        now = datetime.now()
        if (self.last_check_time and 
            (now - self.last_check_time).seconds < self.cache_duration and
            self.cached_status):
            return self.cached_status
        
        status = {}
        
        # Check Ollama status
        ollama_running, ollama_message = self.check_ollama_status()
        status["ollama"] = {
            "running": ollama_running,
            "message": ollama_message
        }
        
        # Check model availability
        if ollama_running:
            model_available, model_message = self.check_model_availability(model_name)
            status["model"] = {
                "available": model_available,
                "message": model_message,
                "name": model_name
            }
        else:
            status["model"] = {
                "available": False,
                "message": "Cannot check model - Ollama not running",
                "name": model_name
            }
        
        # Get system information
        status["system"] = self.get_system_info()
        
        # Overall health check
        status["overall_health"] = {
            "healthy": ollama_running and status["model"]["available"],
            "issues": []
        }
        
        if not ollama_running:
            status["overall_health"]["issues"].append("Ollama service not running")
        
        if not status["model"]["available"]:
            status["overall_health"]["issues"].append(f"Model '{model_name}' not available")
        
        # Check system resources
        system_info = status["system"]
        if not isinstance(system_info, dict) or "error" in system_info:
            status["overall_health"]["issues"].append("Cannot read system information")
        else:
            if system_info.get("memory", {}).get("usage_percent", 0) > 90:
                status["overall_health"]["issues"].append("High memory usage (>90%)")
            
            if system_info.get("disk", {}).get("usage_percent", 0) > 90:
                status["overall_health"]["issues"].append("High disk usage (>90%)")
        
        # Cache the results
        self.cached_status = status
        self.last_check_time = now
        
        return status
    
    def display_status_dashboard(self, model_name: str = "llama3:8b") -> None:
        """
        Display a comprehensive status dashboard in Streamlit.
        
        Args:
            model_name: Name of the model to check
        """
        st.subheader("🔍 System Status Dashboard")
        
        # Get status information
        status = self.get_comprehensive_status(model_name)
        
        # Overall health indicator
        overall_health = status.get("overall_health", {})
        is_healthy = overall_health.get("healthy", False)
        
        if is_healthy:
            st.success("✅ All systems operational")
        else:
            issues = overall_health.get("issues", [])
            st.error(f"❌ Issues detected: {', '.join(issues)}")
        
        # Create columns for different status sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 AI Services")
            
            # Ollama status
            ollama_status = status.get("ollama", {})
            if ollama_status.get("running", False):
                st.success(f"✅ Ollama: {ollama_status.get('message', 'Running')}")
            else:
                st.error(f"❌ Ollama: {ollama_status.get('message', 'Not running')}")
            
            # Model status
            model_status = status.get("model", {})
            if model_status.get("available", False):
                st.success(f"✅ Model ({model_name}): Available")
            else:
                st.error(f"❌ Model ({model_name}): {model_status.get('message', 'Not available')}")
        
        with col2:
            st.subheader("💻 System Resources")
            
            system_info = status.get("system", {})
            if "error" not in system_info:
                # CPU
                cpu_usage = system_info.get("cpu", {}).get("usage_percent", 0)
                st.metric("CPU Usage", f"{cpu_usage}%")
                
                # Memory
                memory_info = system_info.get("memory", {})
                memory_usage = memory_info.get("usage_percent", 0)
                memory_available = memory_info.get("available_gb", 0)
                st.metric("Memory Usage", f"{memory_usage}%", f"{memory_available}GB available")
                
                # Disk
                disk_info = system_info.get("disk", {})
                disk_usage = disk_info.get("usage_percent", 0)
                disk_free = disk_info.get("free_gb", 0)
                st.metric("Disk Usage", f"{disk_usage}%", f"{disk_free}GB free")
            else:
                st.error(f"Error reading system info: {system_info.get('error', 'Unknown error')}")
        
        # Refresh button
        if st.button("🔄 Refresh Status", key="refresh_status"):
            self.cached_status = {}  # Clear cache to force refresh
            st.rerun()
        
        # Last updated timestamp
        if self.last_check_time:
            st.caption(f"Last updated: {self.last_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def start_ollama_service(self) -> Tuple[bool, str]:
        """
        Attempt to start the Ollama service.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Try to start Ollama service
            result = subprocess.run(
                ["ollama", "serve"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return True, "Ollama service started successfully"
            else:
                return False, f"Failed to start Ollama: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return True, "Ollama service start initiated (running in background)"
        except FileNotFoundError:
            return False, "Ollama command not found. Please ensure Ollama is installed."
        except Exception as e:
            return False, f"Error starting Ollama service: {str(e)}"
