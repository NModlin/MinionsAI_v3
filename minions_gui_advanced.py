#!/usr/bin/env python3
"""
MinionsAI v3.1 - Advanced GUI Application
Enhanced Streamlit GUI with advanced features including analytics, performance monitoring,
multi-model support, and enterprise capabilities.
"""

import streamlit as st
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Core imports
from core import AgentManager, ChatHandler, SystemMonitor
from utils import Config, get_config

# Advanced features imports
from core.advanced.analytics import EventType, MetricType
from core.advanced.security import UserRole, Permission
from core.advanced.models import ModelCapability

# Configure Streamlit page
st.set_page_config(
    page_title="MinionsAI v3.1 Advanced",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-healthy { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }
    
    .advanced-sidebar {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


class AdvancedGUI:
    """Advanced GUI application with enterprise features."""
    
    def __init__(self):
        """Initialize the advanced GUI."""
        self.config = get_config()
        self.agent_manager: Optional[AgentManager] = None
        self.chat_handler: Optional[ChatHandler] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.current_user: Optional[str] = None
        self.session_token: Optional[str] = None
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "agent_initialized" not in st.session_state:
            st.session_state.agent_initialized = False
        
        if "multi_agent_mode" not in st.session_state:
            st.session_state.multi_agent_mode = False
        
        if "advanced_features" not in st.session_state:
            st.session_state.advanced_features = True
        
        if "current_page" not in st.session_state:
            st.session_state.current_page = "Chat"
        
        if "user_authenticated" not in st.session_state:
            st.session_state.user_authenticated = False
        
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "llama3:8b"
        
        if "theme" not in st.session_state:
            st.session_state.theme = "default"
    
    def render_header(self):
        """Render the main header."""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 MinionsAI v3.1 Advanced</h1>
            <p>Sophisticated Multi-Agent AI System with Enterprise Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the advanced sidebar."""
        with st.sidebar:
            st.markdown('<div class="advanced-sidebar">', unsafe_allow_html=True)
            
            # Navigation
            st.subheader("🧭 Navigation")
            pages = ["Chat", "Analytics", "Performance", "Models", "Tools", "Security", "Settings"]
            st.session_state.current_page = st.selectbox("Select Page", pages, 
                                                        index=pages.index(st.session_state.current_page))
            
            st.markdown("---")
            
            # System Status
            st.subheader("📊 System Status")
            self._render_system_status()
            
            st.markdown("---")
            
            # Quick Settings
            st.subheader("⚙️ Quick Settings")
            self._render_quick_settings()
            
            st.markdown("---")
            
            # User Info
            if st.session_state.user_authenticated:
                st.subheader("👤 User Info")
                st.write(f"**User:** {self.current_user}")
                if st.button("Logout"):
                    self._logout()
            else:
                if st.button("Login"):
                    st.session_state.current_page = "Security"
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_system_status(self):
        """Render system status indicators."""
        if self.agent_manager and st.session_state.agent_initialized:
            status = self.agent_manager.get_status()
            
            # Agent Status
            agent_status = "🟢 Healthy" if status["initialized"] else "🔴 Error"
            st.write(f"**Agent:** {agent_status}")
            
            # Multi-Agent Status
            if status.get("multi_agent_mode"):
                agent_count = len(status.get("available_agents", []))
                st.write(f"**Agents:** {agent_count} active")
            
            # Performance Status
            if self.agent_manager.performance_monitor:
                metrics = self.agent_manager.performance_monitor.collect_metrics()
                cpu_status = "🟢" if metrics.cpu_usage < 80 else "🟡" if metrics.cpu_usage < 95 else "🔴"
                st.write(f"**CPU:** {cpu_status} {metrics.cpu_usage:.1f}%")
                
                memory_status = "🟢" if metrics.memory_usage < 80 else "🟡" if metrics.memory_usage < 95 else "🔴"
                st.write(f"**Memory:** {memory_status} {metrics.memory_usage:.1f}%")
        else:
            st.write("**Agent:** 🔴 Not initialized")
    
    def _render_quick_settings(self):
        """Render quick settings controls."""
        # Multi-agent toggle
        new_multi_agent = st.checkbox("Multi-Agent Mode", value=st.session_state.multi_agent_mode)
        if new_multi_agent != st.session_state.multi_agent_mode:
            st.session_state.multi_agent_mode = new_multi_agent
            if self.agent_manager:
                if new_multi_agent:
                    self.agent_manager.enable_multi_agent_mode()
                else:
                    self.agent_manager.disable_multi_agent_mode()
        
        # Advanced features toggle
        st.session_state.advanced_features = st.checkbox("Advanced Features", 
                                                         value=st.session_state.advanced_features)
        
        # Theme selection
        themes = ["default", "dark", "light", "blue"]
        st.session_state.theme = st.selectbox("Theme", themes, 
                                             index=themes.index(st.session_state.theme))
        
        # Model selection
        if self.agent_manager and self.agent_manager.model_manager:
            available_models = self.agent_manager.model_manager.get_available_models()
            if available_models:
                st.session_state.selected_model = st.selectbox("Model", available_models,
                                                              index=0 if st.session_state.selected_model not in available_models 
                                                              else available_models.index(st.session_state.selected_model))
    
    def render_chat_page(self):
        """Render the main chat interface."""
        st.header("💬 Chat Interface")
        
        # Initialize agent if needed
        if not st.session_state.agent_initialized:
            if st.button("🚀 Initialize MinionsAI", type="primary"):
                self._initialize_agent()
        
        if not st.session_state.agent_initialized:
            st.info("Click 'Initialize MinionsAI' to start the agent system.")
            return
        
        # Agent selection for multi-agent mode
        selected_agent = None
        if st.session_state.multi_agent_mode and self.agent_manager:
            available_agents = self.agent_manager.get_available_agents()
            if available_agents:
                agent_options = ["Auto-select"] + [agent["name"] for agent in available_agents]
                selected_agent_name = st.selectbox("Select Agent", agent_options)
                if selected_agent_name != "Auto-select":
                    selected_agent = next((a["type"] for a in available_agents if a["name"] == selected_agent_name), None)
        
        # Chat interface
        self._render_chat_interface(selected_agent)
    
    def _render_chat_interface(self, selected_agent: Optional[str] = None):
        """Render the chat interface."""
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show metadata for advanced users
                if st.session_state.advanced_features and "metadata" in message:
                    with st.expander("Message Details"):
                        st.json(message["metadata"])
        
        # Chat input
        if prompt := st.chat_input("What can I help you with?"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self._get_agent_response(prompt, selected_agent)
                    st.markdown(response)
                    
                    # Add assistant message with metadata
                    message_data = {"role": "assistant", "content": response}
                    if st.session_state.advanced_features:
                        message_data["metadata"] = {
                            "timestamp": datetime.now().isoformat(),
                            "agent": selected_agent or "auto",
                            "model": st.session_state.selected_model,
                            "response_time": "calculated_elsewhere"  # Would be calculated in real implementation
                        }
                    
                    st.session_state.messages.append(message_data)
    
    def render_analytics_page(self):
        """Render the analytics dashboard."""
        st.header("📊 Analytics Dashboard")
        
        if not self.agent_manager or not self.agent_manager.analytics_engine:
            st.warning("Analytics engine not available. Enable advanced features to access analytics.")
            return
        
        # Time period selection
        col1, col2 = st.columns(2)
        with col1:
            period_days = st.selectbox("Time Period", [1, 7, 30, 90], index=1)
        with col2:
            if st.button("Refresh Data"):
                st.rerun()
        
        # Generate reports
        usage_report = self.agent_manager.analytics_engine.generate_usage_report(period_days)
        performance_report = self.agent_manager.analytics_engine.generate_performance_report(period_days * 24)
        
        # Usage metrics
        st.subheader("📈 Usage Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Events", usage_report["total_events"])
        with col2:
            st.metric("Unique Users", usage_report["unique_users"])
        with col3:
            st.metric("Avg Response Time", f"{usage_report['avg_response_time']:.2f}s")
        with col4:
            error_rate = (usage_report.get("error_count", 0) / max(usage_report["total_events"], 1)) * 100
            st.metric("Error Rate", f"{error_rate:.1f}%")
        
        # Event breakdown chart
        if usage_report["event_breakdown"]:
            st.subheader("📊 Event Breakdown")
            event_df = pd.DataFrame(list(usage_report["event_breakdown"].items()), 
                                   columns=["Event Type", "Count"])
            fig = px.pie(event_df, values="Count", names="Event Type", 
                        title="Distribution of Events")
            st.plotly_chart(fig, use_container_width=True)
        
        # Daily activity chart
        if usage_report["daily_activity"]:
            st.subheader("📅 Daily Activity")
            daily_df = pd.DataFrame(list(usage_report["daily_activity"].items()), 
                                   columns=["Date", "Activity"])
            daily_df["Date"] = pd.to_datetime(daily_df["Date"])
            fig = px.line(daily_df, x="Date", y="Activity", 
                         title="Daily Activity Trend")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_page(self):
        """Render the performance monitoring page."""
        st.header("⚡ Performance Monitoring")
        
        if not self.agent_manager or not self.agent_manager.performance_monitor:
            st.warning("Performance monitor not available. Enable advanced features to access performance monitoring.")
            return
        
        # Real-time metrics
        st.subheader("🔴 Real-time Metrics")
        
        if st.button("Refresh Metrics"):
            st.rerun()
        
        # Get current metrics
        current_metrics = self.agent_manager.performance_monitor.collect_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("CPU Usage", f"{current_metrics.cpu_usage:.1f}%")
        with col2:
            st.metric("Memory Usage", f"{current_metrics.memory_usage:.1f}%")
        with col3:
            st.metric("Cache Hit Rate", f"{current_metrics.cache_hit_rate:.1%}")
        with col4:
            st.metric("Active Tasks", current_metrics.active_tasks)
        
        # Performance summary
        st.subheader("📊 Performance Summary")
        summary = self.agent_manager.performance_monitor.get_metrics_summary(60)  # Last hour
        
        if "error" not in summary:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**System Performance**")
                st.write(f"Uptime: {summary['uptime_seconds'] / 3600:.1f} hours")
                st.write(f"Total Requests: {summary['total_requests']}")
                st.write(f"Error Rate: {summary['error_rate']:.1%}")
            
            with col2:
                st.write("**Resource Usage**")
                st.write(f"Avg CPU: {summary['avg_cpu_usage']:.1f}%")
                st.write(f"Avg Memory: {summary['avg_memory_usage']:.1f}%")
                st.write(f"Cache Hit Rate: {summary['avg_cache_hit_rate']:.1%}")
    
    def render_models_page(self):
        """Render the model management page."""
        st.header("🧠 Model Management")
        
        if not self.agent_manager or not self.agent_manager.model_manager:
            st.warning("Model manager not available. Enable advanced features to access model management.")
            return
        
        # Model statistics
        st.subheader("📊 Model Statistics")
        model_stats = self.agent_manager.model_manager.get_model_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Models", model_stats["total_models"])
        with col2:
            st.metric("Available Models", model_stats["available_models"])
        with col3:
            st.metric("Default Model", model_stats["default_model"] or "None")
        with col4:
            st.metric("Selection Strategy", model_stats["selection_strategy"])
        
        # Available models
        st.subheader("🔧 Available Models")
        available_models = self.agent_manager.model_manager.get_available_models()
        
        if available_models:
            for model_name in available_models:
                model_info = self.agent_manager.model_manager.get_model_info(model_name)
                if model_info:
                    with st.expander(f"📋 {model_name}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Type:** {model_info.model_type.value}")
                            st.write(f"**Context Length:** {model_info.context_length}")
                            st.write(f"**Parameters:** {model_info.parameters or 'Unknown'}")
                        with col2:
                            st.write(f"**Performance Score:** {model_info.performance_score:.2f}")
                            st.write(f"**Avg Response Time:** {model_info.avg_response_time:.2f}s")
                            st.write(f"**Success Rate:** {model_info.success_rate:.1%}")
                        
                        st.write(f"**Capabilities:** {', '.join([cap.value for cap in model_info.capabilities])}")
                        st.write(f"**Usage Count:** {model_info.usage_count}")
        else:
            st.info("No models available.")
    
    def render_tools_page(self):
        """Render the tools management page."""
        st.header("🛠️ Tool Management")
        
        if not self.agent_manager or not self.agent_manager.tool_manager:
            st.warning("Tool manager not available. Enable advanced features to access tool management.")
            return
        
        # Tool statistics
        st.subheader("📊 Tool Statistics")
        tool_stats = self.agent_manager.tool_manager.get_tool_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tools", tool_stats["total_tools"])
        with col2:
            st.metric("Built-in Tools", tool_stats["builtin_tools"])
        with col3:
            st.metric("Custom Tools", tool_stats["custom_tools"])
        with col4:
            st.metric("Enabled Tools", tool_stats["enabled_tools"])
        
        # Available tools by category
        st.subheader("🔧 Available Tools")
        categories = self.agent_manager.tool_manager.get_tool_categories()
        
        for category, tools in categories.items():
            with st.expander(f"📁 {category.title()} ({len(tools)} tools)"):
                for tool_name in tools:
                    tool_list = self.agent_manager.tool_manager.list_tools(category)
                    tool_info = next((t for t in tool_list if t["name"] == tool_name), None)
                    
                    if tool_info:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{tool_info['name']}** - {tool_info['description']}")
                            st.write(f"Usage: {tool_info['usage_count']} times")
                        with col2:
                            status = "🟢 Enabled" if tool_info["is_enabled"] else "🔴 Disabled"
                            st.write(status)
    
    def render_security_page(self):
        """Render the security management page."""
        st.header("🔒 Security Management")
        
        if not self.agent_manager or not self.agent_manager.security_manager:
            st.warning("Security manager not available. Enable advanced features to access security management.")
            return
        
        # Authentication
        if not st.session_state.user_authenticated:
            st.subheader("🔐 Login")
            
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                
                if submitted and username and password:
                    token = self.agent_manager.auth_manager.login(username, password)
                    if token:
                        st.session_state.user_authenticated = True
                        self.current_user = username
                        self.session_token = token
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        else:
            # Security dashboard
            st.subheader("🛡️ Security Dashboard")
            
            # Recent security events
            security_events = self.agent_manager.security_manager.get_security_events(24)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Security Events (24h)", len(security_events))
            with col2:
                failed_logins = len([e for e in security_events if e["event_type"] == "login_failed"])
                st.metric("Failed Logins", failed_logins)
            with col3:
                active_users = len(set(e["username"] for e in security_events if e["username"]))
                st.metric("Active Users", active_users)
            
            # User management (admin only)
            if self._has_permission(Permission.MANAGE_USERS):
                st.subheader("👥 User Management")
                users = self.agent_manager.security_manager.list_users()
                
                for user in users:
                    with st.expander(f"👤 {user['username']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Email:** {user['email']}")
                            st.write(f"**Role:** {user['role']}")
                            st.write(f"**Created:** {user['created_at'][:10]}")
                        with col2:
                            status = "🟢 Active" if user["is_active"] else "🔴 Inactive"
                            st.write(f"**Status:** {status}")
                            st.write(f"**Last Login:** {user['last_login'][:10] if user['last_login'] else 'Never'}")
    
    def render_settings_page(self):
        """Render the settings page."""
        st.header("⚙️ Settings")
        
        # System settings
        st.subheader("🖥️ System Settings")
        
        with st.form("system_settings"):
            # Agent settings
            st.write("**Agent Configuration**")
            model_name = st.text_input("Model Name", value=self.config.ollama_model)
            base_url = st.text_input("Ollama Base URL", value=self.config.ollama_base_url)
            
            # Performance settings
            st.write("**Performance Settings**")
            cache_size = st.number_input("Cache Size", min_value=100, max_value=10000, value=1000)
            max_concurrent_tasks = st.number_input("Max Concurrent Tasks", min_value=1, max_value=50, value=10)
            
            # UI settings
            st.write("**UI Settings**")
            auto_refresh = st.checkbox("Auto-refresh Analytics", value=True)
            show_debug_info = st.checkbox("Show Debug Information", value=False)
            
            submitted = st.form_submit_button("Save Settings")
            
            if submitted:
                # Update configuration
                self.config.ollama_model = model_name
                self.config.ollama_base_url = base_url
                st.success("Settings saved successfully!")
        
        # Export/Import
        st.subheader("💾 Data Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export Conversations"):
                # Export conversation data
                export_data = {
                    "conversations": st.session_state.messages,
                    "exported_at": datetime.now().isoformat()
                }
                st.download_button(
                    "Download Export",
                    json.dumps(export_data, indent=2),
                    file_name=f"minionsai_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_file = st.file_uploader("Import Conversations", type="json")
            if uploaded_file:
                try:
                    import_data = json.load(uploaded_file)
                    if "conversations" in import_data:
                        st.session_state.messages = import_data["conversations"]
                        st.success("Conversations imported successfully!")
                except Exception as e:
                    st.error(f"Import failed: {e}")
    
    def _initialize_agent(self):
        """Initialize the agent system."""
        try:
            with st.spinner("Initializing MinionsAI..."):
                self.agent_manager = AgentManager(
                    model_name=st.session_state.selected_model,
                    multi_agent_mode=st.session_state.multi_agent_mode,
                    enable_advanced_features=st.session_state.advanced_features
                )
                
                success = self.agent_manager.initialize()
                
                if success:
                    self.chat_handler = ChatHandler(self.agent_manager)
                    self.system_monitor = SystemMonitor()
                    st.session_state.agent_initialized = True
                    st.success("✅ MinionsAI initialized successfully!")
                else:
                    st.error("❌ Failed to initialize MinionsAI")
        
        except Exception as e:
            st.error(f"❌ Initialization error: {e}")
    
    def _get_agent_response(self, message: str, agent_type: Optional[str] = None) -> str:
        """Get response from the agent."""
        if not self.agent_manager:
            return "Agent not initialized"
        
        try:
            return self.agent_manager.process_message(
                message, 
                agent_type=agent_type,
                user_id=self.current_user,
                session_id=st.session_state.get("session_id")
            )
        except Exception as e:
            return f"Error: {e}"
    
    def _has_permission(self, permission: Permission) -> bool:
        """Check if current user has permission."""
        if not self.session_token or not self.agent_manager:
            return False
        
        return self.agent_manager.security_manager.check_permission(self.session_token, permission)
    
    def _logout(self):
        """Logout current user."""
        if self.session_token and self.agent_manager:
            self.agent_manager.auth_manager.logout(self.session_token)
        
        st.session_state.user_authenticated = False
        self.current_user = None
        self.session_token = None
        st.rerun()
    
    def run(self):
        """Run the advanced GUI application."""
        self.render_header()
        self.render_sidebar()
        
        # Route to appropriate page
        if st.session_state.current_page == "Chat":
            self.render_chat_page()
        elif st.session_state.current_page == "Analytics":
            self.render_analytics_page()
        elif st.session_state.current_page == "Performance":
            self.render_performance_page()
        elif st.session_state.current_page == "Models":
            self.render_models_page()
        elif st.session_state.current_page == "Tools":
            self.render_tools_page()
        elif st.session_state.current_page == "Security":
            self.render_security_page()
        elif st.session_state.current_page == "Settings":
            self.render_settings_page()


def main():
    """Main application entry point."""
    app = AdvancedGUI()
    app.run()


if __name__ == "__main__":
    main()
