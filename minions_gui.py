#!/usr/bin/env python3
"""
MinionsAI v3.1 - Main GUI Application
Streamlit-based desktop GUI for the MinionsAI agent system.
"""

import streamlit as st
import logging
from datetime import datetime
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from core import ChatHandler, SystemMonitor
from utils import get_config, format_timestamp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_app():
    """Initialize the Streamlit application."""
    config = get_config()
    
    # Configure Streamlit page
    st.set_page_config(**config.get_streamlit_config())
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .status-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    
    .status-healthy {
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
    
    .status-warning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
    }
    
    .status-error {
        background-color: #f8d7da;
        border-color: #f5c6cb;
    }
    
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid #ddd;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 MinionsAI v3.1</h1>
        <p>Your Intelligent AI Assistant with Web Search Capabilities</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the application sidebar."""
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # System Status Section
        st.subheader("📊 System Status")
        
        # Initialize system monitor
        if "system_monitor" not in st.session_state:
            st.session_state.system_monitor = SystemMonitor()
        
        monitor = st.session_state.system_monitor
        
        # Quick status check
        with st.spinner("Checking system status..."):
            status = monitor.get_comprehensive_status()
        
        overall_health = status.get("overall_health", {})
        is_healthy = overall_health.get("healthy", False)
        
        if is_healthy:
            st.success("✅ All systems operational")
        else:
            issues = overall_health.get("issues", [])
            st.error(f"❌ Issues: {len(issues)} found")
            with st.expander("View Issues"):
                for issue in issues:
                    st.write(f"• {issue}")
        
        # Quick metrics
        ollama_status = status.get("ollama", {})
        model_status = status.get("model", {})
        
        col1, col2 = st.columns(2)
        with col1:
            if ollama_status.get("running", False):
                st.metric("Ollama", "✅ Running")
            else:
                st.metric("Ollama", "❌ Stopped")
        
        with col2:
            if model_status.get("available", False):
                st.metric("Model", "✅ Ready")
            else:
                st.metric("Model", "❌ N/A")
        
        # Conversation Controls
        st.subheader("💬 Conversation")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            if "chat_handler" in st.session_state:
                st.session_state.chat_handler.clear_conversation()
        
        if st.button("📊 View Stats", use_container_width=True):
            st.session_state.show_stats = not st.session_state.get("show_stats", False)
        
        if st.button("💾 Export Chat", use_container_width=True):
            if "chat_handler" in st.session_state:
                export_text = st.session_state.chat_handler.export_conversation()
                st.download_button(
                    label="📥 Download",
                    data=export_text,
                    file_name=f"minions_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        # Settings Section
        st.subheader("⚙️ Settings")
        
        # Model selection (future enhancement)
        st.selectbox(
            "AI Model",
            ["llama3:8b"],
            disabled=True,
            help="Model selection will be available in future versions"
        )
        
        # Search settings
        search_results = st.slider(
            "Search Results",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of search results to return"
        )
        
        # Advanced settings expander
        with st.expander("🔧 Advanced Settings"):
            st.checkbox("Enable Debug Mode", value=False, disabled=True)
            st.checkbox("Auto-save Conversations", value=True, disabled=True)
            st.slider("Response Speed", 1, 10, 5, disabled=True)
        
        # About section
        st.subheader("ℹ️ About")
        st.info(f"""
        **MinionsAI v3.1**
        
        A sophisticated multi-agent AI system with web search capabilities.
        
        **Features:**
        • LangGraph-powered agent
        • DuckDuckGo web search
        • Real-time conversations
        • System monitoring
        
        **Status:** Phase 2 - GUI Application
        """)


def render_main_content():
    """Render the main content area."""
    try:
        # Initialize chat handler
        if "chat_handler" not in st.session_state:
            st.session_state.chat_handler = ChatHandler()

        chat_handler = st.session_state.chat_handler

        # Initialize agent if not already done
        if not st.session_state.get("agent_initialized", False):
            with st.spinner("🚀 Initializing MinionsAI Agent..."):
                try:
                    success = chat_handler.initialize_agent()
                    if success:
                        st.success("✅ MinionsAI Agent initialized successfully!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to initialize agent. Please check system status.")

                        # Show troubleshooting tips
                        with st.expander("🔧 Troubleshooting Tips"):
                            st.markdown("""
                            **Common issues and solutions:**

                            1. **Ollama not running**: Make sure Ollama is installed and running
                               - Check the System Status tab
                               - Try restarting Ollama service

                            2. **Model not available**: The llama3:8b model might not be downloaded
                               - Run: `ollama pull llama3:8b`
                               - Check available models: `ollama list`

                            3. **Connection issues**: Check if Ollama is accessible
                               - Default URL: http://localhost:11434
                               - Try accessing in browser

                            4. **Memory issues**: Ensure sufficient system resources
                               - Close other applications
                               - Check system memory usage
                            """)
                        return

                except Exception as e:
                    st.error(f"❌ Error during agent initialization: {str(e)}")
                    logger.error(f"Agent initialization error: {e}")

                    # Show detailed error information
                    with st.expander("🐛 Error Details"):
                        st.code(str(e))
                        st.markdown("**Possible solutions:**")
                        st.markdown("- Restart the application")
                        st.markdown("- Check system requirements")
                        st.markdown("- Verify Ollama installation")
                    return

        # Main chat interface
        st.subheader("💬 Chat with MinionsAI")

        # Display conversation stats if requested
        if st.session_state.get("show_stats", False):
            stats = chat_handler.get_conversation_stats()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Messages", stats["total_messages"])
            with col2:
                st.metric("Your Messages", stats["user_messages"])
            with col3:
                st.metric("AI Responses", stats["assistant_messages"])
            with col4:
                st.metric("Characters", stats["total_characters"])

        # Chat container
        chat_container = st.container()

        with chat_container:
            # Display chat history
            chat_handler.display_chat_history()

        # Chat input with error handling
        if prompt := st.chat_input("Ask MinionsAI anything..."):
            try:
                # Validate input
                if len(prompt.strip()) == 0:
                    st.warning("⚠️ Please enter a message")
                    return

                if len(prompt) > 4000:  # Reasonable limit
                    st.error("❌ Message too long. Please keep it under 4000 characters.")
                    return

                # Process the input
                chat_handler.process_user_input(prompt)

            except Exception as e:
                st.error(f"❌ Error processing your message: {str(e)}")
                logger.error(f"Error processing user input: {e}")

                # Offer recovery options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Try Again"):
                        st.rerun()
                with col2:
                    if st.button("🗑️ Clear Chat"):
                        chat_handler.clear_conversation()

    except Exception as e:
        st.error(f"❌ Error in main content: {str(e)}")
        logger.error(f"Main content error: {e}")

        # Show recovery options
        st.markdown("**Recovery Options:**")
        col1, col2, col3 = st.columns(3)
        with col1:
                if st.button("🔄 Refresh Page"):
                    st.rerun()
            with col2:
                if st.button("🔧 Check Status"):
                    st.session_state.show_status = True
            with col3:
                if st.button("🆘 Reset App"):
                    # Clear all session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()


def render_status_page():
    """Render the detailed status page."""
    st.header("🔍 System Status Dashboard")
    
    if "system_monitor" not in st.session_state:
        st.session_state.system_monitor = SystemMonitor()
    
    monitor = st.session_state.system_monitor
    monitor.display_status_dashboard()


def main():
    """Main application function."""
    try:
        # Initialize the app
        initialize_app()
        
        # Render header
        render_header()
        
        # Create main layout
        col1, col2 = st.columns([3, 1])
        
        with col2:
            render_sidebar()
        
        with col1:
            # Navigation tabs
            tab1, tab2 = st.tabs(["💬 Chat", "🔍 System Status"])
            
            with tab1:
                render_main_content()
            
            with tab2:
                render_status_page()
        
        # Footer
        st.markdown("---")
        st.markdown(
            f"<div style='text-align: center; color: #666; padding: 1rem;'>"
            f"MinionsAI v3.1 | Running since {format_timestamp()}"
            f"</div>",
            unsafe_allow_html=True
        )
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"Application error: {e}")
        st.info("Please refresh the page or check the system status.")


if __name__ == "__main__":
    main()
