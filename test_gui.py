#!/usr/bin/env python3
"""
MinionsAI v3.1 - Test GUI Application
Simple test version to verify basic functionality.
"""

import streamlit as st
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure Streamlit page
st.set_page_config(
    page_title="MinionsAI v3.1 - Test",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main test application function."""
    st.title("🤖 MinionsAI v3.1 - Test Interface")
    
    st.success("✅ GUI application loaded successfully!")
    
    # Test basic functionality
    st.subheader("🧪 System Test")
    
    # Test imports
    try:
        from core import ChatHandler, SystemMonitor
        st.success("✅ Core modules imported successfully")
    except Exception as e:
        st.error(f"❌ Error importing core modules: {e}")
        return
    
    # Test system monitor
    try:
        monitor = SystemMonitor()
        status = monitor.get_comprehensive_status()
        
        if status.get("overall_health", {}).get("healthy", False):
            st.success("✅ System health check passed")
        else:
            st.warning("⚠️ System health issues detected")
            
    except Exception as e:
        st.error(f"❌ Error checking system status: {e}")
    
    # Test chat handler
    try:
        chat_handler = ChatHandler()
        st.success("✅ Chat handler initialized")
        
        # Test agent initialization
        if st.button("🚀 Test Agent Initialization"):
            with st.spinner("Testing agent initialization..."):
                success = chat_handler.initialize_agent()
                if success:
                    st.success("✅ Agent initialized successfully!")
                    st.balloons()
                else:
                    st.error("❌ Agent initialization failed")
                    
    except Exception as e:
        st.error(f"❌ Error with chat handler: {e}")
    
    # Simple chat interface
    st.subheader("💬 Simple Chat Test")
    
    if "test_messages" not in st.session_state:
        st.session_state.test_messages = []
    
    # Display messages
    for message in st.session_state.test_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Test message..."):
        # Add user message
        st.session_state.test_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Add test response
        with st.chat_message("assistant"):
            response = f"Test response to: {prompt}"
            st.write(response)
            st.session_state.test_messages.append({"role": "assistant", "content": response})
    
    # Sidebar with system info
    with st.sidebar:
        st.header("🔧 Test Controls")
        
        if st.button("🗑️ Clear Test Chat"):
            st.session_state.test_messages = []
            st.rerun()
        
        st.subheader("📊 System Info")
        
        try:
            import psutil
            st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
            memory = psutil.virtual_memory()
            st.metric("Memory Usage", f"{memory.percent}%")
        except Exception as e:
            st.error(f"Error getting system info: {e}")
        
        st.subheader("ℹ️ Test Status")
        st.info("This is a simplified test version of the MinionsAI GUI to verify basic functionality.")

if __name__ == "__main__":
    main()
