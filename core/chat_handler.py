"""
MinionsAI v3.1 - Chat Handler
Manages chat interface logic and conversation flow for the Streamlit GUI.
"""

import streamlit as st
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .agent_manager import AgentManager

logger = logging.getLogger(__name__)


class ChatHandler:
    """
    Handles chat interface logic and manages conversation flow.
    """
    
    def __init__(self):
        """Initialize the Chat Handler."""
        self.agent_manager: Optional[AgentManager] = None
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "agent_initialized" not in st.session_state:
            st.session_state.agent_initialized = False
        
        if "processing" not in st.session_state:
            st.session_state.processing = False
        
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def initialize_agent(self) -> bool:
        """
        Initialize the agent manager.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.agent_manager is None:
                self.agent_manager = AgentManager()
            
            if not st.session_state.agent_initialized:
                success = self.agent_manager.initialize()
                st.session_state.agent_initialized = success
                return success
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing agent: {e}")
            st.error(f"Failed to initialize agent: {e}")
            return False
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the chat history.
        
        Args:
            role: The role of the message sender ('user' or 'assistant')
            content: The message content
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(message)
    
    def display_chat_history(self) -> None:
        """Display the chat history in the Streamlit interface."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def process_user_input(self, user_input: str) -> None:
        """
        Process user input and generate agent response.
        
        Args:
            user_input: The user's input message
        """
        if not user_input.strip():
            return
        
        # Add user message to chat
        self.add_message("user", user_input)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            self._generate_response(user_input)
    
    def _generate_response(self, user_input: str) -> None:
        """
        Generate and display the agent's response with streaming effect.
        
        Args:
            user_input: The user's input message
        """
        if not self.agent_manager or not st.session_state.agent_initialized:
            st.error("Agent not initialized. Please check system status.")
            return
        
        # Show processing indicator
        with st.spinner("MinionsAI is thinking..."):
            st.session_state.processing = True
            
            try:
                # Get response from agent
                response = self.agent_manager.process_message(user_input)
                
                # Display response with typing effect
                message_placeholder = st.empty()
                full_response = ""
                
                # Simulate typing effect
                for chunk in self._chunk_text(response):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)  # Adjust speed as needed
                
                # Final response without cursor
                message_placeholder.markdown(full_response)
                
                # Add to session state
                self.add_message("assistant", response)
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                self.add_message("assistant", error_msg)
            
            finally:
                st.session_state.processing = False
    
    def _chunk_text(self, text: str, chunk_size: int = 3) -> List[str]:
        """
        Split text into chunks for typing effect.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if i + chunk_size < len(words):
                chunk += " "
            chunks.append(chunk)
        
        return chunks
    
    def clear_conversation(self) -> None:
        """Clear the current conversation."""
        st.session_state.messages = []
        if self.agent_manager:
            self.agent_manager.clear_conversation()
        
        # Generate new conversation ID
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        st.success("Conversation cleared!")
        st.rerun()
    
    def export_conversation(self) -> str:
        """
        Export the current conversation as formatted text.
        
        Returns:
            str: Formatted conversation text
        """
        if not st.session_state.messages:
            return "No conversation to export."
        
        export_text = f"MinionsAI Conversation Export\n"
        export_text += f"Conversation ID: {st.session_state.conversation_id}\n"
        export_text += f"Exported at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_text += "=" * 50 + "\n\n"
        
        for i, message in enumerate(st.session_state.messages, 1):
            role = "You" if message["role"] == "user" else "MinionsAI"
            timestamp = message.get("timestamp", "")
            
            export_text += f"[{i}] {role}:\n"
            export_text += f"{message['content']}\n"
            if timestamp:
                export_text += f"Time: {timestamp}\n"
            export_text += "-" * 30 + "\n\n"
        
        return export_text
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current conversation.
        
        Returns:
            Dict containing conversation statistics
        """
        messages = st.session_state.messages
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        
        total_chars = sum(len(m["content"]) for m in messages)
        
        return {
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "total_characters": total_chars,
            "conversation_id": st.session_state.conversation_id,
            "agent_initialized": st.session_state.agent_initialized,
            "currently_processing": st.session_state.processing
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get the current agent status.
        
        Returns:
            Dict containing agent status information
        """
        if self.agent_manager:
            return self.agent_manager.get_status()
        else:
            return {
                "initialized": False,
                "processing": False,
                "message_count": 0,
                "session_id": "N/A",
                "model_name": "N/A",
                "base_url": "N/A"
            }
    
    def shutdown(self) -> None:
        """Shutdown the chat handler and cleanup resources."""
        if self.agent_manager:
            self.agent_manager.shutdown()
        
        # Reset session state
        st.session_state.agent_initialized = False
        st.session_state.processing = False
        
        logger.info("Chat handler shutdown complete")
