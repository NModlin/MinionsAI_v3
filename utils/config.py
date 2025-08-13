"""
MinionsAI v3.1 - Configuration Management
Centralized configuration management for the MinionsAI application.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Configuration class for MinionsAI application.
    """
    
    # Application settings
    app_name: str = "MinionsAI"
    app_version: str = "3.1.0"
    app_description: str = "A sophisticated multi-agent AI system"
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"
    ollama_timeout: int = 30
    
    # UI settings
    page_title: str = "MinionsAI v3.1"
    page_icon: str = "🤖"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    
    # Chat settings
    max_message_length: int = 4000
    typing_speed: float = 0.02
    max_conversation_history: int = 100
    
    # Search settings
    search_max_results: int = 5
    search_timeout: int = 10
    
    # System settings
    log_level: str = "INFO"
    cache_duration: int = 30  # seconds
    auto_refresh_interval: int = 60  # seconds
    
    # File paths
    config_file: str = "minions_config.json"
    log_file: str = "minions.log"
    
    @classmethod
    def load_from_file(cls, config_path: Optional[str] = None) -> 'Config':
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Config instance with loaded settings
        """
        if config_path is None:
            config_path = cls().config_file
        
        config = cls()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                
                # Update config with loaded data
                for key, value in data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                
                logger.info(f"Configuration loaded from {config_path}")
                
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
        else:
            logger.info(f"Configuration file {config_path} not found, using defaults")
        
        return config
    
    def save_to_file(self, config_path: Optional[str] = None) -> bool:
        """
        Save configuration to a JSON file.
        
        Args:
            config_path: Path to save the configuration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if config_path is None:
            config_path = self.config_file
        
        try:
            # Convert dataclass to dictionary
            config_dict = {
                key: value for key, value in self.__dict__.items()
                if not key.startswith('_')
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            return False
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """
        Get Streamlit-specific configuration.
        
        Returns:
            Dict containing Streamlit configuration
        """
        return {
            "page_title": self.page_title,
            "page_icon": self.page_icon,
            "layout": self.layout,
            "initial_sidebar_state": self.sidebar_state
        }
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """
        Get Ollama-specific configuration.
        
        Returns:
            Dict containing Ollama configuration
        """
        return {
            "base_url": self.ollama_base_url,
            "model": self.ollama_model,
            "timeout": self.ollama_timeout
        }
    
    def get_chat_config(self) -> Dict[str, Any]:
        """
        Get chat-specific configuration.
        
        Returns:
            Dict containing chat configuration
        """
        return {
            "max_message_length": self.max_message_length,
            "typing_speed": self.typing_speed,
            "max_conversation_history": self.max_conversation_history
        }
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration from a dictionary.
        
        Args:
            updates: Dictionary of configuration updates
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Configuration updated: {key} = {value}")
            else:
                logger.warning(f"Unknown configuration key: {key}")
    
    def validate(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate URLs
            if not self.ollama_base_url.startswith(('http://', 'https://')):
                logger.error("Invalid Ollama base URL format")
                return False
            
            # Validate numeric values
            if self.ollama_timeout <= 0:
                logger.error("Ollama timeout must be positive")
                return False
            
            if self.max_message_length <= 0:
                logger.error("Max message length must be positive")
                return False
            
            if self.typing_speed < 0:
                logger.error("Typing speed cannot be negative")
                return False
            
            if self.max_conversation_history <= 0:
                logger.error("Max conversation history must be positive")
                return False
            
            if self.search_max_results <= 0:
                logger.error("Search max results must be positive")
                return False
            
            # Validate log level
            valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if self.log_level not in valid_log_levels:
                logger.error(f"Invalid log level: {self.log_level}")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return False
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"MinionsAI Config v{self.app_version}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        return f"Config(app_name='{self.app_name}', version='{self.app_version}', model='{self.ollama_model}')"


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config: The global configuration instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config.load_from_file()
    
    return _config_instance


def reload_config(config_path: Optional[str] = None) -> Config:
    """
    Reload the global configuration from file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Config: The reloaded configuration instance
    """
    global _config_instance
    _config_instance = Config.load_from_file(config_path)
    return _config_instance
