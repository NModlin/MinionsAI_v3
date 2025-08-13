"""
MinionsAI v3.1 - Helper Functions
Utility functions for the MinionsAI application.
"""

import re
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def format_timestamp(timestamp: Optional[str] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a timestamp string or current time.
    
    Args:
        timestamp: ISO format timestamp string, or None for current time
        format_str: Format string for the output
        
    Returns:
        str: Formatted timestamp string
    """
    try:
        if timestamp is None:
            dt = datetime.now()
        else:
            # Try to parse ISO format timestamp
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        return dt.strftime(format_str)
        
    except Exception as e:
        logger.error(f"Error formatting timestamp: {e}")
        return "Invalid timestamp"


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: The filename to sanitize
        max_length: Maximum length for the filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    
    # Trim whitespace and dots from ends
    sanitized = sanitized.strip(' .')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "untitled"
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def get_app_version() -> str:
    """
    Get the application version.
    
    Returns:
        str: Application version string
    """
    return "3.1.0"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def count_words(text: str) -> int:
    """
    Count words in a text string.
    
    Args:
        text: Text to count words in
        
    Returns:
        int: Number of words
    """
    return len(text.split())


def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Estimate reading time for text.
    
    Args:
        text: Text to estimate reading time for
        words_per_minute: Average reading speed
        
    Returns:
        int: Estimated reading time in minutes
    """
    word_count = count_words(text)
    return max(1, round(word_count / words_per_minute))


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL.
    
    Args:
        url: URL string to validate
        
    Returns:
        bool: True if valid URL, False otherwise
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None


def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Text to extract URLs from
        
    Returns:
        List[str]: List of URLs found in the text
    """
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    return url_pattern.findall(text)


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    
    Args:
        text: Text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def create_conversation_summary(messages: List[Dict[str, Any]], max_length: int = 200) -> str:
    """
    Create a summary of a conversation.
    
    Args:
        messages: List of conversation messages
        max_length: Maximum length of summary
        
    Returns:
        str: Conversation summary
    """
    if not messages:
        return "Empty conversation"
    
    # Get first user message as topic
    first_user_msg = None
    for msg in messages:
        if msg.get("role") == "user":
            first_user_msg = msg.get("content", "")
            break
    
    if not first_user_msg:
        return "No user messages found"
    
    # Create summary
    summary = f"Conversation about: {first_user_msg}"
    
    # Add message count
    user_count = sum(1 for msg in messages if msg.get("role") == "user")
    assistant_count = sum(1 for msg in messages if msg.get("role") == "assistant")
    
    summary += f" ({user_count} user messages, {assistant_count} responses)"
    
    return truncate_text(summary, max_length)


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string with fallback.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        import json
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return default


def get_system_info_summary() -> Dict[str, str]:
    """
    Get a summary of system information.
    
    Returns:
        Dict containing system info summary
    """
    try:
        import platform
        import psutil
        
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "cpu_count": str(psutil.cpu_count()),
            "memory_gb": f"{psutil.virtual_memory().total / (1024**3):.1f}",
            "disk_gb": f"{psutil.disk_usage('/').total / (1024**3):.1f}"
        }
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {"error": str(e)}


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        return False


def is_port_available(port: int, host: str = "localhost") -> bool:
    """
    Check if a port is available.
    
    Args:
        port: Port number to check
        host: Host to check on
        
    Returns:
        bool: True if port is available, False otherwise
    """
    try:
        import socket
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0  # Port is available if connection fails
            
    except Exception as e:
        logger.error(f"Error checking port {port}: {e}")
        return False
