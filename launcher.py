#!/usr/bin/env python3
"""
MinionsAI v3.1 - One-Click Launcher
Launches all necessary services and starts the GUI application.
"""

import os
import sys
import subprocess
import time
import webbrowser
import logging
from typing import Tuple, Optional
import requests
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3:8b"
STREAMLIT_PORT = 8501
STREAMLIT_HOST = "localhost"
GUI_SCRIPT = "minions_gui.py"

# Colors for console output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_colored(message: str, color: str = Colors.END) -> None:
    """Print colored message to console."""
    print(f"{color}{message}{Colors.END}")


def print_header() -> None:
    """Print application header."""
    print_colored("=" * 60, Colors.BLUE)
    print_colored("🤖 MinionsAI v3.1 - One-Click Launcher", Colors.BOLD)
    print_colored("Starting your AI assistant...", Colors.BLUE)
    print_colored("=" * 60, Colors.BLUE)
    print()


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_colored("❌ Python 3.8+ is required", Colors.RED)
        print_colored(f"Current version: {version.major}.{version.minor}.{version.micro}", Colors.YELLOW)
        return False
    
    print_colored(f"✅ Python {version.major}.{version.minor}.{version.micro} detected", Colors.GREEN)
    return True


def check_dependencies() -> bool:
    """Check if required Python packages are installed."""
    required_packages = [
        "streamlit",
        "langchain",
        "langgraph", 
        "langchain_ollama",
        "duckduckgo_search",
        "psutil",
        "requests"
    ]
    
    print_colored("🔍 Checking Python dependencies...", Colors.BLUE)
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print_colored(f"  ✅ {package}", Colors.GREEN)
        except ImportError:
            print_colored(f"  ❌ {package}", Colors.RED)
            missing_packages.append(package)
    
    if missing_packages:
        print_colored(f"\n❌ Missing packages: {', '.join(missing_packages)}", Colors.RED)
        print_colored("Please install missing packages with:", Colors.YELLOW)
        print_colored(f"pip install {' '.join(missing_packages)}", Colors.YELLOW)
        return False
    
    print_colored("✅ All dependencies satisfied", Colors.GREEN)
    return True


def is_ollama_running() -> bool:
    """Check if Ollama service is running."""
    try:
        # Check if Ollama process is running
        for process in psutil.process_iter(['pid', 'name', 'cmdline']):
            if 'ollama' in process.info['name'].lower():
                return True
            
            cmdline = process.info.get('cmdline', [])
            if cmdline and any('ollama' in arg.lower() for arg in cmdline):
                return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama process: {e}")
        return False


def check_ollama_api() -> bool:
    """Check if Ollama API is accessible."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def start_ollama() -> bool:
    """Start Ollama service if not running."""
    print_colored("🚀 Starting Ollama service...", Colors.BLUE)
    
    if is_ollama_running() and check_ollama_api():
        print_colored("✅ Ollama is already running", Colors.GREEN)
        return True
    
    try:
        # Try to start Ollama serve in background
        if os.name == 'nt':  # Windows
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:  # Unix-like systems
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        # Wait for service to start
        print_colored("⏳ Waiting for Ollama to start...", Colors.YELLOW)
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            if check_ollama_api():
                print_colored("✅ Ollama service started successfully", Colors.GREEN)
                return True
            print(".", end="", flush=True)
        
        print()
        print_colored("❌ Ollama service failed to start within 30 seconds", Colors.RED)
        return False
        
    except FileNotFoundError:
        print_colored("❌ Ollama not found. Please install Ollama first.", Colors.RED)
        print_colored("Visit: https://ollama.com/download", Colors.YELLOW)
        return False
    except Exception as e:
        print_colored(f"❌ Error starting Ollama: {e}", Colors.RED)
        return False


def check_model() -> bool:
    """Check if the required model is available."""
    print_colored(f"🔍 Checking for model: {OLLAMA_MODEL}", Colors.BLUE)
    
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code != 200:
            print_colored("❌ Cannot access Ollama models API", Colors.RED)
            return False
        
        data = response.json()
        models = data.get('models', [])
        
        for model in models:
            if model.get('name', '').startswith(OLLAMA_MODEL):
                print_colored(f"✅ Model {OLLAMA_MODEL} is available", Colors.GREEN)
                return True
        
        print_colored(f"❌ Model {OLLAMA_MODEL} not found", Colors.RED)
        return False
        
    except Exception as e:
        print_colored(f"❌ Error checking model: {e}", Colors.RED)
        return False


def pull_model() -> bool:
    """Pull the required model if not available."""
    print_colored(f"📥 Pulling model: {OLLAMA_MODEL}", Colors.BLUE)
    print_colored("This may take several minutes...", Colors.YELLOW)
    
    try:
        process = subprocess.run(
            ["ollama", "pull", OLLAMA_MODEL],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        if process.returncode == 0:
            print_colored(f"✅ Model {OLLAMA_MODEL} pulled successfully", Colors.GREEN)
            return True
        else:
            print_colored(f"❌ Failed to pull model: {process.stderr}", Colors.RED)
            return False
            
    except subprocess.TimeoutExpired:
        print_colored("❌ Model pull timed out", Colors.RED)
        return False
    except Exception as e:
        print_colored(f"❌ Error pulling model: {e}", Colors.RED)
        return False


def is_port_available(port: int) -> bool:
    """Check if a port is available."""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((STREAMLIT_HOST, port))
            return result != 0
    except Exception:
        return False


def start_streamlit() -> Optional[subprocess.Popen]:
    """Start the Streamlit application."""
    print_colored("🌐 Starting MinionsAI GUI...", Colors.BLUE)
    
    if not os.path.exists(GUI_SCRIPT):
        print_colored(f"❌ GUI script not found: {GUI_SCRIPT}", Colors.RED)
        return None
    
    if not is_port_available(STREAMLIT_PORT):
        print_colored(f"❌ Port {STREAMLIT_PORT} is already in use", Colors.RED)
        return None
    
    try:
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", GUI_SCRIPT,
            "--server.port", str(STREAMLIT_PORT),
            "--server.address", STREAMLIT_HOST,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
        # Wait a moment for Streamlit to start
        time.sleep(3)
        
        if process.poll() is None:  # Process is still running
            print_colored("✅ MinionsAI GUI started successfully", Colors.GREEN)
            return process
        else:
            print_colored("❌ Failed to start MinionsAI GUI", Colors.RED)
            return None
            
    except Exception as e:
        print_colored(f"❌ Error starting GUI: {e}", Colors.RED)
        return None


def open_browser() -> None:
    """Open the application in the default browser."""
    url = f"http://{STREAMLIT_HOST}:{STREAMLIT_PORT}"
    print_colored(f"🌐 Opening browser: {url}", Colors.BLUE)
    
    try:
        webbrowser.open(url)
        print_colored("✅ Browser opened successfully", Colors.GREEN)
    except Exception as e:
        print_colored(f"❌ Could not open browser: {e}", Colors.RED)
        print_colored(f"Please manually open: {url}", Colors.YELLOW)


def main() -> None:
    """Main launcher function."""
    print_header()
    
    # Check system requirements
    if not check_python_version():
        return
    
    if not check_dependencies():
        return
    
    # Start Ollama service
    if not start_ollama():
        print_colored("❌ Cannot proceed without Ollama service", Colors.RED)
        return
    
    # Check/pull model
    if not check_model():
        print_colored("📥 Model not found, attempting to pull...", Colors.YELLOW)
        if not pull_model():
            print_colored("❌ Cannot proceed without the required model", Colors.RED)
            return
    
    # Start GUI
    streamlit_process = start_streamlit()
    if not streamlit_process:
        print_colored("❌ Failed to start GUI application", Colors.RED)
        return
    
    # Open browser
    time.sleep(2)  # Give Streamlit more time to fully start
    open_browser()
    
    # Success message
    print()
    print_colored("🎉 MinionsAI is now running!", Colors.GREEN)
    print_colored(f"🌐 Access URL: http://{STREAMLIT_HOST}:{STREAMLIT_PORT}", Colors.BLUE)
    print_colored("Press Ctrl+C to stop the application", Colors.YELLOW)
    print()
    
    try:
        # Keep the launcher running
        streamlit_process.wait()
    except KeyboardInterrupt:
        print_colored("\n🛑 Shutting down MinionsAI...", Colors.YELLOW)
        streamlit_process.terminate()
        print_colored("✅ MinionsAI stopped successfully", Colors.GREEN)


if __name__ == "__main__":
    main()
