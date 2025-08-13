#!/usr/bin/env python3
"""
MinionsAI v3.1 - Installation Verification Script
Quick verification script to check if MinionsAI is properly installed and configured.
"""

import sys
import os
import subprocess
import requests
import importlib
from pathlib import Path
import json

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(text: str, color: str) -> None:
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.END}")

def print_header():
    """Print the verification header."""
    print_colored("🔍 MinionsAI v3.1 - Installation Verification", Colors.BOLD)
    print_colored("=" * 50, Colors.BLUE)

def check_python_version():
    """Check if Python version is compatible."""
    print_colored("\n📋 Checking Python Version...", Colors.BLUE)
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_colored(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible", Colors.GREEN)
        return True
    else:
        print_colored(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+", Colors.RED)
        return False

def check_dependencies():
    """Check if required Python packages are installed."""
    print_colored("\n📋 Checking Python Dependencies...", Colors.BLUE)
    
    required_packages = [
        'streamlit',
        'requests', 
        'langchain',
        'langchain_ollama',
        'duckduckgo_search',
        'langgraph',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print_colored(f"✅ {package}", Colors.GREEN)
        except ImportError:
            print_colored(f"❌ {package} - Missing", Colors.RED)
            missing_packages.append(package)
    
    if missing_packages:
        print_colored(f"\n⚠️  Missing packages: {', '.join(missing_packages)}", Colors.YELLOW)
        print_colored("Run: pip install -r requirements.txt", Colors.YELLOW)
        return False
    
    print_colored("✅ All required dependencies installed", Colors.GREEN)
    return True

def check_advanced_dependencies():
    """Check if advanced features dependencies are available."""
    print_colored("\n📋 Checking Advanced Features Dependencies...", Colors.BLUE)
    
    advanced_packages = [
        ('bcrypt', 'bcrypt'),
        ('jwt', 'PyJWT'),
        ('plotly', 'plotly'),
        ('pandas', 'pandas')
    ]
    
    missing_advanced = []
    
    for import_name, package_name in advanced_packages:
        try:
            importlib.import_module(import_name)
            print_colored(f"✅ {package_name}", Colors.GREEN)
        except ImportError:
            print_colored(f"⚠️  {package_name} - Missing (advanced features only)", Colors.YELLOW)
            missing_advanced.append(package_name)
    
    if missing_advanced:
        print_colored(f"\n💡 For advanced features, install: pip install {' '.join(missing_advanced)}", Colors.BLUE)
        return False
    
    print_colored("✅ All advanced dependencies available", Colors.GREEN)
    return True

def check_ollama_service():
    """Check if Ollama service is running."""
    print_colored("\n📋 Checking Ollama Service...", Colors.BLUE)
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print_colored("✅ Ollama service is running", Colors.GREEN)
            return True
        else:
            print_colored(f"❌ Ollama service returned status {response.status_code}", Colors.RED)
            return False
    except requests.exceptions.ConnectionError:
        print_colored("❌ Ollama service not running or not accessible", Colors.RED)
        print_colored("💡 Start with: ollama serve", Colors.YELLOW)
        return False
    except requests.exceptions.Timeout:
        print_colored("❌ Ollama service timeout", Colors.RED)
        return False
    except Exception as e:
        print_colored(f"❌ Error checking Ollama: {e}", Colors.RED)
        return False

def check_ollama_models():
    """Check if required models are available."""
    print_colored("\n📋 Checking Ollama Models...", Colors.BLUE)
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            
            # Check for default model
            default_model = "llama3:8b"
            model_found = False
            
            print_colored("Available models:", Colors.BLUE)
            for model in models:
                model_name = model.get("name", "")
                print_colored(f"  • {model_name}", Colors.GREEN)
                if model_name.startswith(default_model):
                    model_found = True
            
            if model_found:
                print_colored(f"✅ Default model {default_model} is available", Colors.GREEN)
                return True
            else:
                print_colored(f"❌ Default model {default_model} not found", Colors.RED)
                print_colored(f"💡 Download with: ollama pull {default_model}", Colors.YELLOW)
                return False
        else:
            print_colored("❌ Could not retrieve model list", Colors.RED)
            return False
    except Exception as e:
        print_colored(f"❌ Error checking models: {e}", Colors.RED)
        return False

def check_file_structure():
    """Check if required files are present."""
    print_colored("\n📋 Checking File Structure...", Colors.BLUE)
    
    required_files = [
        "launcher.py",
        "minions_gui.py", 
        "requirements.txt",
        "core/__init__.py",
        "core/agent_manager.py"
    ]
    
    optional_files = [
        "minions_gui_advanced.py",
        "core/advanced/__init__.py",
        "QUICKSTART.md"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print_colored(f"✅ {file_path}", Colors.GREEN)
        else:
            print_colored(f"❌ {file_path} - Missing", Colors.RED)
            missing_files.append(file_path)
    
    for file_path in optional_files:
        if Path(file_path).exists():
            print_colored(f"✅ {file_path} (optional)", Colors.GREEN)
        else:
            print_colored(f"⚠️  {file_path} - Missing (optional)", Colors.YELLOW)
    
    if missing_files:
        print_colored(f"\n❌ Missing required files: {', '.join(missing_files)}", Colors.RED)
        return False
    
    print_colored("✅ All required files present", Colors.GREEN)
    return True

def check_system_resources():
    """Check system resources."""
    print_colored("\n📋 Checking System Resources...", Colors.BLUE)
    
    try:
        import psutil
        
        # Check RAM
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print_colored(f"💾 Total RAM: {total_gb:.1f} GB", Colors.BLUE)
        print_colored(f"💾 Available RAM: {available_gb:.1f} GB", Colors.BLUE)
        
        if total_gb >= 8:
            print_colored("✅ Sufficient RAM for MinionsAI", Colors.GREEN)
            ram_ok = True
        elif total_gb >= 4:
            print_colored("⚠️  Minimum RAM available (may be slow)", Colors.YELLOW)
            ram_ok = True
        else:
            print_colored("❌ Insufficient RAM (4GB minimum required)", Colors.RED)
            ram_ok = False
        
        # Check disk space
        disk = psutil.disk_usage('.')
        free_gb = disk.free / (1024**3)
        
        print_colored(f"💽 Free disk space: {free_gb:.1f} GB", Colors.BLUE)
        
        if free_gb >= 5:
            print_colored("✅ Sufficient disk space", Colors.GREEN)
            disk_ok = True
        else:
            print_colored("❌ Insufficient disk space (5GB minimum required)", Colors.RED)
            disk_ok = False
        
        return ram_ok and disk_ok
        
    except ImportError:
        print_colored("⚠️  Cannot check system resources (psutil not available)", Colors.YELLOW)
        return True
    except Exception as e:
        print_colored(f"⚠️  Error checking system resources: {e}", Colors.YELLOW)
        return True

def test_basic_functionality():
    """Test basic MinionsAI functionality."""
    print_colored("\n📋 Testing Basic Functionality...", Colors.BLUE)
    
    try:
        # Test imports
        from core.agent_manager import AgentManager
        print_colored("✅ Core imports successful", Colors.GREEN)
        
        # Test Ollama connection with a simple request
        test_prompt = "Hello"
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:8b",
                "prompt": test_prompt,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            print_colored("✅ Ollama model responds correctly", Colors.GREEN)
            return True
        else:
            print_colored(f"❌ Ollama model test failed: {response.status_code}", Colors.RED)
            return False
            
    except ImportError as e:
        print_colored(f"❌ Import error: {e}", Colors.RED)
        return False
    except requests.exceptions.Timeout:
        print_colored("❌ Ollama model test timed out", Colors.RED)
        print_colored("💡 Model may be loading for the first time (this is normal)", Colors.YELLOW)
        return False
    except Exception as e:
        print_colored(f"❌ Functionality test failed: {e}", Colors.RED)
        return False

def provide_recommendations(results):
    """Provide recommendations based on test results."""
    print_colored("\n📋 Recommendations:", Colors.BLUE)
    
    if all(results.values()):
        print_colored("🎉 All checks passed! MinionsAI is ready to use.", Colors.GREEN)
        print_colored("\nNext steps:", Colors.BLUE)
        print_colored("1. Run: python launcher.py", Colors.GREEN)
        print_colored("2. Wait for browser to open", Colors.GREEN)
        print_colored("3. Click 'Initialize MinionsAI' in the sidebar", Colors.GREEN)
        print_colored("4. Start chatting with your AI assistant!", Colors.GREEN)
        
        if results.get('advanced_deps', False):
            print_colored("\n💡 Advanced features available! Try:", Colors.BLUE)
            print_colored("   python launcher.py --advanced", Colors.GREEN)
    else:
        print_colored("⚠️  Some issues found. Please address them before using MinionsAI:", Colors.YELLOW)
        
        if not results.get('python_version', True):
            print_colored("• Install Python 3.8 or higher", Colors.RED)
        
        if not results.get('dependencies', True):
            print_colored("• Install dependencies: pip install -r requirements.txt", Colors.RED)
        
        if not results.get('ollama_service', True):
            print_colored("• Install and start Ollama service", Colors.RED)
        
        if not results.get('ollama_models', True):
            print_colored("• Download model: ollama pull llama3:8b", Colors.RED)
        
        if not results.get('file_structure', True):
            print_colored("• Ensure all MinionsAI files are present", Colors.RED)
        
        if not results.get('system_resources', True):
            print_colored("• Upgrade system resources (RAM/disk space)", Colors.RED)

def main():
    """Main verification function."""
    print_header()
    
    # Run all checks
    results = {
        'python_version': check_python_version(),
        'dependencies': check_dependencies(),
        'advanced_deps': check_advanced_dependencies(),
        'ollama_service': check_ollama_service(),
        'ollama_models': check_ollama_models(),
        'file_structure': check_file_structure(),
        'system_resources': check_system_resources(),
        'basic_functionality': test_basic_functionality()
    }
    
    # Summary
    print_colored("\n" + "=" * 50, Colors.BLUE)
    print_colored("📊 VERIFICATION SUMMARY", Colors.BOLD)
    print_colored("=" * 50, Colors.BLUE)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        check_display = check_name.replace('_', ' ').title()
        print_colored(f"{status} - {check_display}", Colors.GREEN if result else Colors.RED)
    
    print_colored(f"\nOverall: {passed}/{total} checks passed ({passed/total*100:.1f}%)", Colors.BLUE)
    
    # Provide recommendations
    provide_recommendations(results)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
