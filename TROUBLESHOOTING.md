# MinionsAI v3.1 - Troubleshooting Guide

This guide provides detailed solutions for common issues you might encounter while setting up or using MinionsAI v3.1.

## 🚨 Installation Issues

### Python Version Problems

**Error**: `Python version 3.8+ required`
```bash
# Check your Python version
python --version
# or
python3 --version

# If version is too old, download from python.org
# Make sure to add Python to PATH during installation
```

**Error**: `pip not found` or `pip install fails`
```bash
# Windows - reinstall Python with pip
# Download from python.org and check "Add to PATH"

# macOS - install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

# Linux - install pip
sudo apt update
sudo apt install python3-pip  # Ubuntu/Debian
sudo yum install python3-pip  # CentOS/RHEL
```

### Dependency Installation Issues

**Error**: `Failed building wheel for [package]`
```bash
# Install build tools
# Windows:
pip install --upgrade setuptools wheel

# macOS:
xcode-select --install
pip install --upgrade setuptools wheel

# Linux:
sudo apt install build-essential python3-dev  # Ubuntu/Debian
sudo yum groupinstall "Development Tools"     # CentOS/RHEL
```

**Error**: `Microsoft Visual C++ 14.0 is required` (Windows)
```bash
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Or install Visual Studio Community with C++ workload
```

## 🔧 Ollama Service Issues

### Ollama Installation Problems

**Issue**: Ollama installer fails or won't start
```bash
# Manual installation check
# Windows: Check if ollama.exe is in PATH
where ollama

# macOS/Linux: Check installation
which ollama

# If not found, reinstall from ollama.ai
# Make sure to restart terminal after installation
```

**Issue**: `ollama serve` command not found
```bash
# Add Ollama to PATH manually
# Windows: Add C:\Users\%USERNAME%\AppData\Local\Programs\Ollama to PATH
# macOS: Add /usr/local/bin to PATH
# Linux: Usually installed to /usr/local/bin

# Restart terminal and try again
ollama --version
```

### Ollama Service Connection Issues

**Error**: `Connection refused to localhost:11434`
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start manually
ollama serve

# Check for port conflicts
netstat -an | grep 11434  # Linux/macOS
netstat -an | findstr 11434  # Windows

# Try different port
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

**Error**: `Ollama service crashes on startup`
```bash
# Check system resources
# Ollama needs at least 4GB RAM for llama3:8b

# Check logs
# Windows: Check Event Viewer
# macOS: Console app or ~/Library/Logs/
# Linux: journalctl -u ollama or /var/log/

# Try with verbose logging
OLLAMA_DEBUG=1 ollama serve
```

## 🧠 Model Download Issues

### Model Download Failures

**Error**: `Failed to download model`
```bash
# Check internet connection
ping ollama.ai

# Check available disk space
df -h  # Linux/macOS
dir   # Windows

# Try downloading manually with verbose output
ollama pull llama3:8b --verbose

# If still failing, try a smaller model first
ollama pull tinyllama:1.1b
```

**Error**: `Model download interrupted`
```bash
# Resume interrupted download
ollama pull llama3:8b

# If corrupted, remove and re-download
ollama rm llama3:8b
ollama pull llama3:8b

# Check available models
ollama list
```

### Model Loading Issues

**Error**: `Model not found` or `Model failed to load`
```bash
# Verify model is downloaded
ollama list

# Test model directly
ollama run llama3:8b "Hello, world!"

# If model exists but won't load, check system resources
# llama3:8b needs ~8GB RAM

# Try a smaller model
ollama pull llama3.1:8b  # Often more efficient
```

## 🖥️ GUI and Launcher Issues

### Launcher Startup Problems

**Error**: `ModuleNotFoundError: No module named 'streamlit'`
```bash
# Reinstall requirements
pip install -r requirements.txt

# If using virtual environment, make sure it's activated
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Then reinstall
pip install -r requirements.txt
```

**Error**: `Port 8501 is already in use`
```bash
# Find process using port 8501
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:8501 | xargs kill -9

# Or use different port
python launcher.py --port 8502
```

### GUI Loading Issues

**Issue**: Browser opens but shows "This site can't be reached"
```bash
# Check if Streamlit is actually running
# Look for "You can now view your Streamlit app in your browser"

# If not, check for Python errors in terminal
# Common issues:
# 1. Import errors - reinstall requirements
# 2. Port conflicts - use different port
# 3. Firewall blocking - check firewall settings
```

**Issue**: GUI loads but shows errors or blank page
```bash
# Clear browser cache and cookies for localhost
# Try different browser (Chrome, Firefox, Safari)
# Check browser console for JavaScript errors (F12)

# Try incognito/private browsing mode
# Disable browser extensions that might interfere
```

## 🔄 Runtime Issues

### Performance Problems

**Issue**: Very slow responses or timeouts
```bash
# Check system resources
# Task Manager (Windows) or Activity Monitor (macOS) or htop (Linux)

# MinionsAI + llama3:8b needs:
# - 8GB+ RAM
# - Modern CPU
# - SSD recommended

# Try smaller model
ollama pull tinyllama:1.1b
# Edit launcher.py: OLLAMA_MODEL = "tinyllama:1.1b"
```

**Issue**: High CPU or memory usage
```bash
# Close other applications
# Restart Ollama service
ollama serve

# Monitor resource usage
# Windows: Task Manager
# macOS: Activity Monitor  
# Linux: htop or top

# Consider upgrading hardware or using cloud deployment
```

### Connection and Network Issues

**Error**: `Failed to connect to web search` or `Search unavailable`
```bash
# Check internet connection
ping google.com

# Check if firewall is blocking Python/Streamlit
# Windows: Windows Defender Firewall
# macOS: System Preferences > Security & Privacy > Firewall
# Linux: ufw status or iptables -L

# Try disabling VPN temporarily
# Some VPNs block certain requests
```

**Error**: `SSL certificate verification failed`
```bash
# Update certificates
# Windows: Windows Update
# macOS: Update macOS
# Linux: sudo apt update && sudo apt upgrade ca-certificates

# Temporary workaround (not recommended for production):
export PYTHONHTTPSVERIFY=0  # Linux/macOS
set PYTHONHTTPSVERIFY=0     # Windows
```

## 🔒 Advanced Features Issues

### Security and Authentication Problems

**Issue**: Cannot create users or login fails
```bash
# Check if data directory is writable
ls -la data/  # Should show write permissions

# Create data directory if missing
mkdir -p data
chmod 755 data

# Reset security database (WARNING: removes all users)
rm data/users.json
# Restart MinionsAI - default admin user will be recreated
```

**Issue**: JWT token errors or session issues
```bash
# Clear browser cookies for localhost
# Restart MinionsAI to generate new secret key
# Check system clock - JWT tokens are time-sensitive
```

### Multi-Agent System Issues

**Issue**: Agents not responding or coordination failures
```bash
# Check if all agents initialized properly
# Look for "✅ [Agent Name] registered" messages

# Restart with single agent mode first
python launcher.py
# Then try advanced mode
python launcher.py --advanced

# Check system resources - multi-agent needs more RAM
```

## 🛠️ Advanced Troubleshooting

### Debug Mode

Enable detailed logging:
```bash
# Set environment variables
export MINIONS_DEBUG=1
export STREAMLIT_LOGGER_LEVEL=debug

# Windows:
set MINIONS_DEBUG=1
set STREAMLIT_LOGGER_LEVEL=debug

# Run with debug output
python launcher.py
```

### Log Files

Check log files for detailed error information:
```bash
# MinionsAI logs (if configured)
tail -f logs/minions.log

# Streamlit logs
~/.streamlit/logs/

# Ollama logs
# Windows: %APPDATA%\Ollama\logs\
# macOS: ~/Library/Logs/Ollama/
# Linux: /var/log/ollama/ or ~/.ollama/logs/
```

### Clean Installation

If all else fails, try a clean installation:
```bash
# 1. Stop all MinionsAI processes
# Press Ctrl+C in all terminals

# 2. Remove virtual environment (if used)
rm -rf venv/  # Linux/macOS
rmdir /s venv  # Windows

# 3. Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 4. Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Reset Ollama (optional)
ollama rm llama3:8b
ollama pull llama3:8b

# 6. Clear browser data for localhost
# 7. Restart everything
python launcher.py
```

## 📞 Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Look at terminal/console output** for error messages
3. **Try the clean installation** steps above
4. **Test with minimal setup** (single agent, standard GUI)

### Information to Include

When reporting issues, please include:
- **Operating System** and version
- **Python version** (`python --version`)
- **MinionsAI version** (check git commit or download date)
- **Complete error message** (copy from terminal)
- **Steps to reproduce** the issue
- **System specifications** (RAM, CPU)

### Community Resources

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check `/docs` folder for detailed guides
- **Examples**: See `/examples` for working configurations

---

**Remember**: Most issues are related to system requirements, network connectivity, or missing dependencies. Start with the basics and work your way up to more complex solutions.
