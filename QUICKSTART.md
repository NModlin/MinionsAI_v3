# MinionsAI v3.1 - QuickStart Guide

**Get up and running with MinionsAI in under 10 minutes!** 🚀

MinionsAI v3.1 is a sophisticated multi-agent AI system that works like a desktop application - just one click to start, and you're ready to chat with AI agents that can search the web, analyze data, write code, and collaborate to solve complex problems.

## ⏱️ 10-Minute Setup Timeline

Here's what you'll accomplish in the next 10 minutes:

```
Minutes 0-2:  📥 Download and install dependencies
Minutes 2-4:  🔧 Set up Ollama service
Minutes 4-8:  🧠 Download AI model (llama3:8b)
Minutes 8-9:  ✅ Verify installation
Minutes 9-10: 🚀 Launch and test MinionsAI
```

**Total time**: ~10 minutes (model download may vary by internet speed)

## 📋 Prerequisites & System Requirements

### Required Software
- **Python 3.8 or higher** ([Download here](https://python.org/downloads/))
- **Ollama** ([Download here](https://ollama.ai/download))
- **Git** (optional, for cloning the repository)

### Hardware Requirements
- **RAM**: 8GB minimum (16GB recommended for advanced features)
- **Disk Space**: 5GB free space (for models and data)
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **Network**: Internet connection for model downloads and web search

### Supported Operating Systems
- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Linux (Ubuntu 20.04+, CentOS 8+, etc.)

---

## 🛠️ Installation Steps

### Step 1: Install Python Dependencies

1. **Download or clone MinionsAI v3.1**:
   ```bash
   git clone https://github.com/your-repo/MinionsAI_v3.git
   cd MinionsAI_v3
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   > 💡 **Tip**: Use a virtual environment to avoid conflicts:
   > ```bash
   > python -m venv minions_env
   > # Windows:
   > minions_env\Scripts\activate
   > # macOS/Linux:
   > source minions_env/bin/activate
   > pip install -r requirements.txt
   > ```

### Step 2: Install and Setup Ollama

1. **Download and install Ollama** from [ollama.ai](https://ollama.ai/download)

2. **Start the Ollama service**:
   ```bash
   # The installer usually starts Ollama automatically
   # If not, run:
   ollama serve
   ```

3. **Verify Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```
   You should see a JSON response (even if empty).

### Step 3: Download the Default Model

MinionsAI uses the `llama3:8b` model by default. Download it:

```bash
ollama pull llama3:8b
```

> ⏱️ **Note**: This download is ~4.7GB and may take 5-15 minutes depending on your internet speed.

### Step 4: Verify Installation

Run our comprehensive verification script:

```bash
python verify_installation.py
```

This will check:
- ✅ Python version and dependencies
- ✅ Ollama service status
- ✅ Model availability
- ✅ System resources
- ✅ File structure
- ✅ Basic functionality

**Expected output:**
```
🔍 MinionsAI v3.1 - Installation Verification
==================================================
📋 Checking Python Version...
✅ Python 3.11.0 - Compatible
📋 Checking Python Dependencies...
✅ streamlit
✅ requests
✅ langchain
...
📊 VERIFICATION SUMMARY
==================================================
✅ PASS - Python Version
✅ PASS - Dependencies
✅ PASS - Ollama Service
✅ PASS - Ollama Models
✅ PASS - File Structure
✅ PASS - System Resources
✅ PASS - Basic Functionality

Overall: 7/7 checks passed (100.0%)
🎉 All checks passed! MinionsAI is ready to use.
```

---

## 🚀 First Launch

### Launch MinionsAI

1. **Open a terminal/command prompt** in the MinionsAI directory

2. **Run the one-click launcher**:
   ```bash
   python launcher.py
   ```

3. **Wait for the startup sequence**:
   ```
   🤖 MinionsAI v3.1 - One-Click Launcher
   =====================================
   🔍 Checking system requirements...
   ✅ System requirements satisfied
   🚀 Checking Ollama service...
   ✅ Ollama service is running
   🧠 Checking model: llama3:8b...
   ✅ Model llama3:8b is ready
   🖥️  Starting MinionsAI GUI...
   ✅ MinionsAI GUI started successfully!
   🌐 Opening browser at http://localhost:8501
   🎉 MinionsAI is ready to use!
   ```

4. **Your browser will automatically open** to `http://localhost:8501`

### Understanding the GUI Interface

**Standard Mode Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  🤖 MinionsAI v3.1 - Your AI Assistant                     │
├─────────────────────────────────────────────────────────────┤
│ Sidebar:                    │ Main Chat Area:               │
│ • 🚀 Initialize MinionsAI   │                              │
│ • ⚙️ Settings              │  💬 Chat with AI agents      │
│ • 📊 System Status         │                              │
│ • 🔧 Multi-Agent Toggle    │  [Type your message here...] │
│ • 🌐 Web Search: ON        │  [Send] 📤                   │
│ • 🧠 Model: llama3:8b      │                              │
└─────────────────────────────────────────────────────────────┘
```

**Advanced Mode Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  🤖 MinionsAI v3.1 Advanced - Enterprise AI Platform       │
├─────────────────────────────────────────────────────────────┤
│ Navigation:                 │ Content Area:                 │
│ • 💬 Chat                  │                              │
│ • 📊 Analytics             │  Selected page content        │
│ • ⚡ Performance           │  (Chat, Analytics, etc.)     │
│ • 🧠 Models                │                              │
│ • 🛠️ Tools                 │                              │
│ • 🔒 Security              │                              │
│ • ⚙️ Settings              │                              │
└─────────────────────────────────────────────────────────────┘
```

### Make Your First Query

1. **Click "🚀 Initialize MinionsAI"** in the sidebar (this takes ~30 seconds)

2. **Wait for the green checkmark**: ✅ MinionsAI initialized successfully!

3. **Type your first message** in the chat box:
   ```
   Hello! Can you search for the latest news about artificial intelligence?
   ```

4. **Press Enter or click Send** and watch MinionsAI:
   - Search the web for current AI news
   - Analyze and summarize the findings
   - Present you with a comprehensive response

🎉 **Congratulations!** You're now using MinionsAI v3.1!

---

## 🎛️ Feature Overview

### Standard Mode (Default)
Perfect for everyday use:
- **Single AI Agent**: One intelligent assistant
- **Web Search**: Real-time information retrieval
- **Clean Interface**: Simple, distraction-free chat
- **Fast Performance**: Optimized for quick responses

### Advanced Mode
Enterprise features for power users:
```bash
python launcher.py --advanced
```

**Advanced Mode includes:**
- **Multi-Agent System**: 5 specialized AI agents
  - 🔍 **Research Agent**: Web search and fact-checking
  - 📊 **Analysis Agent**: Data analysis and insights
  - 💻 **Code Agent**: Programming and debugging
  - 📋 **Planning Agent**: Task planning and organization
  - 📝 **Summary Agent**: Information synthesis
- **Analytics Dashboard**: Usage metrics and performance monitoring
- **Security Features**: User authentication and access control
- **Performance Monitoring**: Real-time system metrics
- **Advanced Tools**: File operations, database access, API integrations

### Multi-Agent Capabilities

**When to use Multi-Agent Mode:**
- Complex research projects requiring multiple perspectives
- Code development with planning, implementation, and review
- Data analysis with visualization and reporting
- Strategic planning with risk assessment and optimization

**Example Multi-Agent Workflow:**
1. **Planning Agent** breaks down your request into subtasks
2. **Research Agent** gathers relevant information
3. **Analysis Agent** processes and analyzes the data
4. **Code Agent** creates any necessary scripts or tools
5. **Summary Agent** synthesizes everything into a final report

---

## 🔧 Common Issues & Troubleshooting

### Issue: "Ollama service not running"
**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it:
ollama serve

# On Windows, you might need to run as administrator
```

### Issue: "Model download failed"
**Solutions:**
1. **Check internet connection**
2. **Try downloading manually**:
   ```bash
   ollama pull llama3:8b
   ```
3. **Use a different model** (edit `launcher.py` and change `OLLAMA_MODEL`):
   ```python
   OLLAMA_MODEL = "llama3.1:8b"  # or "mistral:7b"
   ```

### Issue: "Port 8501 already in use"
**Solution:**
```bash
# Use a different port
python launcher.py --port 8502

# Or find and kill the process using port 8501
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID_NUMBER> /F

# macOS/Linux:
lsof -ti:8501 | xargs kill -9
```

### Issue: "Permission denied" errors
**Solutions:**
- **Windows**: Run Command Prompt as Administrator
- **macOS/Linux**: Use `sudo` if needed, or fix permissions:
  ```bash
  chmod +x launcher.py
  ```

### Issue: GUI won't load or shows errors
**Solutions:**
1. **Clear browser cache** and refresh
2. **Try a different browser** (Chrome, Firefox, Safari)
3. **Check the terminal** for error messages
4. **Restart the launcher**:
   ```bash
   # Press Ctrl+C to stop, then restart
   python launcher.py
   ```

### Issue: Slow performance
**Solutions:**
1. **Close other applications** to free up RAM
2. **Use a smaller model**:
   ```bash
   ollama pull llama3.1:8b  # Smaller than llama3:8b
   ```
3. **Enable caching** in Advanced Mode for faster repeated queries

---

## 📚 Next Steps

### Explore Advanced Features
- **[Phase 4 Advanced Features Guide](PHASE4_ADVANCED_FEATURES.md)**: Complete guide to enterprise features
- **Security Setup**: Configure user authentication and access control
- **Multi-Model Configuration**: Add OpenAI, Anthropic, or other model providers
- **Custom Tool Development**: Create specialized tools for your workflow

### Configuration Files
- **`config/config.yaml`**: Main configuration settings
- **`launcher.py`**: Modify default model, ports, and startup options
- **`requirements.txt`**: Add additional Python packages

### Community & Support
- **Documentation**: Check the `/docs` folder for detailed guides
- **Examples**: See `/examples` for sample workflows and use cases
- **Issues**: Report bugs or request features on GitHub

### Performance Optimization
- **Hardware**: More RAM = better performance with larger models
- **Models**: Experiment with different models for your use case:
  - `llama3:8b` - Best balance of speed and capability
  - `llama3.1:8b` - Latest version with improvements
  - `mistral:7b` - Faster, good for simple tasks
  - `codellama:7b` - Specialized for programming tasks

---

## 🎯 Quick Reference

### Essential Commands
```bash
# Standard launch
python launcher.py

# Advanced mode
python launcher.py --advanced

# Custom port
python launcher.py --port 8502

# Check Ollama status
curl http://localhost:11434/api/tags

# Download new model
ollama pull model-name

# Stop MinionsAI
# Press Ctrl+C in the terminal
```

### Default URLs
- **Standard GUI**: http://localhost:8501
- **Ollama API**: http://localhost:11434
- **Advanced GUI**: http://localhost:8501 (with `--advanced` flag)

### File Structure
```
MinionsAI_v3/
├── launcher.py              # 🚀 One-click launcher
├── minions_gui.py          # Standard GUI
├── minions_gui_advanced.py # Advanced GUI
├── requirements.txt        # Python dependencies
├── core/                   # Core system
├── config/                 # Configuration files
└── docs/                   # Documentation
```

---

**🎉 You're all set!** MinionsAI v3.1 is now ready to help you with research, analysis, coding, planning, and much more. Start with simple questions and gradually explore the advanced features as you become more comfortable with the system.

**Happy AI-ing!** 🤖✨
