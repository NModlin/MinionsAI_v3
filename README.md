# MinionsAI v3.1 🤖

**A sophisticated multi-agent AI system with enterprise features, packaged as a simple one-click desktop application.**

MinionsAI v3.1 combines the power of multiple specialized AI agents with an intuitive interface that anyone can use. Whether you need web research, data analysis, code generation, or complex problem-solving, MinionsAI provides professional-grade AI capabilities without the complexity.

## ✨ Key Features

### 🎯 **Core Capabilities**
- **One-Click Launch**: Desktop application simplicity with enterprise power
- **Multi-Agent Collaboration**: 5 specialized AI agents working together
- **Web Search Integration**: Real-time information retrieval and fact-checking
- **Advanced Analytics**: Comprehensive usage and performance monitoring
- **Enterprise Security**: User authentication and role-based access control

### 🤖 **Specialized AI Agents**
- **🔍 Research Agent**: Web search, fact-checking, information gathering
- **📊 Analysis Agent**: Data analysis, pattern recognition, insights
- **💻 Code Agent**: Programming, debugging, code review
- **📋 Planning Agent**: Task planning, project management, coordination
- **📝 Summary Agent**: Information synthesis, report generation

### 🏢 **Enterprise Features**
- **Performance Optimization**: Intelligent caching and async processing
- **Security & Authentication**: Multi-user support with secure access control
- **Advanced Tool Ecosystem**: File operations, database access, API integrations
- **Business Intelligence**: Analytics dashboard with comprehensive reporting
- **Production Ready**: Scalable architecture with monitoring and alerting

## 🚀 Quick Start

**Get up and running in under 10 minutes!**

### Option 1: Follow the QuickStart Guide (Recommended)
📖 **[Read the Complete QuickStart Guide](QUICKSTART.md)** - Step-by-step instructions with troubleshooting

### Option 2: Express Setup
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
3. **Download Model**: `ollama pull llama3:8b`
4. **Verify Setup**: `python verify_installation.py`
5. **Launch**: `python launcher.py`
6. **Open Browser**: Go to `http://localhost:8501`

### Advanced Features
```bash
# Launch with enterprise features
python launcher.py --advanced

# Custom port
python launcher.py --port 8502
```

**Need Help?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

## 📋 System Requirements

- **Python 3.8+** ([Download](https://python.org/downloads/))
- **8GB+ RAM** (16GB recommended for advanced features)
- **5GB+ disk space** (for models and data)
- **Ollama service** ([Download](https://ollama.ai/download))
- **Internet connection** (for model downloads and web search)

**Supported OS**: Windows 10/11, macOS 10.15+, Linux (Ubuntu 20.04+)

## 🎛️ Usage Modes

### Standard Mode
Perfect for everyday AI assistance:
- Single intelligent agent
- Web search capabilities
- Clean, simple interface
- Fast performance

### Advanced Mode
Enterprise features for power users:
- Multi-agent collaboration
- Analytics dashboard
- Performance monitoring
- Security features
- Advanced tool ecosystem

## 📊 Architecture Overview

```
MinionsAI v3.1 Architecture
┌─────────────────────────────────────────────────────────┐
│                    GUI Layer                            │
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │  Standard GUI   │    │     Advanced GUI            │ │
│  │  (Streamlit)    │    │  (Multi-page Dashboard)     │ │
│  └─────────────────┘    └─────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                 Agent Management Layer                  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │           Enhanced Agent Manager                    │ │
│  │  • Single/Multi-agent modes                        │ │
│  │  • Performance monitoring                          │ │
│  │  • Security integration                            │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                Multi-Agent System                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐ │
│  │Research │ │Analysis │ │  Code   │ │Planning/Summary │ │
│  │ Agent   │ │ Agent   │ │ Agent   │ │    Agents       │ │
│  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                Advanced Features                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐ │
│  │Security │ │Analytics│ │Advanced │ │   Performance   │ │
│  │& Auth   │ │Engine   │ │ Tools   │ │   Monitoring    │ │
│  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                    Core Engine                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              LangGraph + Ollama                     │ │
│  │         (llama3:8b + Web Search Tools)              │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Development Phases

MinionsAI v3.1 was built in four comprehensive phases:

- **✅ Phase 1**: Core Agent Engine with LangGraph and Ollama integration
- **✅ Phase 2**: Unified GUI Application with Streamlit interface
- **✅ Phase 3**: Multi-Agent System with specialized agent collaboration
- **✅ Phase 4**: Advanced Features with enterprise capabilities

Each phase builds upon the previous, creating a robust and scalable AI platform.

## 📚 Documentation

### Getting Started
- **[QuickStart Guide](QUICKSTART.md)** - Get running in 10 minutes
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Installation Verification](verify_installation.py)** - Test your setup

### Advanced Features
- **[Phase 4 Advanced Features](PHASE4_ADVANCED_FEATURES.md)** - Enterprise capabilities
- **[Configuration Example](config_example.yaml)** - Customize your setup

### Testing
- **[Multi-Agent Tests](test_multi_agent.py)** - Test multi-agent functionality
- **[Advanced Features Tests](test_advanced_features.py)** - Test enterprise features
- **[GUI Tests](test_gui.py)** - Test user interface

## 🔧 Configuration

MinionsAI v3.1 is highly configurable. Copy `config_example.yaml` to `config.yaml` and customize:

```yaml
# Basic Settings
ollama:
  default_model: "llama3:8b"
  base_url: "http://localhost:11434"

# GUI Settings  
gui:
  port: 8501
  theme: "light"

# Advanced Features
advanced_features:
  enabled: true
  security:
    enable_authentication: true
  analytics:
    enabled: true
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Issues**: Use GitHub Issues for bugs and feature requests
2. **Submit Pull Requests**: Follow our coding standards and include tests
3. **Improve Documentation**: Help make our guides even better
4. **Share Examples**: Show us how you're using MinionsAI

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama Team** for the excellent local LLM platform
- **LangChain/LangGraph** for the agent framework
- **Streamlit** for the beautiful web interface
- **Open Source Community** for the amazing tools and libraries

## 📞 Support

- **Documentation**: Check our comprehensive guides
- **GitHub Issues**: Report bugs and request features
- **Community**: Join discussions and share experiences

---

**MinionsAI v3.1** - Where sophisticated AI meets simple usability. 🚀✨
