# Events-Agent 🤖

> **An intelligent multi-agent research workflow that automatically discovers, analyzes, and shares cutting-edge research across ArXiv, blogs, and Google Scholar.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6.4+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46.1+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Overview

Events-Agent is a research automation system that leverages **LangGraph** and **Google's Gemini AI** to intelligently discover, evaluate, and disseminate research content. The system operates through two distinct workflows:

- **🔬 Multi-Agent Workflow**: Specialized agents for research and posting tasks
- **⚡ Single-Agent Workflow**: Unified agent for streamlined operations

### ✨ Key Features

- **Automated Research Discovery**: Searches ArXiv, web blogs, and Google Scholar simultaneously
- **Intelligent Content Scoring**: AI-powered relevance assessment with configurable thresholds
- **Smart Content Curation**: Automatically selects and formats the most valuable findings
- **Social Media Integration**: Direct posting to X (Twitter) with intelligent content formatting
- **Flexible Deployment**: Command-line, Streamlit UI, and GitHub Actions automation
- **Comprehensive Logging**: Detailed execution tracking and result storage

## 🏗️ Architecture

### Multi-Agent Workflow (Recommended)

Built with **LangGraph**, this workflow provides maximum flexibility and control:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   ArXiv     │───▶│    Blog     │───▶│  Google     │───▶│     X       │
│   Agent     │    │   Agent     │    │   Agent     │    │   Agent     │
│             │    │             │    │   Agent     │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Research Team:**
- **ArXiv Agent**: Academic paper discovery and analysis
- **Blog Agent**: Web content research and evaluation  
- **Google Scholar Agent**: Scholarly literature exploration

**Posting Team:**
- **X Agent**: Content formatting and social media posting

### Single-Agent Workflow

A unified approach for simpler deployments:
- Single agent handles all research and posting tasks
- Reduced resource requirements
- Easier maintenance and debugging

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Google Cloud API key
- X (Twitter) API credentials
- Tavily API key (optional, for enhanced web search)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/nimadindar/Events-Agent.git
   cd Events-Agent
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# X (Twitter) API (for posting)
X_API_KEY=your_x_consumer_key
X_API_KEY_SECRET=your_x_consumer_secret
X_ACCESS_TOKEN=your_x_access_token
X_ACCESS_TOKEN_SECRET=your_x_access_token_secret

TAVILY_API_KEY=your_tavily_api_key
SERP_API_KEY=your_serp_api_key
```

### Configuration File

Edit `multi_agent/config.py` to customize:

```python
class AgentConfig:
    # Research field
    Field = "Spatio Temporal Point Process, Spatio Temporal, Point Process"
    
    # Result limits (adjust based on your LLM's context window)
    ArxivMaxResults = 5
    TavilyMaxResults = 5
    ScholarMaxResults = 2
    
    # Quality thresholds (0-100)
    ArxivMinUsefulness = 70
    BlogMinUsefulness = 70
    ScholarMinUsefulness = 70
    XMinUsefulness = 70
    
    # AI Model settings
    ModelName = "gemini-2.5-flash"
    Temperature = 0.0  # Lower = more consistent, Higher = more creative
```

## 🚀 Usage

### Command Line Interface

#### Multi-Agent Workflow (Recommended)
```bash
python -m multi_agent.main
```

#### Single-Agent Workflow
```bash
python -m single_agent.main
```

### Streamlit Web Interface

Launch the interactive web UI:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` and configure:
- Research field and AI model
- Result limits and quality thresholds
- API keys and execution parameters
- Real-time monitoring and logging

### GitHub Actions Automation

1. **Set up repository secrets** in `Settings → Secrets and variables → Actions`
2. **Configure workflow** in `.github/workflows/run-script.yml`
3. **Schedule runs** (default: every 12 hours) or trigger manually

## 📊 Output & Results

### Generated Files

All results are saved in the `saved/` directory:

- `results.json` - Comprehensive research findings
- `blog_results.json` - Blog post analysis
- `gscholar_results.json` - Google Scholar papers
- `tweets.json` - Posted content tracking
- `agent_logs.log` - Execution logs

### Result Structure

```json
{
  "arxiv_papers": [
    {
      "title": "Paper Title",
      "authors": ["Author 1", "Author 2"],
      "abstract": "Abstract text...",
      "url": "https://arxiv.org/...",
      "usefulness_score": 85,
      "summary": "AI-generated summary..."
    }
  ],
  "blog_posts": [...],
  "scholar_papers": [...],
  "best_content": {
    "type": "arxiv_paper",
    "content": {...},
    "post_text": "Formatted for X..."
  }
}
```

## 🔧 Customization

### Adding New Research Sources

1. Create a new agent in `multi_agent/ResearchTeam/`
2. Implement the required interface
3. Add to the workflow graph in `multi_agent/main.py`

### Modifying Posting Behavior

Edit `multi_agent/PostingTeam/X_node.py` to customize:
- Content formatting
- Posting frequency
- Quality filters
- Error handling

### Custom Prompts

Modify YAML files in `multi_agent/prompts/` to adjust:
- Research criteria
- Content evaluation
- Output formatting

## 🛠️ Development

### Project Structure

```
Events-Agent/
├── multi_agent/           # Multi-agent workflow (recommended)
│   ├── agent_utils/       # Shared utilities
│   ├── ResearchTeam/      # Research agents
│   ├── PostingTeam/       # Social media agents
│   ├── prompts/           # AI prompt templates
│   └── main.py           # Workflow orchestration
├── single_agent/          # Single-agent workflow
│   ├── agents/            # Agent implementation
│   └── prompts/           # Prompt templates
├── tools/                 # Research and posting tools
├── utils/                 # General utilities
├── app.py                # Streamlit web interface
├── config.py             # Configuration settings
└── requirements.txt      # Python dependencies
```

### Key Dependencies

- **LangGraph**: Multi-agent workflow orchestration
- **LangChain**: LLM integration and tool management
- **Google Generative AI**: Advanced language models
- **Streamlit**: Web interface framework
- **Tweepy**: X (Twitter) API integration

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 API Reference

### Core Functions

- `run_agent()` - Execute the research workflow
- `load_llm()` - Initialize AI language model
- `save_to_json()` - Store research results
- `post_to_X()` - Share content on social media

### Configuration Options

See `multi_agent/config.py` for all available settings and their descriptions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
