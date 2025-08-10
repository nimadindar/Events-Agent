# Events Agent

An agentic workflow for conducting research by fetching and analyzing papers from ArXiv and blog posts from the web, saving results in JSON format, and posting the highest-scoring item to X. The agent leverages LangGraph, LangChain, Google Generative AI models, and tools like ArxivLoader and Tavily for research tasks.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running via Command Line](#running-via-command-line)
  - [Running via Streamlit UI](#running-via-streamlit-ui)
  - [Running via GitHub Actions](#running-via-github-actions)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Environment Variables](#environment-variables)
- [Logging](#logging)

---

## Overview

The Events Agent automates searching for relevant academic papers and blog posts, scoring them by relevance, saving results to JSON, and sharing the top item on X. It supports both command-line execution and an interactive Streamlit interface.

### Multi-Agent Workflow

This workflow, built using the **LangGraph** framework, organizes the process into two specialized teams:

1. **Research Team**

   - **ArXiv Agent** – Searches academic papers on ArXiv.
   - **Blog Agent** – Collects related blog information.
   - **Google Scholar Agent** – Finds relevant scholarly literature.

2. **Posting Team**

   - **X Agent** – Posts the highest-scoring research result to X (formerly Twitter).

Using dedicated agents for each step provides **modularity**, **control**, and **flexibility**, allowing updates or replacements without affecting the overall workflow.

---

### Single-Agent Workflow

In this simpler workflow, a **single unified agent** handles the entire process:

1. Searches ArXiv, blogs, and Google Scholar.
2. Selects the most relevant content.
3. Posts the result directly to X.

This approach is easier to manage and requires fewer resources but offers less granularity and flexibility.

> **Note:** The Streamlit application is currently out of date with recent updates; this will be addressed in future releases.

---

## Project Structure

```
├── .github/workflows/
│   └── run-script.yml         # GitHub Actions automation
├── multi_agent/
│   ├── agent_utils/
│   │   ├── utils.py           # Utility functions for agents
│   │   ├── PostingTeam/
│   │   │   └── X_node.py     # X posting agent
│   │   └── ResearchTeam/
│   │       ├── arxiv_node.py # ArXiv research agent
│   │       ├── blog_node.py  # Blog research agent
│   │       └── gscholar_node.py # Google Scholar research agent
│   ├── prompts/               # System prompts for multi-agent workflow
│   └── main.py                # Entry point for multi-agent workflow
├── single_agent/
│   ├── agents/
│   │   ├── agent_utils.py     # Utilities for single agent
│   │   └── build_agent.py     # Single agent builder script
│   ├── prompts/
│   │   └── system_prompt.yaml # System prompt templates
│   └── main.py                # Entry point for single-agent workflow
├── saved/                     # Output and logs
│   ├── results.json           # Search results
│   ├── tweets.json            # Tweeted URLs tracker
│   └── agent_logs.log         # Execution logs
├── tools/
│   └── tools.py               # Custom tools for research and posting
├── utils/
│   └── utils.py               # General utilities
├── app.py                     # Streamlit UI application
├── config.py                  # Configuration settings
├── main.py                    # Main script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/nimadindar/Events-Agent.git
   cd Events-Agent
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables (see [Environment Variables](#environment-variables)).

---

## Usage

### Running via Command Line

1. Edit `config.py` to configure:

   - Research topic (`Field`)
   - Maximum results per source (`ArxivMaxResults`, `TavilyMaxResults`, `ScholarMaxResults`)
   - Minimum usefulness thresholds (`ArxivMinUsefulness`, `BlogMinUsefulness`, `ScholarMinUsefulness`)
   - Google Scholar author IDs (`ScholarPages`)
   - Prompt version (`PromptVersion`) for single agent workflow
   - Model, temperature, verbosity, and invoke input

2. Run single-agent workflow (deprecated and no longer maintained):

   ```bash
   python -B -m single_agent.main
   ```

3. Run multi-agent workflow:

   ```bash
   python -B -m multi_agent.main
   ```

---

### Running via Streamlit UI

1. Start the app:

   ```bash
   streamlit run app.py
   ```

2. Open the provided URL (e.g., `http://localhost:8501`).

3. Configure parameters through the interface:

   - Research field, model, temperature
   - Max results and minimum scores per source
   - Scholar page IDs
   - Verbosity and scheduling options
   - API keys (Google, X, Tavily) or provide via `.env`
   - Agent task description

4. Click **Run Agent** or **Schedule Agent**.

> **Note:** The Streamlit UI is currently outdated and expected to be updated soon.

---

### Running via GitHub Actions

1. Add API keys as **Secrets** under your repository's **Settings → Secrets and variables → Actions**.

2. Trigger the scheduled workflow or manual runs under the **Actions** tab.\
   The default schedule runs every 12 hours and can be modified in `.github/workflows/run-script.yml`.

---

## Configuration

The agent can be configured via:

- **config.py** (for command-line runs) with parameters such as:

  - `PromptDir` (single-agent prompt path)
  - `Field`, `ArxivMaxResults`, `TavilyMaxResults`, `ScholarMaxResults`
  - `ArxivMinUsefulness`, `TavilyMinUsefulness`, `ScholarMinUsefulness`
  - `ScholarPages` (author IDs)
  - `PromptVersion`, `ModelName`, `Temperature`, `Verbose`
  - `InvokeInput` (task description)

- **Streamlit UI** for dynamic configuration.

---

## Dependencies

Core dependencies are listed in `requirements.txt` and include:

- `langchain`
- `langchain-google-genai`
- `langchain-community`
- `tweepy`
- `tavily-python`
- `streamlit`
- `python-dotenv`
- `pyyaml`

Install with:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file at the project root with the following keys:

```bash
GOOGLE_API_KEY=your_google_api_key
X_API_KEY=your_x_consumer_key
X_API_KEY_SECRET=your_x_consumer_secret
X_ACCESS_TOKEN=your_x_access_token
X_ACCESS_TOKEN_SECRET=your_x_access_token_secret
TAVILY_API_KEY=your_tavily_api_key
SERP_API_KEY=your_serp_api_key
```

Sources for keys:

- Google API: Google Cloud Console
- X API: [X Developer Portal](https://developer.x.com)
- Tavily API: [Tavily Dashboard](https://tavily.com)
- Serp API: [Serpapi](https://serpapi.com/)

---

## Logging

- Logs are saved to `./saved/agent_logs.log` with rotation (max 5MB, 3 backups).
- Enable verbose logging via `config.py` or Streamlit UI.
- Streamlit shows logs when verbose mode is enabled.

---