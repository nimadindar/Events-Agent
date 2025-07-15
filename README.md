# Research Agent

A Streamlit-based application for conducting research by fetching and analyzing papers from ArXiv and blog posts from the web, saving results in a JSON file, and posting the highest-scoring item to X. The agent leverages LangChain, Google Generative AI models, and tools like ArxivLoader and Tavily for research tasks.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running via Command Line](#running-via-command-line)
  - [Running via Streamlit UI](#running-via-streamlit-ui)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Environment Variables](#environment-variables)
- [Logging](#logging)


## Overview
The Research Agent is designed to automate the process of searching for relevant academic papers and blog posts, scoring them based on relevance, saving results to a JSON file, and sharing the most relevant item on X. It supports both command-line execution via configuration and an interactive Streamlit web interface for user-friendly operation.

## Features
- **ArXiv Search**: Retrieves and scores academic papers from ArXiv based on a specified research field.
- **Web Search**: Fetches and scores blog posts using the Tavily search tool.
- **JSON Storage**: Saves search results in a structured JSON format in the `./saved` directory.
- **X Integration**: Posts the highest-scoring untweeted paper or blog post to X with a concise summary and hashtag.
- **Streamlit UI**: Provides an intuitive interface for configuring and running the agent.
- **Scheduling**: Supports periodic execution of the agent at user-defined intervals.
- **Logging**: Comprehensive logging for debugging and monitoring agent activities.

## Project Structure
```
├── agents/
│   └── build_agent.py         # Agent creation logic
├── prompts/
│   └── system_prompt.yaml    # System prompt templates for the agent
├── saved/                    # Directory for storing results and logs
│   ├── results.json          # Output JSON file for search results
│   ├── tweets.json           # Tracks tweeted URLs to avoid duplicates
│   └── agent_logs.log        # Log file for agent execution
├── tools/
│   └── tools.py              # Custom tools for ArXiv, Tavily, JSON handling, and X posting
├── utils/
│   └── utils.py              # Utility functions for logging and prompt loading
├── app.py                    # Streamlit application for the web interface
├── config.py                 # Configuration settings for the agent
├── main.py                   # Main script for running the agent
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nimadindar/SpatioTemporal-Agent.git
   cd SpatioTemporal-Agent
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables (see [Environment Variables](#environment-variables)).

## Usage

### Running via Command Line
1. Configure the agent by editing `config.py`:
   - Set `Field` to your desired research topic (e.g., "Spatio Temporal Point Process").
   - Adjust `ArxivMaxResults` and `TavilyMaxResults` (recommended: 5 to avoid LLM ontext window issues).
   - Specify `ModelName` (e.g., "gemini-2.0-flash") and `Temperature` (e.g., 0.0 for deterministic output).
   - Set `PromptVersion` (1 or 2) to select the desired prompt template.
   - Enable/disable `Verbose` logging.
   - Define `InvokeInput` for the agent's task.
2. Run the agent:
   ```bash
   python main.py
   ```
   This executes the agent once, fetching papers and blog posts, saving results to `./saved/results.json`, and posting to X.

### Running via Streamlit UI
1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the provided URL in your browser (e.g., `http://localhost:8501`).
3. Configure the agent via the web interface:
   - Enter the research field.
   - Select the model name and temperature.
   - Set the maximum number of results for ArXiv and Tavily searches.
   - Enable verbose logging if desired.
   - Optionally enable the scheduler to run the agent periodically.
   - Insert the api keys related to Google AI, X and tavily. Alternatively you can also pass them in a .env file and the code will load them for you.
   - Specify the agent task in the text area.
4. Click "Run Agent" to execute immediately or "Schedule Agent" for periodic execution.
5. View the output and logs in the UI.

## Configuration
The agent can be configured via:
- **config.py**: Edit this file for command-line execution. Key parameters include:
  - `PromptDir`: Path to the system prompt YAML file.
  - `Field`: Research topic.
  - `ArxivMaxResults` and `TavilyMaxResults`: Maximum results for searches.
  - `PromptVersion`: Selects the prompt template (1 or 2).
  - `ModelName`: Google Generative AI model.
  - `Temperature`: Controls response creativity.
  - `Verbose`: Enables detailed logging.
  - `InvokeInput`: Task description for the agent.
- **Streamlit UI**: Dynamically set these parameters through the web interface.

## Dependencies
Key dependencies are listed in `requirements.txt` and include:
- `langchain`
- `langchain-google-genai`
- `langchain-community`
- `tweepy`
- `tavily-python`
- `streamlit`
- `python-dotenv`
- `pyyaml`

Install them using:
```bash
pip install -r requirements.txt
```

## Environment Variables
Create a `.env` file in the project root with the following variables:
```bash
GOOGLE_API_KEY=your_google_api_key
X_API_KEY=your_x_consumer_key
X_API_KEY_SECRET=your_x_consumer_secret
X_ACCESS_TOKEN=your_x_access_token
X_ACCESS_TOKEN_SECRET=your_x_access_token_secret
TAVILY_API_KEY=your_tavily_api_key
```
Obtain these keys from:
- Google API: Google Cloud Console
- X API: X Developer Portal (https://developer.x.com)
- Tavily API: Tavily Dashboard (https://tavily.com)

## Logging
- Logs are saved to `./saved/agent_logs.log` with a maximum size of 5MB and up to 3 backup files.
- Verbose logging can be enabled in `config.py` or the Streamlit UI.
- Streamlit logs are displayed in the UI when verbose mode is enabled.