import re
import yaml
import logging
import traceback
from typing import Any
from logging.handlers import RotatingFileHandler

from multi_agent.config import AgentConfig

from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_buffer = []

    def emit(self, record):
        log_entry = self.format(record)
        self.log_buffer.append(log_entry)

    def get_logs(self):
        return self.log_buffer  

    def clear_logs(self):
        self.log_buffer = []
    
class LoggingCallbackHandler(BaseCallbackHandler):
    def __init__(self, logger):
        self.logger = logger

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Log when LLM starts processing."""
        self.logger.debug(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response, **kwargs):
        """Log LLM output."""
        self.logger.debug(f"LLM response: {response.generations}")

    def on_tool_start(self, tool, input_str, **kwargs):
        """Log when a tool is called."""
        self.logger.info(f"Tool '{tool.name}' called with input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        """Log tool output."""
        self.logger.info(f"Tool output: {output}")

    def on_tool_error(self, error, **kwargs):
        """Log tool errors."""
        self.logger.error(f"Tool error: {str(error)}")
        self.logger.debug(f"Tool error stack trace: {traceback.format_exc()}")

    def on_agent_action(self, action, **kwargs):
        """Log intermediate agent actions."""
        self.logger.debug(f"Agent action: {action}")

    def on_agent_finish(self, finish, **kwargs):
        """Log when agent finishes."""
        clean_output = str(finish.return_values).encode('ascii', 'replace').decode()
        self.logger.info(f"Agent finished with output: {clean_output}")

class StreamToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):

        for line in buf.rstrip().splitlines():
            line = line.rstrip()
            if line:
                cleaned_line = line.encode("utf-8", errors="replace").decode("utf-8")
                self.logger.log(self.level, f"Console: {cleaned_line}")

    def flush(self):
        pass  


def load_yaml(input_path:str) -> str:

    with open(input_path, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

    return prompt_config


def load_prompt(input_path: str, template_id: int) -> PromptTemplate:
    
    prompt_config = load_yaml(input_path)

    system_prompt_template = prompt_config[f"template_{template_id}"]

    env_vars = {
            "field": AgentConfig.Field,
            "arxiv_max_results": AgentConfig.ArxivMaxResults,
            "arxiv_min_usefulness": AgentConfig.ArxivMinUsefulness,
            "tavily_api_key": AgentConfig.TAVILY_API_KEY,
            "tavily_max_results": AgentConfig.TavilyMaxResults,
            "blog_min_usefulness": AgentConfig.BlogMinUsefulness,
            "author_ids_list": AgentConfig.ScholarPages,
            "scholar_max_results": AgentConfig.ScholarMaxResults,
            "serp_api_key": AgentConfig.SERP_API_KEY,
            "scholar_min_usefulness": AgentConfig.ScholarMinUsefulness,
            "consumer_key": AgentConfig.X_API_KEY,
            "consumer_secret": AgentConfig.X_API_KEY_SECRET,
            "access_token": AgentConfig.X_ACCESS_TOKEN,
            "access_token_secret": AgentConfig.X_ACCESS_TOKEN_SECRET,
        }
    
    formatted_system_prompt = system_prompt_template.format(**env_vars)

    return formatted_system_prompt

def setup_logging():
    logger = logging.getLogger('AgentLogger')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.stream.reconfigure(encoding='utf-8')

    file_handler = RotatingFileHandler(
        './saved/agent_logs.log', maxBytes=5*1024*1024, backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger  

def update_agent_config(field, arxiv_max_results, arxiv_min_usefulness,
                        tavily_max_results, blog_min_usefulness,
                        scholar_max_results, scholar_min_usefulness, scholar_user_ID,
                        model_name, temperature, verbose, invoke_input,
                        GOOGLE_API_KEY, X_API_KEY, X_API_KEY_SECRET,
                        X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET, TAVILY_API_KEY, SERP_API_KEY):
    """
    Update the global AgentConfig attributes dynamically.
    """
    from multi_agent.config import AgentConfig as GlobalAgentConfig
    
    GlobalAgentConfig.Field = field
    GlobalAgentConfig.ArxivMaxResults = arxiv_max_results
    GlobalAgentConfig.ArxivMinUsefulness = arxiv_min_usefulness
    GlobalAgentConfig.TavilyMaxResults = tavily_max_results
    GlobalAgentConfig.BlogMinUsefulness = blog_min_usefulness
    GlobalAgentConfig.ScholarPages = scholar_user_ID
    GlobalAgentConfig.ScholarMaxResults = scholar_max_results
    GlobalAgentConfig.ScholarMinUsefulness = scholar_min_usefulness
    GlobalAgentConfig.ModelName = model_name
    GlobalAgentConfig.Temperature = temperature
    GlobalAgentConfig.Verbose = verbose
    GlobalAgentConfig.InvokeInput = invoke_input
    GlobalAgentConfig.GOOGLE_API_KEY = GOOGLE_API_KEY 
    GlobalAgentConfig.X_API_KEY = X_API_KEY
    GlobalAgentConfig.X_API_KEY_SECRET = X_API_KEY_SECRET
    GlobalAgentConfig.X_ACCESS_TOKEN = X_ACCESS_TOKEN
    GlobalAgentConfig.X_ACCESS_TOKEN_SECRET = X_ACCESS_TOKEN_SECRET
    GlobalAgentConfig.TAVILY_API_KEY = TAVILY_API_KEY
    GlobalAgentConfig.SERP_API_KEY = SERP_API_KEY

    return GlobalAgentConfig


def normalize_url(url):
    """ 
    This function uses a regular expression to match the arXiv ID format.
    Example:
    input url: http://arxiv.org/abs/2211.11179v1
    output: 2211.11179

    The other urls will be returned as is.    
    """
    match = re.search(r"(\d{4}\.\d{4,5})(v\d)?", url)
    if match:
        return match.group(1)  
    return url


def load_prompt_multi_agent(node_name: str, **env_vars: Any) -> str:
    prompt_config = load_yaml(f"./multi_agent/prompts/{node_name}_node_prompt.yaml")
    system_prompt_template = prompt_config["prompt"]
    return system_prompt_template.format(**env_vars)