import os
import yaml
import logging
import traceback
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

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
        self.logger.info(f"Agent finished with output: {finish.return_values}")

class StreamToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):

        for line in buf.rstrip().splitlines():
            line = line.rstrip()
            if line:
                self.logger.log(self.level, f"Console: {line}")

    def flush(self):
        pass  


def load_yaml(input_path:str) -> str:

    with open(input_path, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

    return prompt_config


def load_prompt(input_path: str, field_input: str, template_id: int) -> PromptTemplate:
    
    prompt_config = load_yaml(input_path)

    system_prompt_template = prompt_config[f"template_{template_id}"]

    env_vars = {
            "consumer_key": os.getenv("X_API_KEY"),
            "consumer_secret": os.getenv("X_API_KEY_SECRET"),
            "access_token": os.getenv("X_ACCESS_TOKEN"),
            "access_token_secret": os.getenv("X_ACCESS_TOKEN_SECRET"),
        }
    
    formatted_system_prompt = system_prompt_template.format(
        field=field_input,
        **env_vars
    )

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

    file_handler = RotatingFileHandler(
        './saved/agent_logs.log', maxBytes=5*1024*1024, backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger