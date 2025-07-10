import os
import yaml
import logging
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler



def load_yaml(input_path:str) -> str:

    with open(input_path, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

    return prompt_config


def load_prompt(input_path: str, field_input: str) -> PromptTemplate:
    
    prompt_config = load_yaml(input_path)

    system_prompt_template = prompt_config["template"]

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