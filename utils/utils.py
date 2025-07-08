import os
import yaml
import logging
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

logger = logging.getLogger(__name__)
class LoggingCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        logger.info(f"[LLM Start] Prompt: {prompts}")

    def on_llm_end(self, response, **kwargs):
        logger.info(f"[LLM End] Response: {response.generations}")

    def on_tool_start(self, tool, input_str, **kwargs):
        logger.info(f"[Tool Start] Tool: {tool}, Input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        logger.info(f"[Tool End] Output: {output}")

    def on_chain_error(self, error, **kwargs):
        logger.error(f"[Error] {error}")