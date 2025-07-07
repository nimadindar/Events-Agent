import yaml
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import PromptTemplate


def load_yaml(input_path:str) -> str:

    with open(input_path, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

    return prompt_config


def load_prompt(input_path: str, field_input: str) -> PromptTemplate:
    prompt_config = load_yaml(input_path)

    return PromptTemplate(
        input_variables=[
            "field",
            "consumer_key",
            "consumer_secret",
            "access_token",
            "access_token_secret",
            "agent_scratchpad"  
        ],
        partial_variables={
            "field": field_input,
            "consumer_key": os.getenv("X_API_KEY", "your_consumer_key"),
            "consumer_secret": os.getenv("X_API_KEY_SECRET", "your_consumer_secret"),
            "access_token": os.getenv("X_ACCESS_TOKEN", "your_access_token"),
            "access_token_secret": os.getenv("X_ACCESS_TOKEN_SECRET", "your_access_token_secret")
        },
        template=prompt_config["template"]
    )