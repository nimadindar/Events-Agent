import yaml

from langchain.prompts import PromptTemplate


def load_yaml(input_path:str) -> str:

    with open(input_path, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

    return prompt_config


def load_prompt(input_path : str, input_variables : str) -> str:

    prompt_config = load_yaml(input_path)

    prompt = PromptTemplate(
        input_variables=prompt_config['input_variables'],
        template=prompt_config['template'])
    
    formatted_prompt = prompt.format(field = input_variables)

    return formatted_prompt
