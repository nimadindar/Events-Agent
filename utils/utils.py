import yaml

def load_yaml(input_path:str) -> str:

    with open(input_path, "r") as f:
        prompt_config = yaml.safe_load(f)

    return prompt_config