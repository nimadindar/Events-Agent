from langchain.prompts import PromptTemplate

from utils.utils import load_yaml
from agents.build_agent import BuildAgent

prompt_config = load_yaml("./prompts/system_prompt.yaml")

prompt = PromptTemplate(
    input_variables=prompt_config['input_variables'],
    template=prompt_config['template']
)

print(prompt)
