from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain.agents import create_react_agent
from langchain_core.tools import BaseTool


class BuildAgent:
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        prompt: BasePromptTemplate,
    ):
        self.llm = llm
        self.tools = tools
        self.prompt = prompt

    def build_agent(self):

        return create_react_agent(self.llm, self.tools, self.prompt)