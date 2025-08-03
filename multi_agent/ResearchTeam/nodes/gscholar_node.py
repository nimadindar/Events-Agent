from typing import List, Optional, Literal

from langchain_core.messages import HumanMessage
# from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from ...agent_utils.make_supervisor_node import State
from ...agent_utils.utils import load_llm
from tools.tools import get_scholar_papers, save_to_json
from config import AgentConfig

llm = load_llm(
    llm = AgentConfig.LLM,
    model=AgentConfig.ModelName,
    api_key=AgentConfig.GOOGLE_API_KEY
)

gscholar_agent = create_react_agent(llm, tools=[get_scholar_papers, save_to_json])

    
def gscholar_node(state: State) -> Command[Literal["supervisor"]]:
    result = gscholar_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="gscholar")
            ]
        },
        goto="supervisor",
    )
