from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from ..agent_utils.utils import load_llm, State

from tools.tools import get_scholar_papers, save_to_json
from utils.utils import load_prompt_multi_agent
from config import AgentConfig

input_var = {
    "author_ids_list": AgentConfig.ScholarPages,
    "scholar_max_results": AgentConfig.ScholarMaxResults,
    "scholar_min_usefulness": AgentConfig.ScholarMinUsefulness,
    "serp_api_key": AgentConfig.SERP_API_KEY
}

gscholar_system_prompt = load_prompt_multi_agent(
    "gscholar",
    **input_var
)

llm = load_llm(
    llm = AgentConfig.LLM,
    model=AgentConfig.ModelName,
    api_key=AgentConfig.GOOGLE_API_KEY
)

gscholar_agent = create_react_agent(
    llm,
    tools=[get_scholar_papers, save_to_json],
    prompt=gscholar_system_prompt
)

def gscholar_node(state: State) -> Command:
    result = gscholar_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="gscholar")
            ]
        },
        goto="X"
    )
