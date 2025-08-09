from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from ..agent_utils.utils import load_llm, State

from tools.tools import tavily_tool, save_to_json
from utils.utils import load_prompt_multi_agent
from config import AgentConfig

input_var = {
    "field": AgentConfig.Field,
    "tavily_api_key": AgentConfig.TAVILY_API_KEY,
    "tavily_max_results": AgentConfig.TavilyMaxResults,
    "blog_min_usefulness": AgentConfig.BlogMinUsefulness
}

blog_system_prompt = load_prompt_multi_agent(
    "blog",
    **input_var
)

llm = load_llm(
    llm = AgentConfig.LLM,
    model=AgentConfig.ModelName,
    api_key=AgentConfig.GOOGLE_API_KEY
)

blog_agent = create_react_agent(
    llm,
    tools=[tavily_tool, save_to_json],
    prompt=blog_system_prompt
)

def blog_node(state: State) -> Command:
    result = blog_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="blog")
            ]
        },
        goto="gscholar"
    )
