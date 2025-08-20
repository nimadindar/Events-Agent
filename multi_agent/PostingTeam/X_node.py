from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from ..agent_utils.utils import load_llm, State

from ..tools.tools import post_to_X, json_reader_tool
from ..utils.utils import load_prompt_multi_agent
from multi_agent.config import AgentConfig

input_var = {
    "consumer_key": AgentConfig.X_API_KEY,
    "consumer_secret": AgentConfig.X_API_KEY_SECRET,
    "access_token": AgentConfig.X_ACCESS_TOKEN,
    "access_token_secret": AgentConfig.X_ACCESS_TOKEN_SECRET
}

X_system_prompt = load_prompt_multi_agent(
    "X",
    **input_var
)

llm = load_llm(
    llm = AgentConfig.LLM,
    model=AgentConfig.ModelName,
    api_key=AgentConfig.GOOGLE_API_KEY
)

X_agent = create_react_agent(
    llm,
    tools=[post_to_X, json_reader_tool],
    prompt=X_system_prompt
)

def X_node(state: State) -> Command:
    result = X_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="X")
            ]
        }
        # graph ends here
    )
