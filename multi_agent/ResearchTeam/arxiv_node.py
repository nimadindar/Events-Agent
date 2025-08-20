from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from ..agent_utils.utils import load_llm, State

from ..tools.tools import ArxivTool, save_to_json
from ..utils.utils import load_prompt_multi_agent
from multi_agent.config import AgentConfig

input_var = {
    "field": AgentConfig.Field,
    "arxiv_max_results": AgentConfig.ArxivMaxResults,
    "arxiv_min_usefulness": AgentConfig.ArxivMinUsefulness
}

arxiv_system_prompt = load_prompt_multi_agent(
    "arxiv",
    **input_var
)

llm = load_llm(
    llm = AgentConfig.LLM,
    model=AgentConfig.ModelName,
    api_key=AgentConfig.GOOGLE_API_KEY
)

arxiv_agent = create_react_agent(
    llm,
    tools=[ArxivTool, save_to_json],
    prompt=arxiv_system_prompt
)

def arxiv_node(state: State) -> Command:
    result = arxiv_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="arxiv")
            ]
        },
        goto="blog"
        # goto = END
    )

# Sample for running the agent seperately

# from langchain_core.messages import HumanMessage
# from langgraph.graph import StateGraph, START, END


# research_builder = StateGraph(State)
# research_builder.add_node("arxiv", arxiv_node)
# research_builder.add_edge(START, "arxiv")

# research_graph = research_builder.compile()


# for s in research_graph.stream(
#     {
#         "messages": [
#             ("user", f"Research about {AgentConfig.Field} and then give me the results.")
#         ],
#     },
#     {"recursion_limit": 150},
# ):
#     print(s)
#     print("---")