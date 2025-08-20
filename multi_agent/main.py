from langgraph.graph import StateGraph, START

from .agent_utils.utils import load_llm, State

from .ResearchTeam.arxiv_node import arxiv_node
from .ResearchTeam.blog_node import blog_node
from .ResearchTeam.gscholar_node import gscholar_node
from .PostingTeam.X_node import X_node

from multi_agent.config import AgentConfig

llm = load_llm(
    llm = AgentConfig.LLM,
    model=AgentConfig.ModelName,
    api_key=AgentConfig.GOOGLE_API_KEY
)

research_builder = StateGraph(State)

research_builder.add_node("arxiv", arxiv_node)
research_builder.add_node("blog", blog_node)
research_builder.add_node("gscholar", gscholar_node)
research_builder.add_node("X", X_node)

research_builder.add_edge(START, "arxiv")
research_builder.add_edge("arxiv", "blog")
research_builder.add_edge("blog", "gscholar")
research_builder.add_edge("gscholar", "X")

research_graph = research_builder.compile()

for s in research_graph.stream(
    {
        "messages": [
            ("user", f"Research about {AgentConfig.Field} and then post the results.")
        ],
    },
    {"recursion_limit": 150},
):
    print(s)
    print("---")