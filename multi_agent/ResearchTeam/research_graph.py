from langgraph.graph import StateGraph, START

from .nodes.arxiv_node import arxiv_node
from .nodes.blog_node import blog_node
from .nodes.gscholar_node import gscholar_node

from ..agent_utils.make_supervisor_node import State, make_supervisor_node
from ..agent_utils.utils import load_llm

from config import AgentConfig

llm = load_llm(
    llm = AgentConfig.LLM,
    model=AgentConfig.ModelName,
    api_key=AgentConfig.GOOGLE_API_KEY
)


research_supervisor_node = make_supervisor_node(
    llm=llm,
    members=["arxiv", "blog", "gscholar"]
)

research_builder = StateGraph(State)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("arxiv", arxiv_node)
research_builder.add_node("blog", blog_node)
research_builder.add_node("gscholar", gscholar_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()

from IPython.display import Image, display

with open("graph.png", "wb") as f:
    f.write(research_graph.get_graph().draw_mermaid_png())

print("Saved to graph.png — open this file to view the image.")