import os
from functools import partial
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from .utils.utils import State

from .ResearchTeam import arxiv_node, blog_node, gscholar_node
from .PostingTeam import X_node



FIELD = "Spatio Temporal Point Process, Spatio Temporal, Point Process, Contextual dataset, Survey data"

# Model Config
MODEL_NAME = "gemini-2.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Agent path
next_state_arxiv = "blog"
next_state_blog = "gscholar"
next_state_gscholar = "X"
next_state_X = END


llm = ChatGoogleGenerativeAI(
                model=MODEL_NAME,
                temperature=0,
                google_api_key=GOOGLE_API_KEY
        )

def main():
    agent_builder = StateGraph(State)

    agent_builder.add_node("arxiv", partial(arxiv_node.arxiv_node, next_state=next_state_arxiv))
    agent_builder.add_node("blog", partial(blog_node.blog_node, next_state=next_state_blog))
    agent_builder.add_node("gscholar", partial(gscholar_node.gscholar_node, next_state=next_state_gscholar))
    agent_builder.add_node("X", partial(X_node.X_node, next_state=next_state_X))

    agent_builder.add_edge(START, "arxiv")
    agent_builder.add_edge("arxiv", "blog")
    agent_builder.add_edge("blog", "gscholar")
    agent_builder.add_edge("gscholar", "X")

    agent_graph = agent_builder.compile()

    for s in agent_graph.stream(
        {
            "messages": [
                ("user", f"Perform research about {FIELD} and save the results with your reasoning as based on your instructions. \
                 Then, load the saved results, and post on X and save the required files.")
            ],
        },
        {"recursion_limit": 150},
    ):
        print(s)
        print("---")


if __name__ == "__main__":
    main()
