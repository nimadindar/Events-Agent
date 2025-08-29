import os
import yaml
from functools import partial
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langchain.tools import tool

from ..tools.research_tools import tavily_tool, save_to_json
from ..utils.utils import State, DebugHandler


# Output file name: The file will be saved in ./saved/output_file_name.json
FILE_NAME = "blog_results.json"

# Path to system prompt
BLOG_PROMPT_DIR = "./multi_agent/prompts/blog_node_prompt.yaml"

# Prompt Config
FIELD = "Spatio Temporal Point Process, Spatio Temporal, Point Process, Contextual dataset, Survey data"
BLOG_MAX_RESULTS = 5
BLOG_MIN_USEFULNESS = 60

# Tavily Config
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model Config
MODEL_NAME = "gemini-2.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEXT_STATE = END

DOMAINS_INCLUDED = [
    "journals.plos.org",
    "spatialedge.co",
    "events2025.github.io",
    "spacetimecausality.github.io"
]


# The base tools are edited for handling file names on code side for deterministic results.
@tool("blog_save_to_json")
def blog_save_to_json(content: dict) -> str:
    """Save blog results to a specified path"""
    return save_to_json.func(content=content, file_name=FILE_NAME)

@tool("blog_search")
def blog_search(query):
     """search web using tavily tool"""
     return tavily_tool.func(query,TAVILY_API_KEY,DOMAINS_INCLUDED,BLOG_MAX_RESULTS)

INPUT_VAR = {
    "field": FIELD,
    "blog_min_usefulness": BLOG_MIN_USEFULNESS
}

with open(BLOG_PROMPT_DIR, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

BLOG_SYSTEM_PROMPT = prompt_config["prompt"].format(**INPUT_VAR)

llm = ChatGoogleGenerativeAI(
                model=MODEL_NAME,
                temperature=0,
                google_api_key=GOOGLE_API_KEY
        )

blog_agent = create_react_agent(
    llm,
    tools=[blog_search, blog_save_to_json],
    prompt=BLOG_SYSTEM_PROMPT
)

def blog_node(state: State, next_state) -> Command:
    result = blog_agent.invoke(
        state,
        config={"callbacks": [DebugHandler()], "run_name": "blog_agent"})
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="blog")
            ]
        },
        goto=next_state
    )

def blog_main(next_state):

    research_builder = StateGraph(State)
    research_builder.add_node("blog", partial(blog_node, next_state=next_state))
    research_builder.add_edge(START, "blog")

    research_graph = research_builder.compile()

    for s in research_graph.stream(
        {
            "messages": [
                ("user", f"Search for relevant blog posts about {FIELD} on the websites and then save the results as a json object based on the instructions you have.")
            ],
        },
        {"recursion_limit": 150},
    ):
        print(s)
        print("---")

if __name__ == "__main__":
    blog_main(NEXT_STATE)