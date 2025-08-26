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

from ..tools.research_tools import get_scholar_papers, save_to_json
from ..utils.utils import State, DebugHandler


# Path to system prompt
GSCHOLAR_PROMPT_DIR = "./multi_agent/prompts/gscholar_node_prompt.yaml"

# Prompt Config
FIELD = "Spatio Temporal Point Process, Spatio Temporal, Point Process, Contextual dataset, Survey data"
GSCHOLAR_MAX_RESULTS = 1
GSCHOLAR_MIN_USEFULNESS = 60
AUTHOR_IDS = ["Wnxq0mgAAAAJ"]

# Tavily Config
SERP_API_KEY = os.getenv("SERP_API_KEY")

# Model Config
MODEL_NAME = "gemini-2.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEXT_STATE = END

INPUT_VAR = {
    "field": FIELD, 
    "author_ids_list": AUTHOR_IDS,
    "scholar_max_results": GSCHOLAR_MAX_RESULTS,
    "scholar_min_usefulness": GSCHOLAR_MIN_USEFULNESS,
    "serp_api_key": SERP_API_KEY
}

with open(GSCHOLAR_PROMPT_DIR, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

GSCHOLAR_SYSTEM_PROMPT = prompt_config["prompt"].format(**INPUT_VAR)

llm = ChatGoogleGenerativeAI(
                model=MODEL_NAME,
                temperature=0,
                google_api_key=GOOGLE_API_KEY
        )

gscholar_agent = create_react_agent(
    llm,
    tools=[get_scholar_papers, save_to_json],
    prompt=GSCHOLAR_SYSTEM_PROMPT
)

def gscholar_node(state: State, next_state) -> Command:
    result = gscholar_agent.invoke(
        state,
        config={"callbacks": [DebugHandler()], "run_name": "gscholar_agent"})
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="gscholar")
            ]
        },
        goto=next_state
    )

def gscholar_main(next_state):

    research_builder = StateGraph(State)
    research_builder.add_node("gscholar", partial(gscholar_node, next_state=next_state))
    research_builder.add_edge(START, "gscholar")

    research_graph = research_builder.compile()

    for s in research_graph.stream(
        {
            "messages": [
                ("user", f"Search for relevant papers about {FIELD} on the given google scholar pages and then save the results as a json object based on the instructions you have.")
            ],
        },
        {"recursion_limit": 150},
    ):
        print(s)
        print("---")

if __name__ == "__main__":
    gscholar_main(NEXT_STATE)