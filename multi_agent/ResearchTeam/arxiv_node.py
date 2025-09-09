import os
import yaml
from datetime import datetime
from functools import partial
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langchain.tools import tool

from ..tools.research_tools import ArxivTool, save_to_json
from ..utils.utils import State, DebugHandler


# Path to system prompt
ARXIV_PROMPT_DIR = "./multi_agent/prompts/arxiv_node_prompt.yaml"

# Prompt Config
FIELD = "Spatio Temporal Point Process, Spatio Temporal, Point Process, Contextual dataset, Survey data"
ARXIV_MAX_RESULTS = 10
ARXIV_MIN_USEFULNESS = 60

# Time Frame used to get data
START_DATE = "20250101000000"
END_DATE = datetime.now().strftime("%Y%m%d%H%M%S")  # returns this moment

# Model Config
MODEL_NAME = "gemini-2.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEXT_STATE = END


@tool("arxiv_tool")
def arxiv_tool(query:str) -> str:
    """arxiv results filtered based on year"""
    return ArxivTool.func(f"{query} AND submittedDate: [{START_DATE} TO {END_DATE}]", max_results= ARXIV_MAX_RESULTS)


INPUT_VAR = {
    "field": FIELD,
    "arxiv_min_usefulness": ARXIV_MIN_USEFULNESS
}

with open(ARXIV_PROMPT_DIR, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

ARXIV_SYSTEM_PROMPT = prompt_config["prompt"].format(**INPUT_VAR)

llm = ChatGoogleGenerativeAI(
                model=MODEL_NAME,
                temperature=0,
                google_api_key=GOOGLE_API_KEY
        )

arxiv_agent = create_react_agent(
    llm,
    tools=[arxiv_tool, save_to_json],
    prompt=ARXIV_SYSTEM_PROMPT
)

def arxiv_node(state: State, next_state) -> Command:
    result = arxiv_agent.invoke(
        state,
        config={"callbacks": [DebugHandler()], "run_name": "arxiv_agent"})
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="arxiv")
            ]
        },
        goto=next_state
    )


def arxiv_main(next_state):

    research_builder = StateGraph(State)
    research_builder.add_node("arxiv", partial(arxiv_node, next_state=next_state))
    research_builder.add_edge(START, "arxiv")

    research_graph = research_builder.compile()

    for s in research_graph.stream(
        {
            "messages": [
                ("user", f"Search for relevant papers about {FIELD} on arxiv and then save the results as a json object based on the instructions you have.")
            ],
        },
        {"recursion_limit": 150},
    ):
        print(s)
        print("---")

if __name__ == "__main__":
    arxiv_main(NEXT_STATE)