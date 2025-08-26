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

from ..tools.posting_tools import post_to_X, fetch_filtered_items, save_tweet
from ..utils.utils import State, DebugHandler


# Path tp system prompt
X_PROMPT_DIR = "./multi_agent/prompts/X_node_prompt.yaml"

# Prompt config
FIELD = "Spatio Temporal Point Process, Spatio Temporal, Point Process, Contextual dataset, Survey data"
SOURCE = "all"
DATE = "01.01.2025"
X_MIN_USEFULNESS = 75

# Model Config
MODEL_NAME = "gemini-2.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEXT_STATE = END

INPUT_VAR = {
    "Field": FIELD,
    "source":SOURCE,
    "date": DATE,
    "X_min_usefulness": X_MIN_USEFULNESS,
    "consumer_key": os.getenv("X_API_KEY"),
    "consumer_secret": os.getenv("X_API_KEY_SECRET"),
    "access_token": os.getenv("X_ACCESS_TOKEN"),
    "access_token_secret": os.getenv("X_ACCESS_TOKEN_SECRET")
}

with open(X_PROMPT_DIR, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

X_SYSTEM_PROMPT = prompt_config["prompt"].format(**INPUT_VAR)

llm = ChatGoogleGenerativeAI(
                model=MODEL_NAME,
                temperature=0,
                google_api_key=GOOGLE_API_KEY
        )

X_agent = create_react_agent(
    llm,
    tools=[fetch_filtered_items, post_to_X, save_tweet],
    prompt=X_SYSTEM_PROMPT
)

def X_node(state: State, next_state) -> Command:
    result = X_agent.invoke(
        state,
        config={"callbacks": [DebugHandler()], "run_name": "X_agent"})
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="X")
            ]
        },
        goto=next_state
    )

def X_main(next_state):

    posting_builder = StateGraph(State)
    posting_builder.add_node("X", partial(X_node, next_state=next_state))
    posting_builder.add_edge(START, "X")

    posting_graph = posting_builder.compile()

    for s in posting_graph.stream(
        {
            "messages": [
                ("user", f"Load the most related enteries to {FIELD}, then select one and post a tweet about it according to the guidlines you have.")
            ],
        },
        {"recursion_limit": 150},
    ):
        print(s)
        print("---")

if __name__ == "__main__":
    X_main(NEXT_STATE)