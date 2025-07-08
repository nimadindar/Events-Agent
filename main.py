import os 
import time
import schedule
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from agents.build_agent import BuildAgent
from utils.utils import load_prompt, LoggingCallbackHandler
from tools.tools import save_json, post_to_X, ArxivTool, tavily_tool

from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


logging.basicConfig(
    filename='./saved/agent_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def run_agent():

    loaded_prompt = load_prompt(
        "./prompts/system_prompt.yaml",
        "Spatio Temporal Point Process, Point Process, STPP, Events, SpatioTemporal, Contextual Datasets (Satellite Data), Survey data"
    )

    prompt = ChatPromptTemplate.from_messages([
    ("system", loaded_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")])
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    tools = [ArxivTool, tavily_tool, post_to_X]

    agent = BuildAgent(llm, tools, prompt).build_agent()

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        callbacks=[LoggingCallbackHandler()]
    )

    try:
        result = agent_executor.invoke({"input": "Find a paper and tweet about it."})

    except Exception as e:
        print(f"Error running agent: {str(e)}")

if __name__ == "__main__":

    run_agent()
    # schedule.every(24).hours.do(run_agent)
    
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)