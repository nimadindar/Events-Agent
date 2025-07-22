import os 
import time
import schedule
import traceback

from config import AgentConfig
from agents.build_agent import BuildAgent
from tools.tools import save_to_json, post_to_X, ArxivTool, tavily_tool, json_reader_tool, get_scholar_papers
from utils.utils import load_prompt, setup_logging, LoggingCallbackHandler

from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def run_agent():

    logger = setup_logging()

    callback_handler = LoggingCallbackHandler(logger)

    try:
        loaded_prompt = load_prompt(
            AgentConfig.PromptDir,
            AgentConfig.PromptVersion
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", loaded_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        llm = ChatGoogleGenerativeAI(
            model=AgentConfig.ModelName,
            temperature=AgentConfig.Temperature,
            google_api_key=AgentConfig.GOOGLE_API_KEY
        )

        tools = [ArxivTool, tavily_tool, get_scholar_papers, save_to_json, json_reader_tool, post_to_X]

        agent = BuildAgent(llm, tools, prompt).build_agent()

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=AgentConfig.Verbose,
            callbacks=[callback_handler]
        )

        result = agent_executor.invoke({"input": AgentConfig.InvokeInput})

        return result

    except Exception as e:
        raise
    
if __name__ == "__main__":
    run_agent()

    # schedule.every(2).hours.do(run_agent)
    
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)