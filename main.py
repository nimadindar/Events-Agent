import os 
import time
import schedule
import traceback

from config import AgentConfig
from agents.build_agent import BuildAgent
from tools.tools import save_to_json, post_to_X, ArxivTool, tavily_tool, json_reader_tool
from utils.utils import load_prompt, setup_logging, LoggingCallbackHandler

from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def run_agent():

    logger = setup_logging()
    logger.info("Starting a new agent execution")

    callback_handler = LoggingCallbackHandler(logger)
    logger.info("Initialized LoggingCallbackHandler")

    try:

        logger.info(f"Loading prompt from {AgentConfig.PromptDir}")
        loaded_prompt = load_prompt(
            AgentConfig.PromptDir,
            AgentConfig.Field,
            AgentConfig.ArxivMaxResults,
            AgentConfig.TavilyMaxResults,
            AgentConfig.PromptVersion
        )
        logger.debug(f"Loaded prompt content: {loaded_prompt}")

        logger.info("Creating ChatPromptTemplate")
        prompt = ChatPromptTemplate.from_messages([
            ("system", loaded_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        logger.debug(f"Prompt template created: {prompt}")

        logger.info("Initializing ChatGoogleGenerativeAI model")
        llm = ChatGoogleGenerativeAI(
            model=AgentConfig.ModelName,
            temperature=AgentConfig.Temperature,
            google_api_key=AgentConfig.GOOGLE_API_KEY
        )
        logger.debug(f"LLM initialized with model={AgentConfig.ModelName}, temperature={AgentConfig.Temperature}")

        tools = [ArxivTool, tavily_tool, save_to_json, json_reader_tool, post_to_X]
        logger.info(f"Tools initialized: {[tool.name for tool in tools]}")

        logger.info("Building agent")
        agent = BuildAgent(llm, tools, prompt).build_agent()
        logger.debug(f"Agent built: {type(agent).__name__}")

        logger.info("Creating AgentExecutor")
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=AgentConfig.Verbose,
            callbacks=[callback_handler]
        )
        logger.debug(f"AgentExecutor created with verbose={AgentConfig.Verbose}")

        logger.info(f"Invoking agent with input: {AgentConfig.InvokeInput}")
        result = agent_executor.invoke({"input": AgentConfig.InvokeInput})
        logger.info("Agent execution completed")
        logger.debug(f"Agent result: {result}")


        logger.info(f"Final output: {result.get('output', 'No output returned')}")
        return result

    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        raise
    
if __name__ == "__main__":
    run_agent()

    # print(json_reader_tool())
    # schedule.every(24).hours.do(run_agent)
    
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)