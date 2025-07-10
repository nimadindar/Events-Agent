import os 
import sys
import time
import logging
import schedule
import traceback
from dotenv import load_dotenv
load_dotenv()

from agents.build_agent import BuildAgent
from tools.tools import save_to_json, post_to_X, ArxivTool, tavily_tool
from utils.utils import load_prompt, setup_logging, LoggingCallbackHandler, StreamToLogger

from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def run_agent():

    logger = setup_logging()
    logger.info("Starting agent execution")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    logger.info("Redirected stdout and stderr to logger")

    callback_handler = LoggingCallbackHandler(logger)
    logger.info("Initialized LoggingCallbackHandler")

    try:

        logger.info("Loading prompt from './prompts/system_prompt.yaml'")
        loaded_prompt = load_prompt(
            "./prompts/system_prompt.yaml",
            "Spatio Temporal Point Process",
            2
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
            model="gemini-2.0-flash",
            temperature=0.0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        logger.debug("LLM initialized with model=gemini-2.0-flash, temperature=0.0")

        tools = [ArxivTool, tavily_tool, save_to_json]
        logger.info(f"Tools initialized: {[tool.name for tool in tools]}")

        logger.info("Building agent")
        agent = BuildAgent(llm, tools, prompt).build_agent()
        logger.debug(f"Agent built: {type(agent).__name__}")

        logger.info("Creating AgentExecutor")
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            callbacks=[callback_handler]
        )
        logger.debug("AgentExecutor created with verbose=True")

        logger.info("Invoking agent with input: 'Find a paper and tweet about it.'")
        result = agent_executor.invoke({"input": "Search for related papers and blog posts and save an organized json file."})
        logger.info("Agent execution completed")
        logger.debug(f"Agent result: {result}")


        logger.info(f"Final output: {result.get('output', 'No output returned')}")
        return result

    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        raise
    
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger.info("Restored original stdout and stderr")

if __name__ == "__main__":
    run_agent()
    # schedule.every(24).hours.do(run_agent)
    
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)