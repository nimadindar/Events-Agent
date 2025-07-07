import os 
from dotenv import load_dotenv
load_dotenv()

from utils.utils import load_prompt
from agents.build_agent import BuildAgent
from tools.tools import save_json, post_to_X, ArxivTool

from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI

def run_agent():

    prompt = load_prompt(
        "./prompts/system_prompt.yaml",
        "Point Process, Spatio Temporal Point Process, Events, SpatioTemporal, Contexual Datasets (Satellite Images), Survey Data"
    )


    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


    tavily_tool = TavilySearch(
        max_results=5,
        api_key=os.getenv("TAVILY_API_KEY")
    )


    tools = [ArxivTool, tavily_tool, save_json, post_to_X]


    agent = BuildAgent(llm, tools, prompt).build_agent()


    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )


    try:
        result = agent_executor.invoke({
            "input": (
                "Execute the workflow to search for ArXiv papers and blog posts related to the field, "
                "save the results as JSON, and post the top 2-3 entries to X. "
                "Use the provided X API credentials in the prompt for posting. "
                "Ensure all steps follow the system prompt instructions."
            )
        })
        print("Agent execution result:", result["output"])
    except Exception as e:
        print(f"Error running agent: {str(e)}")

if __name__ == "__main__":
    run_agent()