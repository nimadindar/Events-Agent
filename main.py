import os 
from dotenv import load_dotenv
load_dotenv()

from utils.utils import load_prompt
from agents.build_agent import BuildAgent
from tools.tools import save_json, post_to_X, ArxivTool

from langchain.tools import Tool
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI



def run_agent():

    # prompt = load_prompt(
    #     "./prompts/system_prompt.yaml",
    #     "Spatio Temporal Point Process"
    # )

    import yaml

    with open("./prompts/system_prompt.yaml", "r", encoding="utf-8") as f:
        prompt_data = yaml.safe_load(f)

    system_prompt_template = prompt_data["template"]
    # input_variables = prompt_data["input_variables"]

    env_vars = {
            "consumer_key": os.getenv("X_API_KEY"),
            "consumer_secret": os.getenv("X_API_KEY_SECRET"),
            "access_token": os.getenv("X_ACCESS_TOKEN"),
            "access_token_secret": os.getenv("X_ACCESS_TOKEN_SECRET"),
        }
    
    user_field = "Spatio Temporal point process"

    formatted_system_prompt = system_prompt_template.format(
        field=user_field,
        **env_vars
    )

    # prompt = ChatPromptTemplate.from_messages([
    # ("system", formatted_system_prompt),
    # ("human", "{input}"),
    # ("placeholder", "{agent_scratchpad}")])
    
    # prompt = ChatPromptTemplate.from_messages([
    # ("system", "Use Arxiv tool to fetch one paper about machine learning, create a single sentence \
    #  summary of this paper and post it on X using post_to_X tool. For Arxiv tool create a search query \
    #  and pass it to the function and for post_to_X use the following cresentials:\
    # consumer_key =  \
    # consumer_secret =  \
    # access_token =  \
    # access_token_secret = "),
    # ("human", "{input}"),
    # ("placeholder", "{agent_scratchpad}")])

    prompt = ChatPromptTemplate.from_messages([
    ("system", "use the tool which is the arxiv fetcher tool to get papers from Arxiv. Use the users given field."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")])

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    # llm = ChatOpenAI(
    #     model="mistralai/mistral-small-3.2-24b-instruct:free",
    #     openai_api_base="https://openrouter.ai/api/v1",
    #     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    # )

    tavily_tool = TavilySearch(
        max_results=5,
        api_key=os.getenv("TAVILY_API_KEY")
    )

    tools = [ArxivTool]


    agent = BuildAgent(llm, tools, prompt).build_agent()


    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    try:
        result = agent_executor.invoke({"input": ("Run the the system prmpt and obtain relevant papers.")})

    except Exception as e:
        print(f"Error running agent: {str(e)}")

if __name__ == "__main__":
    run_agent()


    


