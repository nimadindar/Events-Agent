import os
from dotenv import load_dotenv
load_dotenv()

class AgentConfig:
    PromptDir = "./prompts/system_prompt.yaml"      # path to system prompt yaml file

    Field = "Spatio Temporal Point Process"         # the field that agent conducts research

    ArxivMaxResults = 5                             # maximum number of retreived papers from arxiv tool !!! HINT: Lager number might overload LLM's contex window !!! 
    TavilyMaxResults = 5                            # maximum number of retrieved search results from tavily search tool !!! HINT: Lager number might overload LLM's contex window !!! 
    
    PromptVersion = 2                               # system prompt version
    
    ModelName = "gemini-2.0-flash"                  # only ChatGoogleGenerativeAI models are supported
    Temperature = 0.0                               # the level of creativity in LLM responses. Higher temp causes creative but unexpected behaviour
    
    Verbose = True                                  # LLM log the intermediate states or not
    
    InvokeInput = "Search for the related papers and blog posts and save an organized json file, and finally post a paper or blog post with the highest score to X."

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")    
    X_API_KEY = os.getenv("X_API_KEY")
    X_API_KEY_SECRET = os.getenv("X_API_KEY_SECRET")
    X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
    X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
    
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

