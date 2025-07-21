import os
from dotenv import load_dotenv
load_dotenv()

class AgentConfig:
    PromptDir = "./prompts/system_prompt.yaml"      # path to system prompt yaml file

    Field = "Spatio Temporal Point Process"         # the field that agent conducts research

    ArxivMaxResults = 5                             # maximum number of retreived papers from arxiv tool !!! HINT: Lager number might overload LLM's contex window !!! 
    ArxivMinUsefulness = 70                         # minimum score required to save an Arxiv paper

    TavilyMaxResults = 5                            # maximum number of retrieved search results from tavily search tool !!! HINT: Lager number might overload LLM's contex window !!! 
    BlogMinUsefulness = 70                          # minimum usefulness score for saving the blog posts

    # ScholarPages = ["2UgplA4AAAAJ","WoqSEpYAAAAJ", "e27EmFsAAAAJ", "Wnxq0mgAAAAJ"]  # google scholar pages of ["Yanan Xin", "Sebastian J. Vollmer", "Gerrit Grossmann", "Seth Flaxman"]
    ScholarPages = ["Wnxq0mgAAAAJ"]                 # google scholar page of ["Seth Flaxman"]
    ScholarMaxResults = 1                           # maximum number of retrieved google scholar results !!! HINT: Lager number might overload LLM's contex window !!!
    ScholarMinUsefulness = 50                       # minimum usefulness score for saving google scholar papers
    
    PromptVersion = 4                               # system prompt version
    
    ModelName = "gemini-2.5-pro"                  # only ChatGoogleGenerativeAI models are supported
    Temperature = 0.0                               # the level of creativity in LLM responses. Higher temp causes creative but unexpected behaviour
    
    Verbose = True                                  # LLM log the intermediate states or not
    
    InvokeInput = "Search for the related papers and blog posts and save an organized json file, and finally post a paper or blog post with the highest score to X."

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")   
    SERP_API_KEY = os.getenv("SERP_API_KEY") 
    X_API_KEY = os.getenv("X_API_KEY")
    X_API_KEY_SECRET = os.getenv("X_API_KEY_SECRET")
    X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
    X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET")
    
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

