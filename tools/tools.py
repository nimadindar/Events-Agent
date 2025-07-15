import os
import json
import tweepy
from pathlib import Path
from datetime import datetime
from typing import Union, List

from langchain.tools import tool
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import ArxivLoader

from config import AgentConfig

@tool
def json_reader_tool() -> str:
    """
    Reads './saved/results.json', selects the highest-scoring untweeted result,
    and tracks tweeted URLs in './saved/tweets.json' to avoid duplicates.

    Returns:
        str: A JSON string of the selected item or an error message.
    """
    json_file_path = "./saved/results.json"
    tweeted_file_path = "./saved/tweets.json"

    try:
        if not os.path.exists(json_file_path):
            return json.dumps({"error": f"JSON file '{json_file_path}' not found"})

        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "results" not in data:
            return json.dumps({"error": "Invalid format: expected a dictionary with a 'results' key"})

        all_results = data["results"]
        if not all_results:
            return json.dumps({"error": "No results found in 'results.json'"})

        tweeted_urls = set()
        tweeted_list = []

        if os.path.exists(tweeted_file_path):
            with open(tweeted_file_path, "r", encoding="utf-8") as f:
                tweeted_data = json.load(f)
                if isinstance(tweeted_data, dict):
                    tweeted_list = tweeted_data.get("tweeted", [])
                elif isinstance(tweeted_data, list):
                    tweeted_list = tweeted_data  # fallback
                tweeted_urls = set(item["url"] for item in tweeted_list if isinstance(item, dict) and "url" in item)

        untweeted_items = [item for item in all_results if item.get("url") and item["url"] not in tweeted_urls]
        if not untweeted_items:
            return json.dumps({"error": "No untweeted items available"})

        selected_item = max(untweeted_items, key=lambda x: x.get("similarity_score", 0))

        tweeted_list.append({"url": selected_item["url"]})
        tweeted_data = {"tweeted": tweeted_list}

        with open(tweeted_file_path, "w", encoding="utf-8") as f:
            json.dump(tweeted_data, f, ensure_ascii=False, indent=2)

        return json.dumps(selected_item, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Error processing JSON files: {str(e)}"})


@tool
def save_to_json(content: Union[str, dict]) -> str:
    """
    Save new result entries into a single 'results' list inside './saved/results.json'.
    
    Args:
        content: A JSON string or Python dictionary with a 'results' key pointing to a list.
        
    Returns:
        str: A message indicating success or failure.
    """
    try:
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                return "Error: Invalid JSON string provided."

        if not isinstance(content, dict) or "results" not in content or not isinstance(content["results"], list):
            return "Error: Content must be a dictionary with a 'results' key containing a list."

        output_dir = Path("./saved")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "results.json"

        merged_data = {"results": []}
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, dict) and "results" in existing_data:
                        merged_data["results"] = existing_data["results"]
            except json.JSONDecodeError:
                return f"Error: Existing file {output_file} contains invalid JSON."
            except Exception as e:
                return f"Error reading existing file: {str(e)}"

        merged_data["results"].extend(content["results"])

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2)

        return f"Successfully merged content into {output_file}"

    except Exception as e:
        return f"Error saving JSON to file: {str(e)}"

    

@tool
def post_to_X(
    consumer_key: str,
    consumer_secret: str,
    access_token: str,
    access_token_secret: str,
    content: str
) -> str:
    """
    Posts a message to X using the Tweepy library.

    This function authenticates with the X API using provided credentials and posts the given content
    as a tweet. It ensures the content is within the 280-character limit, handles authentication errors,
    network issues, and other potential failures, and returns a status message indicating success or failure.

    Args:
        consumer_key (str): The X API consumer key for authentication.
        consumer_secret (str): The X API consumer secret for authentication.
        access_token (str): The X API access token for the user.
        access_token_secret (str): The X API access token secret for the user.
        content (str): The content to post on X (must be 280 characters or fewer).

    Returns:
        str: A message indicating the result of the operation.
             - On success: "Successfully posted to X"
             - On failure: A descriptive error message, e.g., "Error posting to X: <error details>"

    Raises:
        ValueError: If any input parameter is invalid (e.g., empty or non-string).
        tweepy.TweepyException: If there is an issue with the X API (e.g., authentication failure, rate limits).
    """
    if not all(isinstance(arg, str) for arg in [consumer_key, consumer_secret, access_token, access_token_secret, content]):
        return "Error posting to X: All arguments must be strings"
    
    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        return "Error posting to X: Credentials cannot be empty"
    
    if not content:
        return "Error posting to X: Content cannot be empty"
    
    if len(content) > 280:
        content = content[:277] + "..."  
    
    try:
        x_client = tweepy.Client(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret
        )
    except Exception as e:
        return f"Error initializing X client: {str(e)}"
    
    try:
        x_client.create_tweet(text=content)
        return "Successfully posted to X"
    except tweepy.TweepyException as e:
        return f"Error posting to X: {str(e)}"
    except Exception as e:
        return f"Unexpected error posting to X: {str(e)}"
    

@tool
def ArxivTool(query: str, max_results: int = 5) -> str:
    """
    Search ArXiv for papers based on a provided query and return relevant results in JSON format.

    This tool uses the ArxivLoader to fetch papers from ArXiv matching the given query. It is designed for use in an AI agent
    workflow to curate recent publications in a specific field. The tool filters results to include only papers published on the
    current date (July 06, 2025) and returns a JSON string containing paper details. The LLM is responsible for generating a
    precise and relevant query based on the field of interest (e.g., "Point Process machine learning 2025").

    Args:
        query (str): The search query to find relevant ArXiv papers. Should include specific keywords, subfields, or methodologies
                     relevant to the field, and ideally include the current year (2025) to narrow results.
        max_results (int, optional): Maximum number of papers to retrieve. Defaults to 10. Must be a positive integer.

    Returns:
        str: A JSON string containing an array of paper details, each with the following fields:
             - source: "arxiv"
             - title: Paper title
             - authors: List of author names
             - publish_date: Publication date in DD-MM-YYYY format
             - summary: Concise summary (50-100 words) of the paper’s key contributions
             - url: ArXiv URL (e.g., https://arxiv.org/abs/1234.5678)
             If no results are found or an error occurs, returns a JSON string with an empty array and an error message.

    Raises:
        ValueError: If the query is empty or not a string, or if max_results is not a positive integer.
        Exception: For network issues, ArXiv API errors, or other unexpected failures.
    """

    if not isinstance(query, str) or not query.strip():
        return json.dumps({
            "results": [],
            "error": "Query must be a non-empty string"
        })
    
    if not isinstance(max_results, int) or max_results <= 0:
        return json.dumps({
            "results": [],
            "error": "max_results must be a positive integer"
        })
    
    current_date = datetime.now().strftime("%d-%m-%Y")
    
    try:
        loader = ArxivLoader(query=query, load_max_docs=max_results)
        documents = loader.load()
        
        results = []
        for doc in documents:
            metadata = doc.metadata
            publish_date = metadata.get("published", "")
            
            try:
                publish_date_obj = datetime.strptime(publish_date, "%Y-%m-%d")
                formatted_date = publish_date_obj.strftime("%d-%m-%Y")
            except ValueError:
                formatted_date = "Unknown"
            
            # if formatted_date != current_date:
            #     continue
            
            authors = metadata.get("authors", "").split(", ")
            
            summary = doc.page_content[:200]
            summary = doc.page_content  
            
            entry = {
                "source": "arxiv",
                "title": metadata.get("title", "Unknown"),
                "authors": authors if authors else ["Unknown"],
                "publish_date": formatted_date,
                "summary": summary,
                "url" : metadata.get('entry_id', '')
            }
            results.append(entry)
        
        if not results:
            return json.dumps({
                "results": [],
                "error": f"No papers found for query '{query}' on {current_date}"
            })
        
        return json.dumps({"results": results})
    
    except Exception as e:
        return json.dumps({
            "results": [],
            "error": f"Error querying ArXiv: {str(e)}"
        })


@tool
def tavily_tool(query, tavily_api_key, max_results: int = 5) -> str:
    """
    Creates and returns a configured instance of TavilySearch tool.

    Args:
        query (str) : Search string to search in web.
        tavily_api_key (str) : API key to use tavily tool.
        max_results (int): Maximum number of results to return. Default is 5.

    Returns:
        str : search results as string.
    """
    os.environ["TAVILY_API_KEY"] = tavily_api_key 
    
    includer_domains = [
        "https://journals.plos.org/ploscompbiol/issue",
        "https://www.spatialedge.co/",
        "https://events2025.github.io/docs/conferences.html",
        "https://spacetimecausality.github.io/",
    ]
    
    tavily_tool = TavilySearch(
        max_results=max_results,
        api_key=tavily_api_key,
        include_domains=includer_domains,
    )
    return tavily_tool.invoke(query)