import os
import json
import arxiv
import tweepy
from pathlib import Path
from typing import Union, List, Dict

from langchain.tools import tool
from serpapi import GoogleSearch
from langchain_tavily import TavilySearch

from utils.utils import normalize_url

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
                # if isinstance(tweeted_data, dict):
                tweeted_list = tweeted_data.get("tweeted", [])
                # elif isinstance(tweeted_data, list):
                #     tweeted_list = tweeted_data  
                tweeted_urls = set(normalize_url(item["url"]) for item in tweeted_list if isinstance(item, dict) and "url" in item)

        untweeted_items = [item for item in all_results if normalize_url(item["url"]) not in tweeted_urls]

        if not untweeted_items:
            return json.dumps({"error": "No untweeted items available"})

        selected_item = max(untweeted_items, key=lambda x: x.get("usefulness_score", 0))    # criteria to select the tweet

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

        existing_results = []
        existing_urls = set()

        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, dict) and "results" in existing_data:
                        existing_results = existing_data["results"]
                        existing_urls = {normalize_url(entry.get("url")) for entry in existing_results}
            except json.JSONDecodeError:
                return f"Error: Existing file {output_file} contains invalid JSON."
            except Exception as e:
                return f"Error reading existing file: {str(e)}"

        new_unique_results = [
            entry for entry in content["results"]
            if normalize_url(entry.get("url")) not in existing_urls
        ]

        if not new_unique_results:
            return "No new unique results to save."
        
        merged_results = existing_results + new_unique_results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"results": merged_results}, f, indent=2)

        return f"Successfully added {len(new_unique_results)} new unique results to {output_file}"
  
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
        # content = content[:277] + "..."
        return "Error posting to X: Content cannot be more than 280 characters. Try with shorter Content." 
    
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

    This tool uses the arxiv client to fetch papers from ArXiv matching the given query sorted by date in descending order. It is designed for use in an AI agent
    workflow to curate recent publications in a specific field. The LLM is responsible for generating a precise and relevant query based on the field of interest. 

    Args:
        query (str): Carefully crafted query string to search for papers.
        max_results (int, optional): Maximum number of papers to retrieve. 

    Returns:
        str: A JSON string containing an array of paper details, each with the following fields:
             - source: "arxiv"
             - title: Paper title
             - authors: List of author names
             - publish_date: Publication date in DD-MM-YYYY format
             - summary: Concise summary (50-100 words) of the paper’s key contributions
             - url: ArXiv URL (e.g., https://arxiv.org/abs/1234.5678)
             If no results are found or an error occurs, returns a JSON string with an empty array and an error message.
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

    current_year = "2025"  # Filter for 2025 papers

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        results = []
        for result in client.results(search):
            publish_date = result.published
            formatted_date = publish_date.strftime("%d-%m-%Y")

            if not publish_date.strftime("%Y") == current_year:
                continue

            authors = [author.name for author in result.authors]
            summary = result.summary[:500] 

            entry = {
                "source": "arxiv",
                "title": result.title,
                "authors": authors if authors else ["Unknown"],
                "publish_date": formatted_date,
                "summary": summary,
                "url": result.entry_id
            }
            results.append(entry)

        if not results:
            return json.dumps({
                "results": [],
                "error": f"No papers found for query '{query}' in {current_year}"
            })

        return json.dumps({"results": results}, ensure_ascii=False)

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
    
    include_domains = [
    "journals.plos.org",
    "spatialedge.co",
    "events2025.github.io",
    "spacetimecausality.github.io"
]
    
    tavily_tool = TavilySearch(
        max_results=max_results,
        api_key=tavily_api_key,
        include_domains=include_domains,
    )
    return tavily_tool.invoke(query)


@tool
def get_scholar_papers(author_ids: List[str], scholar_max_results: int, api_key: str) -> Union[List[Dict], Dict]:
    """
    Fetches recent papers for a list of Google Scholar author IDs using SerpAPI.
    Returns a combined list of dicts with title, link, authors, publish_date, url, and abstract.
    If an error occurs for a specific author, includes it in the results.

    Args:
        author_ids (List[str]): List of Google Scholar author IDs
        api_key (str): SerpAPI key
        max_scholar_results (int): Max results per author

    Returns:
        Union[List[Dict], Dict]: List of papers or error message
    """
    ABSTRACT_ENABLED = True

    if not api_key:
        return {"error": "Missing SERP_API key."}

    all_papers = []

    for author_id in author_ids:
        params = {
            "engine": "google_scholar_author",
            "author_id": author_id,
            "api_key": api_key,
            "num": scholar_max_results,
            "sort": "pubdate"
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            for article in results.get("articles", []):

                citation_id = article.get("citation_id")

                params_citation = {
                    "engine": "google_scholar_author",
                    "view_op": "view_citation",
                    "citation_id": citation_id,
                    "api_key": api_key }     

                search_paper = GoogleSearch(params_citation)
                citation_data = search_paper.get_dict()
                abstract = citation_data.get("citation")["description"]
                publish_date = citation_data.get("citation")["publication_date"]
                url = citation_data.get("citation")["link"]


                paper = {
                    "source": "Gscholar",
                    "title": article.get("title"),
                    "authors": article.get("authors"),
                    "Publish_date": publish_date,
                    "url": url,                    
                    "abstract": abstract,
                }
                all_papers.append(paper)

        except Exception as e:
            all_papers.append({
                "source": "Gscholar",
                "author_id": author_id,
                "error": f"Failed to fetch data: {str(e)}"
            })

    return all_papers