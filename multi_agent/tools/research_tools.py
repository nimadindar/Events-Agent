import os
import json
import arxiv
import tweepy
from pathlib import Path
from typing import Union, List, Dict

from langchain.tools import tool
from serpapi import GoogleSearch
from langchain_tavily import TavilySearch

from ..utils.utils import normalize_url


@tool
def save_to_json(content: Union[str, dict], source: str) -> str:
    """
    Save new result entries into a single 'results' list inside a source-specific file:
      - arxiv     -> ./saved/arxiv_results.json
      - blog      -> ./saved/blog_results.json
      - gscholar  -> ./saved/gscholar_results.json

    Args:
        content: A JSON string or Python dict with a 'results' key pointing to a list.
        source: One of {'arxiv', 'blog', 'gscholar'} determining the output file.

    Returns:
        str: A message indicating success or failure.

    Raises:
        ValueError: If 'source' is not one of the allowed names.
    """
    allowed = {"arxiv": "arxiv_results.json",
               "blog": "blog_results.json",
               "gscholar": "gscholar_results.json"}

    if source not in allowed:
        raise ValueError(f"Invalid source '{source}'. Must be one of {sorted(allowed.keys())}.")

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
        output_file = output_dir / allowed[source]

        existing_results = []
        existing_urls = set()

        if output_file.exists():
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, dict) and "results" in existing_data and isinstance(existing_data["results"], list):
                        existing_results = existing_data["results"]

                        existing_urls = {normalize_url(entry.get("url")) for entry in existing_results if isinstance(entry, dict)}
            except json.JSONDecodeError:
                return f"Error: Existing file {output_file} contains invalid JSON."
            except Exception as e:
                return f"Error reading existing file: {str(e)}"

        new_unique_results = []
        for entry in content["results"]:
            if not isinstance(entry, dict):
                continue
            url = entry.get("url")
            if not url:
                continue
            norm = normalize_url(url)  
            if norm not in existing_urls:
                new_unique_results.append(entry)

        if not new_unique_results:
            return f"No new unique results to save for source '{source}'."

        merged_results = existing_results + new_unique_results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"results": merged_results}, f, ensure_ascii=False, indent=2)

        return f"Successfully added {len(new_unique_results)} new unique results to {output_file}"

    except Exception as e:
        return f"Error saving JSON to file: {str(e)}"
    

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