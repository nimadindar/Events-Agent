import os, re, json, arxiv
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Union, List, Dict, Optional

# from langchain.tools import tool
from langchain_core.tools import tool 
from serpapi import GoogleSearch
from langchain_tavily import TavilySearch

from ..utils.utils import normalize_url



SAVE_DIR = "saved"

class SaveToJSONArgs(BaseModel):
    """Arguments for the save_to_json tool."""
    json_string: str = Field(..., description="A valid JSON string to write to disk.")
    file_name: str = Field(
        ...,
        description=(
            "Base file name (without path). Extension is optional; "
            "the tool will enforce '.json'."
        ),
    )

def _sanitize_filename(name: str) -> Optional[str]:
    """
    Sanitize a file name to avoid path traversal and illegal characters.
    Returns a cleaned base name without extension, or None if invalid.
    """
    if not isinstance(name, str):
        return None
    name = name.strip()
    name = os.path.basename(name)

    if "." in name:
        name = name.rsplit(".", 1)[0]

    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)

    if not name or set(name) == {"."} or len(name) > 255:
        return None

    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if name.upper() in reserved:
        return None

    return name

@tool("save_to_json", args_schema=SaveToJSONArgs)
def save_to_json(json_string: str, file_name: str) -> str:
    """
    Save a JSON string inside 'save/<file_name>.json' if no duplicate exists.

    Behavior:
    - Validates that `json_string` is valid JSON (parses it first).
    - Ensures the 'save' directory exists.
    - Sanitizes `file_name` to avoid path traversal and illegal names.
    - Enforces a '.json' extension.
    - If a file with the same name already exists, returns an error and does NOT overwrite.

    Returns:
      A human-readable status message describing success or the specific error.
    """
    base = _sanitize_filename(file_name)
    if not base:
        return "ERROR: Invalid `file_name`. Provide a simple base name without paths or reserved characters."

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        return f"ERROR: `json_string` is not valid JSON: {e}"

    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
    except OSError as e:
        return f"ERROR: Could not create directory '{SAVE_DIR}': {e}"

    target_path = os.path.join(SAVE_DIR, f"{base}.json")

    if os.path.exists(target_path):
        return f"ERROR: File '{base}.json' already exists in '{SAVE_DIR}'. No file was written."

    try:
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        return f"ERROR: Failed to write file '{target_path}': {e}"

    return f"OK: Saved JSON to '{target_path}'."


@tool
def save_to_json_deprecated(content: Union[str, dict], file_name: str) -> str:
    """
    Save new result entries into a single json file:

    Args:
        content: A JSON string or Python dict.
        filename: name of the file to save.

    Returns:
        str: A message indicating success or failure.
    """
    # if source not in allowed:
    #     raise ValueError(f"Invalid source '{source}'. Must be one of {sorted(allowed.keys())}.")

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
        output_file = output_dir / file_name

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
            return f"No new unique results to save for file name: '{file_name}'."

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
                "error": f"No papers found for query '{query}'"
            })

        return json.dumps({"results": results}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "results": [],
            "error": f"Error querying ArXiv: {str(e)}"
        })


@tool
def tavily_tool(query, tavily_api_key, domains_included, start_date, end_date, max_results = 5) -> str:
    """
    Creates and returns a configured instance of TavilySearch tool.

    Args:
        query (str) : Search string to search in web.
        tavily_api_key (str) : API key to use tavily tool.
        domains_included : Domains used to search for query.
        start_date, end_date : Time frame to search for the blog posts.
        max_results (int): Maximum number of results to return. Default is 5.

    Returns:
        str : search results as string.
    """
    os.environ["TAVILY_API_KEY"] = tavily_api_key 
    
    tavily_tool = TavilySearch(
        max_results=max_results,
        api_key=tavily_api_key,
        search_depth = 'advanced',
        include_domains=domains_included,
        topic = "general",
        start_date = start_date,
        end_date= end_date,
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