import os
import json
import tweepy
from datetime import datetime
from typing import List, Dict, Any

from langchain.tools import tool
from langchain_community.document_loaders import ArxivLoader

@tool
def save_json(input_data, save_dir="./saved/results.json") -> str:
    """
    Save input_data to a JSON file, appending to existing data if the file exists.
    
    Args:
        input_data: Data to save (expected to be a JSON-serializable object, e.g., list or dict).
        save_dir: Path to the JSON file (default: "./saved/results.json").
    
    Raises:
        ValueError: If input_data is not valid JSON-serializable.
        OSError: If there are issues with file operations.
    """
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    try:
        json.dumps(input_data)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Input data is not valid JSON: {str(e)}")
    
    if os.path.exists(save_dir):
        try:
            with open(save_dir, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    raise ValueError(f"Existing file {save_dir} contains invalid JSON")
            
            if not isinstance(existing_data, list):
                raise ValueError(f"Existing file {save_dir} does not contain a JSON array")
            
            if isinstance(input_data, dict):
                input_data = [input_data]
            elif not isinstance(input_data, list):
                raise ValueError("Input data must be a dict or list for appending")
            
            existing_data.extend(input_data)
            data_to_save = existing_data
        except OSError as e:
            raise OSError(f"Error reading file {save_dir}: {str(e)}")
    else:

        if isinstance(input_data, dict):
            data_to_save = [input_data]
        elif isinstance(input_data, list):
            data_to_save = input_data
        else:
            raise ValueError("Input data must be a dict or list for saving")
    
    try:
        with open(save_dir, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)
            return "Data saved to json successfully!"
    except OSError as e:
        raise OSError(f"Error writing to file {save_dir}: {str(e)}")
    

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

    Example:
        ```python
        result = post_to_X(
            consumer_key="your_consumer_key",
            consumer_secret="your_consumer_secret",
            access_token="your_access_token",
            access_token_secret="your_access_token_secret",
            content="New breakthrough in AI! Check it out: https://example.com #AI"
        )
        print(result)  # Output: "Successfully posted to X" or "Error posting to X: <error>"
        ```

    Notes:
        - The content is automatically truncated to 280 characters to comply with X's limit.
        - Ensure all credentials are valid and have the necessary permissions to post tweets.
        - This tool is designed for use in an AI agent workflow, such as posting curated research findings.
        - If the X API returns an error (e.g., rate limit exceeded, invalid credentials), the error message
          will include details for debugging.
        - Network issues or missing dependencies (e.g., tweepy) may also cause failures.
        - The function does not support posting media or other advanced tweet features.

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
def ArxivTool(query: str, max_results: int = 10) -> str:
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
            
            if formatted_date != current_date:
                continue
            
            authors = metadata.get("authors", "").split(", ")
            
            # summary = doc.page_content[:500]
            summary = doc.page_content  
            
            
            # Construct result entry
            entry = {
                "source": "arxiv",
                "title": metadata.get("title", "Unknown"),
                "authors": authors if authors else ["Unknown"],
                "publish_date": formatted_date,
                "summary": summary,
                "url": f"https://arxiv.org/abs/{metadata.get('entry_id', '').split('/')[-1]}",
            }
            results.append(entry)
        
        # If no results match the date, return empty array with message
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