from __future__ import annotations
import tweepy
import json
import re
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any, Tuple
from datetime import datetime, date as dt_date
from pydantic import BaseModel, Field, validator

# try:
#     from langchain_core.tools import tool
# except Exception:
#     try:
#         from langchain.tools import tool  
#     except Exception:
#         def tool(*a, **k):
#             def _wrap(fn): return fn
#             return _wrap

SAVE_DIR = Path("./saved")
TWEETS_FILE = SAVE_DIR / "tweets.json"
SOURCES: Tuple[str, ...] = ("arxiv", "blog", "gscholar")


def _parse_input_date(d: Optional[str]) -> Optional[dt_date]:
    if not d:
        return None

    return datetime.strptime(d.strip(), "%d-%m-%Y").date()


def _parse_publish_date(d: Any) -> Optional[dt_date]:
    if not d or (isinstance(d, str) and d.strip().lower() == "unknown"):
        return None
    s = str(d).strip()
    fmts = ["%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y", "%d.%m.%Y", "%Y.%m.%d"]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date()
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        return None


def _load_json(path: Path) -> Any:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


_ARXIV_ABS_RE = re.compile(r"https?://(?:www\.)?arxiv\.org/abs/([^?#]+)", re.IGNORECASE)
_ARXIV_VERSION_RE = re.compile(r"^(?P<id>\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    u = url.strip().lower()
    u = u.replace("https://", "http://", 1)

    m = _ARXIV_ABS_RE.match(u)
    if m:
        tail = m.group(1) 
        m2 = _ARXIV_VERSION_RE.match(tail)
        if m2:
            base_id = m2.group("id")  
            return f"arxiv:{base_id}"
        return f"arxiv:{tail}"
    return u


def _load_tweets() -> List[Dict[str, str]]:
    """
    Always return a normalized list of dicts: [{"url": ..., "posting_reason": ...}, ...]
    Accept legacy formats:
      - {"tweets": ["url1", "url2", ...]}
      - ["url1", "url2", ...]
      - {"tweets": [{"url": "...", "posting_reason": "..."} , ...]}
    Missing posting_reason becomes "" (empty string).
    """
    raw = _load_json(TWEETS_FILE)
    normalized: List[Dict[str, str]] = []

    def _as_entry(u: Any) -> Optional[Dict[str, str]]:
        if isinstance(u, str):
            return {"url": u, "posting_reason": ""}
        if isinstance(u, dict) and "url" in u:
            pr = u.get("posting_reason", "")
            return {"url": str(u["url"]), "posting_reason": str(pr) if pr is not None else ""}
        return None

    if isinstance(raw, dict) and isinstance(raw.get("tweets"), list):
        for u in raw["tweets"]:
            e = _as_entry(u)
            if e: normalized.append(e)
    elif isinstance(raw, list):
        for u in raw:
            e = _as_entry(u)
            if e: normalized.append(e)

    return normalized

def _save_tweets(entries: List[Dict[str, str]]) -> None:
    _save_json(TWEETS_FILE, {"tweets": entries})


def _tweet_norm_set(entries: List[Dict[str, str]]) -> set[str]:
    return {_normalize_url(e.get("url", "")) for e in entries if isinstance(e, dict)}


def _load_results_for_source(source: str) -> List[Dict[str, Any]]:
    data = _load_json(SAVE_DIR / f"{source}_results.json")
    results = data.get("results", []) if isinstance(data, dict) else []
    return [r for r in results if isinstance(r, dict)]


def _safe_int(x, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _pick_best_by_date(items):
    """Most recent publish_date (tie-break: higher usefulness_score)."""
    def key(it: Dict[str, Any]):
        pd = _parse_publish_date(it.get("publish_date"))
        sc = _safe_int(it.get("usefulness_score"))
        return (pd is None, pd or dt_date.min, sc)
    return max(items, key=key, default=None)


def _pick_best_by_score(items):
    """Highest usefulness_score (tie-break: newer publish_date)."""
    def key(it: Dict[str, Any]):
        sc = _safe_int(it.get("usefulness_score"))
        pd = _parse_publish_date(it.get("publish_date"))
        return (sc, pd is None, pd or dt_date.min)
    return max(items, key=key, default=None)


# class FetchFilteredItemsArgs(BaseModel):
class FetchFilteredItemsArgs():
    source: Literal["arxiv", "blog", "gscholar", "all"] = Field(
        ..., description='Which source(s) to read: "arxiv", "blog", "gscholar", or "all".'
    )
    min_usefulness_score: Optional[int] = Field(
        None, ge=0, le=100, description="Keep items with usefulness_score >= this value."
    )
    date: Optional[str] = Field(
        None, description='Keep items with publish_date >= this date. Format: "dd-mm-yyyy".'
    )

    @validator("date")
    def _validate_date(cls, v):
        if v:
            _ = _parse_input_date(v)  
        return v


# @tool("fetch_filtered_items", args_schema=FetchFilteredItemsArgs)
def fetch_filtered_items(
    source: Literal["arxiv", "blog", "gscholar", "all"],
    min_usefulness_score: Optional[int] = None,
    date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch items from ./saved/{source}_results.json (or all), filter by usefulness score and/or date,
    and EXCLUDE items whose URLs already appear in ./saved/tweets.json (duplicate prevention).
    """

    srcs = SOURCES if source == "all" else (source,)
    raw_items: List[Dict[str, Any]] = []
    sources_used: List[str] = []

    for s in srcs:
        items = _load_results_for_source(s)
        if items:
            raw_items.extend(items)
            sources_used.append(s)

    if source != "all" and not sources_used:
        return {
            "results": [],
            "meta": {
                "error": f"No data found for source '{source}'. Expected './saved/{source}_results.json'.",
                "sources_used": [],
                "min_usefulness_score": min_usefulness_score,
                "date_filter": date,
                "returned": 0,
            },
        }

    min_score = min_usefulness_score if min_usefulness_score is not None else None
    cutoff = _parse_input_date(date) if date else None

    tweeted_entries = _load_tweets()
    tweeted_norm = _tweet_norm_set(tweeted_entries)

    filtered: List[Dict[str, Any]] = []
    excluded_already_tweeted = 0

    for it in raw_items:
        if min_score is not None:
            try:
                if int(it.get("usefulness_score", -1)) < min_score:
                    continue
            except Exception:
                continue

        if cutoff:
            pub = _parse_publish_date(it.get("publish_date"))
            if not pub or pub < cutoff:
                continue

        norm_url = _normalize_url(str(it.get("url", "")))
        if norm_url and norm_url in tweeted_norm:
            excluded_already_tweeted += 1
            continue

        filtered.append(it)

    def _sort_key(x: Dict[str, Any]):
        try:
            sc = int(x.get("usefulness_score", -1))
        except Exception:
            sc = -1
        pd = _parse_publish_date(x.get("publish_date"))
        return (-sc, pd is None, pd or date.min)

    filtered.sort(key=_sort_key)

    return {
        "results": filtered,
        "meta": {
            "sources_used": sources_used,
            "min_usefulness_score": min_score,
            "date_filter": date,
            "already_tweeted_excluded": excluded_already_tweeted,
            "returned": len(filtered),
        },
    }


# class SaveTweetArgs(BaseModel):
class SaveTweetArgs():
    url: str = Field(..., description="The URL of the posted item.")
    posting_reason: str = Field(
        ..., description="Short reason for posting (e.g., why it was selected)."
    )


# @tool("save_tweet", args_schema=SaveTweetArgs)
def save_tweet(url: str, posting_reason: str) -> Dict[str, Any]:
    """
    Record a tweeted item in ./saved/tweets.json with fields:
      { "tweets": [ { "url": <url>, "posting_reason": <str> }, ... ] }
    Prevents duplicates using normalized URLs (handles arXiv URL variants).
    """
    entries = _load_tweets() 
    norm_existing = _tweet_norm_set(entries)
    norm_new = _normalize_url(url)

    if norm_new in norm_existing:
        existing = next((e for e in entries if _normalize_url(e.get("url", "")) == norm_new), None)
        return {"status": "duplicate", "entry": existing or {"url": url, "posting_reason": ""}}

    new_entry = {"url": url, "posting_reason": posting_reason}
    entries.append(new_entry)
    _save_tweets(entries)
    return {"status": "saved", "entry": new_entry}


# @tool
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