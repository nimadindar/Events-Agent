import os, sys, logging
from dotenv import load_dotenv
load_dotenv()

from ..tools import posting_tools
from langchain_google_genai import ChatGoogleGenerativeAI


# Posting Config
SOURCE = "arxiv"
DATE = "01-01-2025"
X_MIN_USEFULNESS = 80
MODE = "score"

# Model Config (LLM for crafting tweet text)
MODEL_NAME = "gemini-2.5-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def get_result(
    source=SOURCE, 
    min_usefulness_score=X_MIN_USEFULNESS, 
    date=DATE,
    mode=MODE):
    """
    Return a single best entry from fetch_filtered_items.

    Args:
        source: which source to use
        min_usefulness_score: threshold for filtering
        date: cutoff for filtering
        mode: how to pick the best item:
              - "date": most recent publish_date
              - "score": highest usefulness_score

    Falls back automatically if dates/scores are missing.
    """

    payload = posting_tools.fetch_filtered_items(source, min_usefulness_score, date)
    items = payload.get("results", [])
    meta  = payload.get("meta", {})

    if not items:
        return {"result": None,
                "meta": {**meta, "selection_mode": "none", "error": "No results."}}

    if mode == "date":
        best = posting_tools._pick_best_by_date(items)
        selection = "date"
        if not best or posting_tools._parse_publish_date(best.get("publish_date")) is None:
            best = posting_tools._pick_best_by_score(items)
            selection = "score_fallback"
    else:  # mode == "score"
        best = posting_tools._pick_best_by_score(items)
        selection = "score"
        if not best or posting_tools._safe_int(best.get("usefulness_score")) < 0:
            best = posting_tools._pick_best_by_date(items)
            selection = "date_fallback"

    return {"result": best,
            "meta": {**meta, "selection_mode": selection}}


def craft_tweet_text(entry_data):
    entry = entry_data.get("result") if isinstance(entry_data, dict) else entry_data

    if not entry:
        return None  # nothing to tweet

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )

    prompt = f"""
    - Write a tweet that:
         * Briefly summarizes the methodology (1–2 short sentences or a crisp clause).
         * Includes the full URL (e.g., "https://...").
         * Includes exactly ONE relevant hashtag (derived from the entry’s topic), placed at the end.
         * Is strictly UNDER 280 characters including the URL.
    - Style constraints: no emojis, no code blocks, no quotes, no extra hashtags, no @mentions, no line breaks, no trailing spaces.
    - If the text exceeds 280 characters, shorten by compressing wording while preserving the methodology, URL, and single hashtag.
       Example tweet text:
        "Vaswani et al. (2017) “Attention Is All You Need”: introduces the Transformer—replaces recurrence with multi-head self-attention + positional encodings for fast, parallel sequence modeling; sets SOTA in MT. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) #MachineLearning"

    - Write a tweet text for the following entry:
    {entry} 
    """

    response = llm.invoke(prompt)
    tweet_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    return tweet_text


def main():
    try:
        entry_data = get_result()  
        tweet_text = craft_tweet_text(entry_data)

        if not tweet_text:
            logging.info("No results to be tweeted.")
            return 0
        try:
            consumer_key = os.getenv("X_API_KEY")
            consumer_secret = os.getenv("X_API_KEY_SECRET")
            access_token = os.getenv("X_ACCESS_TOKEN")
            access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
        except RuntimeError as e:
            logging.error(str(e))
            return 1

        status = posting_tools.post_to_X(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            content=tweet_text,
        )

        if status.startswith("Successfully posted"):
            entry = entry_data.get("result") if isinstance(entry_data, dict) else entry_data
            url = (entry or {}).get("url")

            source = globals().get("SOURCE", "N/A")
            min_score = globals().get("X_MIN_USEFULNESS", "N/A")
            cutoff_date = globals().get("DATE", "N/A")
            mode = globals().get("MODE", "N/A")

            posting_reason = (
                f"The entry was posted with source: {source}, "
                f"min usefulness score: {min_score}, date: {cutoff_date}. "
                f"Selection mode: {mode}."
            )
            if url:
                try:
                    posting_tools.save_tweet(url=url, posting_reason=posting_reason)
                    logging.info("Entry posted successfully and URL saved.")
                except Exception as e:
                    logging.warning(f"Entry posted, but failed to save URL: {e}")
            else:
                logging.warning("Entry posted, but no URL found to save.")

            return 0
        else:
            logging.error(f"Posting failed: {status}")
            return 1

    except Exception as e:
        logging.exception(f"Unhandled error in main: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())