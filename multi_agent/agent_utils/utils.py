from langchain_google_genai import ChatGoogleGenerativeAI


def load_llm(llm, model, temperature=0.0, api_key=None):
    """
    Load the LLM and return an instance of the LLM.
    """
    if not api_key:
        raise ValueError("API key must be provided for the LLM.")
    
    if llm == "google":

        llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=api_key
        )
    else:
        # Add support for other LLMs if needed
        raise ValueError(f"Unsupported LLM type: {llm}. Only 'google' is supported.")

    return llm