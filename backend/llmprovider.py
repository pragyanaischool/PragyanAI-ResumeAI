import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

def get_llm(provider: str = "groq", model_name: str = None):
    """
    Initializes and returns an LLM instance based on the specified provider.
    
    Args:
        provider (str): The name of the LLM provider. Supported providers are "groq" and "openai".
        model_name (str): The name of the specific model to use. Defaults to a sensible choice for each provider.
    
    Returns:
        The initialized LLM instance.
    
    Raises:
        ValueError: If the required API key for the provider is not set, or if the provider is unsupported.
    """
    if provider == "groq":
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        model = model_name if model_name else "llama3-8b-8192"
        return ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=model)
    
    elif provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        model = model_name if model_name else "gpt-3.5-turbo"
        return ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model=model)
        
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Please choose from 'groq' or 'openai'.")
