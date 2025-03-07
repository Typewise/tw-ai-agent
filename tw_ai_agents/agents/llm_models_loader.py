"""Utility functions for the react agent.

This module provides utility functions for the react agent, such as loading chat models.
"""

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from langchain_community.cache import SQLiteCache

# Load environment variables
load_dotenv()


def load_chat_model(model_name: str, **kwargs: Any) -> BaseChatModel:
    """Load a chat model based on the model name.

    Args:
        model_name: The name of the model to load, in the format "provider/model_name".
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        A chat model instance.

    Raises:
        ValueError: If the model provider is not supported.
    """
    # Split the model name into provider and model
    parts = model_name.split("/")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid model name format: {model_name}. Expected format: 'provider/model_name'"
        )

    provider, model = parts

    # Load the appropriate model based on the provider
    if provider.lower() == "openai":
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Create OpenAI chat model
        model = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 1000),
        )

        # To be changed to redis cache
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))

        return model
    else:
        raise ValueError(f"Unsupported model provider: {provider}")


@lru_cache(maxsize=3)
def get_llm_model(model_name: str = "openai/gpt-4o") -> BaseChatModel:
    return load_chat_model(model_name)
