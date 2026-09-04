"""Centralized configuration for the tourism agent.

All environment variables and tunable constants live here so the rest
of the codebase never calls os.environ directly.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    opentripmap_api_key: str
    openweather_api_key: str

    # Qwen3.6 Plus Preview, served free (rate-limited) via OpenRouter.
    # Supports tool calling, which the agent relies on.
    llm_model: str = "qwen/qwen3.6-plus-preview:free"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_temperature: float = 0.3

    rag_persist_dir: str = "data/chroma_db"
    rag_collection_name: str = "city_guides"
    rag_top_k: int = 4

    agent_max_iterations: int = 8


def load_settings() -> Settings:
    """Load settings from environment variables, failing fast if required
    keys are missing."""
    required = {
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your keys. "
            "OpenRouter keys are free: https://openrouter.ai/keys"
        )

    return Settings(
        openrouter_api_key=required["OPENROUTER_API_KEY"],
        opentripmap_api_key=os.getenv("OPENTRIPMAP_API_KEY", ""),
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY", ""),
    )


settings = load_settings()
