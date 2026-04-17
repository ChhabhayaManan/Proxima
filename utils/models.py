import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

DEFAULT_OLLAMA_MODEL_NAME = "llama3"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_DATA_GENERATION_PROVIDER = "ollama"
DATA_GENERATION_PROVIDER_ENV = "PROXIMA_DATA_GENERATION_PROVIDER"
DEFAULT_SCORING_PROVIDER = "ollama"
SCORING_PROVIDER_ENV = "PROXIMA_SCORING_PROVIDER"

_OLLAMA_MODEL_CONFIG: dict[str, Any] = {
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "model_name": DEFAULT_OLLAMA_MODEL_NAME,
    "temperature": DEFAULT_TEMPERATURE,
}

def normalize_model_provider(provider: str | None) -> str:
    normalized_provider = (provider or DEFAULT_DATA_GENERATION_PROVIDER).strip().lower()
    if normalized_provider != "ollama":
        print(f"Warning: Only Ollama is supported. Forcing provider 'ollama' instead of '{normalized_provider}'.")
        return "ollama"
    return normalized_provider


def set_data_generation_provider(provider: str | None) -> str:
    normalized_provider = normalize_model_provider(provider)
    os.environ[DATA_GENERATION_PROVIDER_ENV] = normalized_provider
    return normalized_provider


def get_data_generation_provider() -> str:
    return normalize_model_provider(os.getenv(DATA_GENERATION_PROVIDER_ENV))


def set_scoring_provider(provider: str | None) -> str:
    normalized_provider = normalize_model_provider(provider)
    os.environ[SCORING_PROVIDER_ENV] = normalized_provider
    return normalized_provider


def get_scoring_provider() -> str:
    return normalize_model_provider(os.getenv(SCORING_PROVIDER_ENV) or DEFAULT_SCORING_PROVIDER)


def get_provider_display_name(provider: str) -> str:
    return "Ollama Offline"


def configure_ollama_model(
    base_url: str | None = None,
    model_name: str = DEFAULT_OLLAMA_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
) -> None:
    _OLLAMA_MODEL_CONFIG["base_url"] = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    _OLLAMA_MODEL_CONFIG["model_name"] = model_name
    _OLLAMA_MODEL_CONFIG["temperature"] = temperature


def get_ollama_model(
    model_name: str | None = None,
    temperature: float | None = None,
):
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is required for Ollama models. Install it with `pip install langchain-ollama`."
        ) from exc

    return ChatOllama(
        base_url=_OLLAMA_MODEL_CONFIG["base_url"],
        model=model_name or _OLLAMA_MODEL_CONFIG["model_name"],
        temperature=temperature if temperature is not None else _OLLAMA_MODEL_CONFIG["temperature"],
    )


def get_structured_ollama_model(
    output_schema: Any,
    model_name: str | None = None,
    temperature: float | None = None,
):
    return get_ollama_model(
        model_name=model_name,
        temperature=temperature,
    ).with_structured_output(output_schema)


def get_structured_model(
    output_schema: Any,
    provider: str,
    model_name: str | None = None,
    temperature: float | None = None,
):
    # Only Ollama is used
    return get_structured_ollama_model(
        output_schema,
        model_name=model_name,
        temperature=temperature,
    )


def get_model_for_provider(
    provider: str,
    model_name: str | None = None,
    temperature: float | None = None,
):
    # Only Ollama is used
    return get_ollama_model(
        model_name=model_name,
        temperature=temperature,
    )
