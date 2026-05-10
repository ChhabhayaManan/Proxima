import os
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite-preview"
DEFAULT_TEMPERATURE = 0.2

_MODEL_CONFIG: dict[str, Any] = {
    "api_key": None,
    "model_name": DEFAULT_MODEL_NAME,
    "temperature": DEFAULT_TEMPERATURE,
}

DEFAULT_GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

_GROQ_MODEL_CONFIG: dict[str, Any] = {
    "api_key": None,
    "model_name": DEFAULT_GROQ_MODEL_NAME,
    "temperature": DEFAULT_TEMPERATURE,
}

DEFAULT_OLLAMA_MODEL_NAME = "llama3.2"

_OLLAMA_MODEL_CONFIG: dict[str, Any] = {
    "base_url": "http://localhost:11434",
    "api_key": None,
    "model_name": DEFAULT_OLLAMA_MODEL_NAME,
    "temperature": DEFAULT_TEMPERATURE,
}


def configure_google_model(
    api_key: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
) -> None:
    resolved_api_key = (
        (api_key or "").strip()
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not resolved_api_key:
        raise ValueError(
            "A Gemini/Google API key is required. Enter it when prompted or set GEMINI_API_KEY / GOOGLE_API_KEY."
        )

    _MODEL_CONFIG["api_key"] = resolved_api_key
    if model_name is not None:
        _MODEL_CONFIG["model_name"] = model_name
    if temperature is not None:
        _MODEL_CONFIG["temperature"] = temperature

    os.environ["GEMINI_API_KEY"] = resolved_api_key
    os.environ["GOOGLE_API_KEY"] = resolved_api_key


def get_google_model(
    model_name: str | None = None,
    temperature: float | None = None,
) -> ChatGoogleGenerativeAI:
    if _MODEL_CONFIG["api_key"] is None:
        configure_google_model()

    return ChatGoogleGenerativeAI(
        model=model_name or _MODEL_CONFIG["model_name"],
        temperature=temperature if temperature is not None else _MODEL_CONFIG["temperature"],
        api_key=_MODEL_CONFIG["api_key"],
    )


def get_structured_google_model(
    output_schema: Any,
    model_name: str | None = None,
    temperature: float | None = None,
):
    return get_google_model(
        model_name=model_name,
        temperature=temperature,
    ).with_structured_output(output_schema)


def configure_groq_model(
    api_key: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
) -> None:
    resolved_api_key = (api_key or "").strip() or os.getenv("GROQ_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "A Groq API key is required. Provide it or set GROQ_API_KEY."
        )

    _GROQ_MODEL_CONFIG["api_key"] = resolved_api_key
    if model_name is not None:
        _GROQ_MODEL_CONFIG["model_name"] = model_name
    if temperature is not None:
        _GROQ_MODEL_CONFIG["temperature"] = temperature

    os.environ["GROQ_API_KEY"] = resolved_api_key


def get_groq_model(
    model_name: str | None = None,
    temperature: float | None = None,
):
    if _GROQ_MODEL_CONFIG["api_key"] is None:
        configure_groq_model()

    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise ImportError(
            "langchain-groq is required for Groq models. Install it with `pip install langchain-groq`."
        ) from exc

    return ChatGroq(
        model=model_name or _GROQ_MODEL_CONFIG["model_name"],
        temperature=temperature if temperature is not None else _GROQ_MODEL_CONFIG["temperature"],
        api_key=_GROQ_MODEL_CONFIG["api_key"],
    )


def get_structured_groq_model(
    output_schema: Any,
    model_name: str | None = None,
    temperature: float | None = None,
):
    return get_groq_model(
        model_name=model_name,
        temperature=temperature,
    ).with_structured_output(output_schema)


def configure_ollama_model(
    base_url: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
) -> None:
    if base_url:
        _OLLAMA_MODEL_CONFIG["base_url"] = base_url.strip()
    if api_key:
        _OLLAMA_MODEL_CONFIG["api_key"] = api_key.strip()
    if model_name is not None:
        _OLLAMA_MODEL_CONFIG["model_name"] = model_name
    if temperature is not None:
        _OLLAMA_MODEL_CONFIG["temperature"] = temperature


def get_ollama_model(
    model_name: str | None = None,
    temperature: float | None = None,
):
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is required for Local models. Install it with `pip install langchain-ollama`."
        ) from exc

    kwargs = {
        "model": model_name or _OLLAMA_MODEL_CONFIG["model_name"],
        "temperature": temperature if temperature is not None else _OLLAMA_MODEL_CONFIG["temperature"],
        "base_url": _OLLAMA_MODEL_CONFIG["base_url"],
    }
    
    if _OLLAMA_MODEL_CONFIG["api_key"]:
        kwargs["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {_OLLAMA_MODEL_CONFIG['api_key']}"}
        }

    return ChatOllama(**kwargs)


def get_provider_model(provider: str, model_name: str | None = None, temperature: float | None = None):
    """Dynamically return the requested LLM from the provider string."""
    if provider.lower() == "groq":
        return get_groq_model(model_name=model_name, temperature=temperature)
    elif provider.lower() in ("google gemini", "gemini", "google"):
        return get_google_model(model_name=model_name, temperature=temperature)
    elif provider.lower() in ("ollama", "ollama (local)"):
        return get_ollama_model(model_name=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_structured_provider_model(provider: str, output_schema: Any, model_name: str | None = None, temperature: float | None = None):
    """Dynamically return a structured output model from the requested provider."""
    model = get_provider_model(provider=provider, model_name=model_name, temperature=temperature)
    return model.with_structured_output(output_schema)

