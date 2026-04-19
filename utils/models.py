"""
Model utilities for Proxima — uses the native `ollama` Python client directly.
No LangChain wrapper; structured output is handled by passing `format=schema`
to the Ollama chat API, which is supported by all models server-side.
"""
import json
import os
import re
import copy
from typing import Any, Type

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

DEFAULT_OLLAMA_MODEL_NAME = "llama3.1"
DEFAULT_TEMPERATURE = 0.2
OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"

# Mutable global config — updated by configure_ollama_model()
_OLLAMA_CONFIG: dict[str, Any] = {
    "base_url": os.getenv(OLLAMA_BASE_URL_ENV, "http://localhost:11434"),
    "model_name": DEFAULT_OLLAMA_MODEL_NAME,
    "temperature": DEFAULT_TEMPERATURE,
}


def configure_ollama_model(
    model_name: str = DEFAULT_OLLAMA_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    base_url: str | None = None,
) -> None:
    """Update the global Ollama config used by all model helpers."""
    previous_model = _OLLAMA_CONFIG.get("model_name")
    
    _OLLAMA_CONFIG["model_name"] = model_name
    _OLLAMA_CONFIG["temperature"] = temperature
    if base_url:
        _OLLAMA_CONFIG["base_url"] = base_url

    if previous_model and previous_model != model_name:
        try:
            client = _get_client()
            client.generate(model=previous_model, prompt="", keep_alive=0)
        except Exception:
            pass


def _get_client():
    """Return a configured native Ollama client."""
    import ollama
    return ollama.Client(
        host=_OLLAMA_CONFIG["base_url"],
        timeout=120.0,
    )

def _resolve_schema(schema: dict) -> dict:
    """Inline all $defs so Ollama grammar parser doesn't choke on $refs."""
    defs = schema.get("$defs", {})
    schema_str = json.dumps(schema)
    for name, definition in defs.items():
        ref = f'"$ref": "#/$defs/{name}"'
        schema_str = schema_str.replace(ref, json.dumps(definition)[1:-1])
    result = json.loads(schema_str)
    result.pop("$defs", None)
    return result


def _chat(messages: list[dict], format: dict | None = None) -> str:
    """
    Call ollama.chat and return the response text.
    Passes format= only when structured output is needed.
    """
    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": _OLLAMA_CONFIG["model_name"],
        "messages": messages,
        "options": {
            "temperature": _OLLAMA_CONFIG["temperature"],
            "num_ctx": 16384,
            "num_predict": 2048,
            "num_gpu": 99,
        },
    }
    if format is not None:
        kwargs["format"] = _resolve_schema(format)

    response = client.chat(**kwargs)
    return response.message.content


def invoke_plain(prompt: str) -> str:
    """Send a plain text prompt and return the text response."""
    return _chat([{"role": "user", "content": prompt}])


def invoke_structured(prompt: str, schema: Type[BaseModel]) -> BaseModel:
    """
    Send a prompt and parse the response into a Pydantic model.
    Uses Ollama's native `format` parameter — no tool-calling required.
    """
    raw = _chat(
        [{"role": "user", "content": prompt}],
        format=schema.model_json_schema(),
    )
    # Strip markdown fences in case model ignores format=
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        return schema.model_validate_json(clean)
    except Exception:
        # Try extracting JSON object/array from the text
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return schema.model_validate_json(match.group(0))
        raise


# ---------------------------------------------------------------------------
# Compatibility shims — keeps existing agent code working without changes
# ---------------------------------------------------------------------------

def get_provider_display_name(provider: str = "ollama") -> str:
    return "Ollama Offline"


def normalize_model_provider(provider: str | None) -> str:
    return "ollama"


def get_data_generation_provider() -> str:
    return "ollama"


def get_scoring_provider() -> str:
    return "ollama"


def get_ollama_model(model_name: str | None = None, temperature: float | None = None):
    """
    Returns a thin callable wrapper around invoke_plain, mimicking the
    LangChain `.invoke(prompt)` interface so existing agents need minimal changes.
    """
    class _PlainInvoker:
        def invoke(self, prompt: str) -> str:
            return invoke_plain(prompt)

    return _PlainInvoker()


def get_structured_ollama_model(
    output_schema: Type[BaseModel],
    model_name: str | None = None,
    temperature: float | None = None,
):
    """
    Returns a thin callable wrapper around invoke_structured, mimicking the
    LangChain `.invoke(prompt) -> BaseModel` interface.
    """
    class _StructuredInvoker:
        def invoke(self, prompt: str) -> BaseModel:
            return invoke_structured(prompt, output_schema)

    return _StructuredInvoker()


def get_structured_model(
    output_schema: Type[BaseModel],
    provider: str = "ollama",
    model_name: str | None = None,
    temperature: float | None = None,
):
    return get_structured_ollama_model(output_schema, model_name, temperature)


def get_model_for_provider(
    provider: str = "ollama",
    model_name: str | None = None,
    temperature: float | None = None,
):
    return get_ollama_model(model_name, temperature)
