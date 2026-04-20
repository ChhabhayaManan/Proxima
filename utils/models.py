import os
from typing import Any, Literal, cast

from dotenv import load_dotenv


load_dotenv()

DEFAULT_MODEL_NAME = "gemini-3.1-flash-lite-preview"
DEFAULT_OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3.5:9b")
DEFAULT_TEMPERATURE = 0.2
DEFAULT_OLLAMA_CONTEXT_WINDOW = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

ModelProvider = Literal["google", "groq", "ollama"]

_MODEL_CONFIG: dict[str, Any] = {
    "api_key": None,
    "model_name": DEFAULT_MODEL_NAME,
    "temperature": DEFAULT_TEMPERATURE,
}

DEFAULT_GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

_GROQ_MODEL_CONFIG: dict[str, Any] = {
    "api_key": None,
    "model_name": DEFAULT_GROQ_MODEL_NAME,
    "temperature": DEFAULT_TEMPERATURE,
}

_OLLAMA_MODEL_CONFIG: dict[str, Any] = {
    "model_name": DEFAULT_OLLAMA_MODEL_NAME,
    "temperature": DEFAULT_TEMPERATURE,
    "num_ctx": DEFAULT_OLLAMA_CONTEXT_WINDOW,
    "base_url": DEFAULT_OLLAMA_BASE_URL,
}

_PROVIDER_LABELS: dict[ModelProvider, str] = {
    "google": "Gemini",
    "groq": "Groq",
    "ollama": "Ollama",
}

_OLLAMA_RUNTIME_HEALTH: dict[tuple[str, int, str], bool] = {}


def _ensure_langchain_compat_globals() -> None:
    """Backfill globals expected by older langchain-core releases.

    This environment currently mixes a newer `langchain` package with an older
    `langchain-core`, and the latter still reads root-module globals such as
    `langchain.verbose`. Newer `langchain` builds no longer define them.
    """

    try:
        import langchain
    except ImportError:
        return

    if not hasattr(langchain, "verbose"):
        langchain.verbose = False
    if not hasattr(langchain, "debug"):
        langchain.debug = False
    if not hasattr(langchain, "llm_cache"):
        langchain.llm_cache = None


def normalize_provider(
    provider: str | None,
    *,
    default: ModelProvider,
) -> ModelProvider:
    resolved_provider = (provider or default).strip().lower()
    if resolved_provider not in _PROVIDER_LABELS:
        supported_providers = ", ".join(sorted(_PROVIDER_LABELS))
        raise ValueError(
            f"Unsupported model provider '{provider}'. Supported providers: {supported_providers}."
        )
    return cast(ModelProvider, resolved_provider)


def get_provider_display_name(provider: str) -> str:
    normalized_provider = normalize_provider(provider, default="google")
    return _PROVIDER_LABELS[normalized_provider]


def _format_ollama_runtime_error(
    *,
    model_name: str,
    num_ctx: int,
    base_url: str,
    error: Exception,
) -> str:
    return (
        f"Ollama is reachable at {base_url}, but the local model `{model_name}` could not be loaded "
        f"with num_ctx={num_ctx}. This is an Ollama runtime/model issue rather than a workflow bug.\n"
        "What to try:\n"
        f"1. Run `ollama run {model_name}` directly and confirm the model can answer outside this workflow.\n"
        "2. Restart the Ollama app/server and try again.\n"
        "3. Free RAM or switch to a smaller local model.\n"
        "4. If you need the workflow immediately, leave the Ollama prompt blank and use the hosted provider path.\n"
        f"Original Ollama error: {error}"
    )


def configure_google_model(
    api_key: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
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
    _MODEL_CONFIG["model_name"] = model_name
    _MODEL_CONFIG["temperature"] = temperature

    os.environ["GEMINI_API_KEY"] = resolved_api_key
    os.environ["GOOGLE_API_KEY"] = resolved_api_key


def get_google_model(
    model_name: str | None = None,
    temperature: float | None = None,
) -> Any:
    if _MODEL_CONFIG["api_key"] is None:
        configure_google_model()

    _ensure_langchain_compat_globals()

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError(
            "Google model support is unavailable because `langchain-google-genai` "
            "cannot be imported with the currently installed LangChain packages. "
            "This environment appears to have incompatible versions of "
            "`langchain-google-genai` and `langchain-core`. "
            "Use the Ollama provider for offline runs, or install compatible Google/LangChain package versions."
        ) from exc

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


def configure_ollama_model(
    model_name: str = DEFAULT_OLLAMA_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_OLLAMA_CONTEXT_WINDOW,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> None:
    resolved_model_name = model_name.strip()
    if not resolved_model_name:
        raise ValueError("An Ollama model name is required, for example `llama3.1` or `gpt-oss:20b`.")

    resolved_base_url = (base_url or "").strip() or DEFAULT_OLLAMA_BASE_URL

    _OLLAMA_MODEL_CONFIG["model_name"] = resolved_model_name
    _OLLAMA_MODEL_CONFIG["temperature"] = temperature
    _OLLAMA_MODEL_CONFIG["num_ctx"] = num_ctx
    _OLLAMA_MODEL_CONFIG["base_url"] = resolved_base_url

    os.environ["OLLAMA_MODEL_NAME"] = resolved_model_name
    os.environ["OLLAMA_BASE_URL"] = resolved_base_url


def verify_ollama_model_runtime(
    model_name: str | None = None,
    *,
    num_ctx: int | None = None,
    base_url: str | None = None,
) -> None:
    resolved_model_name = model_name or _OLLAMA_MODEL_CONFIG["model_name"]
    resolved_num_ctx = num_ctx if num_ctx is not None else _OLLAMA_MODEL_CONFIG["num_ctx"]
    resolved_base_url = base_url or _OLLAMA_MODEL_CONFIG["base_url"]

    cache_key = (resolved_model_name, resolved_num_ctx, resolved_base_url)
    if _OLLAMA_RUNTIME_HEALTH.get(cache_key):
        return

    try:
        from ollama import Client

        client = Client(host=resolved_base_url, timeout=20)
        client.chat(
            model=resolved_model_name,
            messages=[{"role": "user", "content": "Reply with OK"}],
            stream=False,
            think=False,
            options={
                "num_ctx": resolved_num_ctx,
                "num_predict": 1,
                "temperature": 0,
            },
            keep_alive=0,
        )
    except Exception as exc:
        raise RuntimeError(
            _format_ollama_runtime_error(
                model_name=resolved_model_name,
                num_ctx=resolved_num_ctx,
                base_url=resolved_base_url,
                error=exc,
            )
        ) from exc

    _OLLAMA_RUNTIME_HEALTH[cache_key] = True


def get_ollama_model(
    model_name: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
    base_url: str | None = None,
):
    if not _OLLAMA_MODEL_CONFIG["model_name"]:
        configure_ollama_model()

    _ensure_langchain_compat_globals()

    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is required for Ollama models. Install it with `pip install langchain-ollama`."
        ) from exc

    return ChatOllama(
        model=model_name or _OLLAMA_MODEL_CONFIG["model_name"],
        temperature=temperature if temperature is not None else _OLLAMA_MODEL_CONFIG["temperature"],
        num_ctx=num_ctx if num_ctx is not None else _OLLAMA_MODEL_CONFIG["num_ctx"],
        base_url=base_url or _OLLAMA_MODEL_CONFIG["base_url"],
    )


def get_structured_ollama_model(
    output_schema: Any,
    model_name: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
    base_url: str | None = None,
):
    # json_schema uses Ollama's structured-output API instead of relying on prompt-only JSON formatting.
    return get_ollama_model(
        model_name=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        base_url=base_url,
    ).with_structured_output(output_schema, method="json_schema")


def configure_groq_model(
    api_key: str | None = None,
    model_name: str = DEFAULT_GROQ_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
) -> None:
    resolved_api_key = (api_key or "").strip() or os.getenv("GROQ_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "A Groq API key is required. Provide it or set GROQ_API_KEY."
        )

    _GROQ_MODEL_CONFIG["api_key"] = resolved_api_key
    _GROQ_MODEL_CONFIG["model_name"] = model_name
    _GROQ_MODEL_CONFIG["temperature"] = temperature

    os.environ["GROQ_API_KEY"] = resolved_api_key


def get_groq_model(
    model_name: str | None = None,
    temperature: float | None = None,
):
    if _GROQ_MODEL_CONFIG["api_key"] is None:
        configure_groq_model()

    _ensure_langchain_compat_globals()

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


def configure_model_provider(
    provider: str,
    *,
    api_key: str | None = None,
    model_name: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_OLLAMA_CONTEXT_WINDOW,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> None:
    normalized_provider = normalize_provider(provider, default="google")

    if normalized_provider == "google":
        configure_google_model(
            api_key=api_key,
            model_name=model_name or _MODEL_CONFIG["model_name"],
            temperature=temperature,
        )
        return

    if normalized_provider == "groq":
        configure_groq_model(
            api_key=api_key,
            model_name=model_name or _GROQ_MODEL_CONFIG["model_name"],
            temperature=temperature,
        )
        return

    configure_ollama_model(
        model_name=model_name or _OLLAMA_MODEL_CONFIG["model_name"],
        temperature=temperature,
        num_ctx=num_ctx,
        base_url=base_url,
    )


def get_model(
    provider: str,
    *,
    model_name: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
    base_url: str | None = None,
):
    normalized_provider = normalize_provider(provider, default="google")

    if normalized_provider == "google":
        return get_google_model(
            model_name=model_name,
            temperature=temperature,
        )

    if normalized_provider == "groq":
        return get_groq_model(
            model_name=model_name,
            temperature=temperature,
        )

    return get_ollama_model(
        model_name=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        base_url=base_url,
    )


def get_structured_model(
    provider: str,
    output_schema: Any,
    *,
    model_name: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
    base_url: str | None = None,
):
    normalized_provider = normalize_provider(provider, default="google")

    if normalized_provider == "google":
        return get_structured_google_model(
            output_schema,
            model_name=model_name,
            temperature=temperature,
        )

    if normalized_provider == "groq":
        return get_structured_groq_model(
            output_schema,
            model_name=model_name,
            temperature=temperature,
        )

    return get_structured_ollama_model(
        output_schema,
        model_name=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        base_url=base_url,
    )
