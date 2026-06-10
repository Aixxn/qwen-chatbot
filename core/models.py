import logging

import ollama

logger = logging.getLogger(__name__)

PREFERRED_MODELS = ("qwen3:8b", "qwen3:14b")


def _model_name(model: object) -> str:
    """Extract a model name from Ollama SDK objects or dictionaries."""
    if isinstance(model, dict):
        return str(model.get("model") or model.get("name") or "")
    return str(getattr(model, "model", "") or getattr(model, "name", ""))


def get_list_of_models() -> list[str]:
    """Return locally installed Ollama chat models that support tool calling."""
    models: list[str] = []
    try:
        local_models = ollama.list()["models"]
    except Exception as exc:
        raise RuntimeError("Unable to communicate with the Ollama service") from exc

    for model in local_models:
        name = _model_name(model)
        if not name or name.endswith(":cloud"):
            continue
        try:
            info = ollama.show(name)
            capabilities = getattr(info, "capabilities", None)
            if capabilities is None and isinstance(info, dict):
                capabilities = info.get("capabilities", [])
            if "tools" in (capabilities or []):
                models.append(name)
        except Exception:
            logger.warning("Unable to inspect Ollama model %s", name)

    return sorted(models, key=_model_sort_key)


def _model_sort_key(name: str) -> tuple[int, str]:
    try:
        return (PREFERRED_MODELS.index(name), name)
    except ValueError:
        return (len(PREFERRED_MODELS), name)


def get_default_model(models: list[str]) -> str | None:
    """Choose a default tool-capable model."""
    if not models:
        return None
    for preferred in PREFERRED_MODELS:
        if preferred in models:
            return preferred
    return models[0]


def check_if_model_is_available(model_name: str) -> None:
    """Ensure the model is available locally."""
    if model_name.endswith(":cloud"):
        raise RuntimeError("Cloud Ollama models are not allowed for this chatbot.")
    try:
        ollama.show(model_name)
    except ollama.ResponseError as exc:
        raise RuntimeError(
            f"Model '{model_name}' is not available locally. Pull it with Ollama."
        ) from exc
    except Exception as exc:
        raise RuntimeError("Unable to communicate with the Ollama service") from exc
