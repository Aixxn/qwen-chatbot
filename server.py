import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from core.agent import CustomerServiceAgent, create_customer_service_agent
from core.document_loader import (
    DEFAULT_EMBEDDING_MODEL,
    SUPPORTED_EXTENSIONS,
    rebuild_documents_database,
)
from core.models import (
    check_if_model_is_available,
    get_default_model,
    get_list_of_models,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DOCUMENTS_PATH = "documents"
DEFAULT_MODEL = os.getenv("CHATBOT_MODEL", "").strip()
DOCUMENTS_PATH = os.getenv("CHATBOT_DOCUMENTS_PATH", DEFAULT_DOCUMENTS_PATH).strip()
WATCH_INTERVAL_SECONDS = 3
WATCH_DEBOUNCE_SECONDS = 2


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class IndexRequest(BaseModel):
    model: str
    documents_path: str = DEFAULT_DOCUMENTS_PATH
    reload: bool = True


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatHistoryMessage] = Field(default_factory=list)


class ModelsResponse(BaseModel):
    models: list[str]
    default_model: str | None


class StatusResponse(BaseModel):
    model: str | None
    documents_path: str
    folder_valid: bool
    initialized: bool
    chunk_count: int
    indexing: bool = False
    error: str | None = None


class IndexResponse(StatusResponse):
    message: str


class ChatResponse(BaseModel):
    answer: str
    searches: list[str]


@dataclass
class AppState:
    agent: CustomerServiceAgent | None = None
    model: str | None = None
    documents_path: str = DOCUMENTS_PATH or DEFAULT_DOCUMENTS_PATH
    indexing: bool = False
    error: str | None = None
    fingerprint: tuple[tuple[str, int, int], ...] = ()


state = AppState()
state_lock = threading.RLock()
stop_watcher = threading.Event()
app = FastAPI(title="Customer Service RAG Chatbot")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def start_background_indexer() -> None:
    thread = threading.Thread(target=_watch_documents, daemon=True)
    thread.start()


@app.on_event("shutdown")
def stop_background_indexer() -> None:
    stop_watcher.set()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/api/models")
def api_models() -> ModelsResponse:
    models = get_list_of_models()
    return ModelsResponse(models=models, default_model=get_default_model(models))


@app.get("/api/status")
def api_status() -> StatusResponse:
    return _status_response()


@app.post("/api/index")
def api_index(request: IndexRequest) -> IndexResponse:
    documents_path = request.documents_path.strip() or DEFAULT_DOCUMENTS_PATH
    if not os.path.isdir(documents_path):
        raise HTTPException(
            status_code=400,
            detail=f"The documents folder does not exist: {documents_path}",
        )

    try:
        agent = _build_agent(request.model, documents_path, reload=request.reload)
    except Exception as exc:
        logger.exception("Failed to initialize RAG agent")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    chunk_count = _table_count(agent)
    if chunk_count == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "No supported document chunks were found. Add documents to the "
                "folder and re-index."
            ),
        )

    with state_lock:
        state.agent = agent
        state.model = request.model
        state.documents_path = documents_path
        state.error = None
        state.fingerprint = _documents_fingerprint(documents_path)
    status = _status_response()
    return IndexResponse(**status.model_dump(), message="Documents indexed")


@app.post("/api/chat")
def api_chat(request: ChatRequest) -> ChatResponse:
    agent = _require_agent()
    try:
        answer, searches = agent.ask(
            request.message, message_history=_convert_history(request.history)
        )
    except Exception as exc:
        logger.exception("Failed to answer chat request")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(answer=answer, searches=searches)


@app.post("/api/chat/stream")
def api_chat_stream(request: ChatRequest) -> StreamingResponse:
    agent = _require_agent()

    def events() -> Iterator[str]:
        try:
            for event_type, payload in agent.stream(
                request.message, message_history=_convert_history(request.history)
            ):
                event = {"type": event_type, "payload": payload}
                yield f"data: {json.dumps(event)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'payload': {}})}\n\n"
        except Exception as exc:
            logger.exception("Failed to stream chat request")
            yield f"data: {json.dumps({'type': 'error', 'payload': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _require_agent() -> CustomerServiceAgent:
    with state_lock:
        agent = state.agent
        indexing = state.indexing
        error = state.error
    if agent is None:
        detail = "The support assistant is preparing its knowledge base."
        if error:
            detail = error
        elif indexing:
            detail = "The support assistant is indexing its knowledge base."
        raise HTTPException(
            status_code=400,
            detail=detail,
        )
    return agent


def _status_response() -> StatusResponse:
    with state_lock:
        agent = state.agent
        documents_path = state.documents_path
        return StatusResponse(
            model=state.model,
            documents_path=documents_path,
            folder_valid=os.path.isdir(documents_path),
            initialized=agent is not None,
            chunk_count=_table_count(agent) if agent else 0,
            indexing=state.indexing,
            error=state.error,
        )


def _table_count(agent: CustomerServiceAgent | None) -> int:
    if agent is None:
        return 0
    try:
        return len(agent.vector_store)
    except Exception:
        return 0


def _convert_history(messages: list[ChatHistoryMessage]) -> list[ModelMessage]:
    history: list[ModelMessage] = []
    for message in messages:
        if message.role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=message.content)]))
        else:
            history.append(ModelResponse(parts=[TextPart(content=message.content)]))
    return history


def _watch_documents() -> None:
    _refresh_index_if_needed(force=True)

    while not stop_watcher.wait(WATCH_INTERVAL_SECONDS):
        _refresh_index_if_needed()


def _refresh_index_if_needed(force: bool = False) -> None:
    with state_lock:
        documents_path = state.documents_path
        previous_fingerprint = state.fingerprint
        if state.indexing:
            return

    current_fingerprint = _documents_fingerprint(documents_path)
    if not force and current_fingerprint == previous_fingerprint:
        return

    if not force:
        time.sleep(WATCH_DEBOUNCE_SECONDS)
        current_fingerprint = _documents_fingerprint(documents_path)
        with state_lock:
            if current_fingerprint == state.fingerprint:
                return

    _rebuild_index(documents_path, current_fingerprint)


def _rebuild_index(
    documents_path: str, fingerprint: tuple[tuple[str, int, int], ...]
) -> None:
    with state_lock:
        state.indexing = True
        state.error = None

    try:
        model = _configured_model()
        agent = _build_agent(model, documents_path, reload=True)
        chunk_count = _table_count(agent)
        if chunk_count == 0:
            raise RuntimeError(
                "No supported document chunks were found in the documents folder."
            )

        with state_lock:
            state.agent = agent
            state.model = model
            state.documents_path = documents_path
            state.fingerprint = fingerprint
            state.error = None
        logger.info("Knowledge base indexed with %s chunks", chunk_count)
    except Exception as exc:
        logger.exception("Failed to refresh knowledge base")
        with state_lock:
            state.error = str(exc)
            if not state.fingerprint:
                state.fingerprint = fingerprint
    finally:
        with state_lock:
            state.indexing = False


def _build_agent(
    model: str, documents_path: str, reload: bool
) -> CustomerServiceAgent:
    check_if_model_is_available(model)
    check_if_model_is_available(DEFAULT_EMBEDDING_MODEL)
    if reload:
        vector_store = rebuild_documents_database(
            DEFAULT_EMBEDDING_MODEL, documents_path
        )
        return CustomerServiceAgent(
            llm_model=model,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            vector_store=vector_store,
        )
    return create_customer_service_agent(
        llm_model=model,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        documents_path=documents_path,
        reload=False,
    )


def _configured_model() -> str:
    if DEFAULT_MODEL:
        return DEFAULT_MODEL

    models = get_list_of_models()
    model = get_default_model(models)
    if model is None:
        raise RuntimeError("No local Ollama chat model with tool support was found.")
    return model


def _documents_fingerprint(path: str) -> tuple[tuple[str, int, int], ...]:
    root = Path(path)
    if not root.is_dir():
        return ()

    fingerprint: list[tuple[str, int, int]] = []
    for file_path in sorted(root.rglob("*")):
        if (
            not file_path.is_file()
            or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS
        ):
            continue
        try:
            stat = file_path.stat()
        except OSError:
            continue
        fingerprint.append(
            (str(file_path.relative_to(root)), stat.st_mtime_ns, stat.st_size)
        )
    return tuple(fingerprint)
