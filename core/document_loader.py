import logging
import os
from pathlib import Path
from typing import Any

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from markitdown import MarkItDown

logger = logging.getLogger(__name__)

PERSIST_DIRECTORY = "storage"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSIONS = 768
TABLE_NAME = "documents"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".md",
    ".html",
    ".csv",
    ".json",
}

ollama_embed = get_registry().get("ollama")
_db_connection: lancedb.DBConnection | None = None


def get_db_connection() -> lancedb.DBConnection:
    """Return a singleton LanceDB connection."""
    global _db_connection
    db_path = os.path.join(PERSIST_DIRECTORY, "lancedb")
    if _db_connection is None:
        os.makedirs(db_path, exist_ok=True)
        _db_connection = lancedb.connect(db_path)
    return _db_connection


def get_embedding_function(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    """Create an Ollama embedding function for LanceDB."""
    return ollama_embed.create(name=model_name)


_default_embedding_func = get_embedding_function()


class Document(LanceModel):
    """Document chunk stored in LanceDB."""

    text: str = _default_embedding_func.SourceField()
    source: str
    page: int
    vector: Vector(EMBEDDING_DIMENSIONS) = _default_embedding_func.VectorField()  # type: ignore[valid-type]


def split_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks."""
    clean_text = text.strip()
    if not clean_text:
        return []
    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks: list[str] = []
    start = 0
    while start < len(clean_text):
        end = start + chunk_size
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def load_file(path: Path) -> list[dict[str, Any]]:
    """Load a supported file and return chunk dictionaries."""
    documents: list[dict[str, Any]] = []
    try:
        result = MarkItDown(enable_plugins=False).convert(str(path))
        if not result.text_content:
            return documents

        for chunk in split_text(result.text_content):
            documents.append({"text": chunk, "source": str(path), "page": 1})
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
    return documents


def load_documents(path: str) -> list[dict[str, Any]]:
    """Load all supported documents in a directory."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"The specified path is not a directory: {path}")

    documents: list[dict[str, Any]] = []
    root = Path(path)
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            logger.info("Loading: %s", file_path)
            documents.extend(load_file(file_path))

    logger.info("Loaded %s document chunks", len(documents))
    return documents


def load_documents_into_database(
    model_name: str, documents_path: str, reload: bool = True
) -> lancedb.table.Table:
    """Load documents into LanceDB or return the existing table."""
    db = get_db_connection()

    if reload:
        raw_documents = load_documents(documents_path)
        if TABLE_NAME in db.table_names():
            db.drop_table(TABLE_NAME)

        table = db.create_table(TABLE_NAME, schema=Document)
        if raw_documents:
            table.add(raw_documents)
        return table

    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)

    return db.create_table(TABLE_NAME, schema=Document)


def rebuild_documents_database(
    model_name: str, documents_path: str
) -> lancedb.table.Table:
    """Rebuild LanceDB after documents have been validated."""
    db = get_db_connection()
    raw_documents = load_documents(documents_path)
    if not raw_documents:
        raise ValueError(
            "No supported document chunks were found in the documents folder."
        )

    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)

    table = db.create_table(TABLE_NAME, schema=Document)
    table.add(raw_documents)
    return table
