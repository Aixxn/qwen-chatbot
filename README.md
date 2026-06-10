# Customer Service Chatbot

An intelligent customer-service chatbot with retrieval-augmented generation
for online retail businesses. The app runs locally with Ollama, indexes store
support documents into LanceDB, and presents a customer-facing chat interface
for questions about returns, shipping, warranties, product troubleshooting, and
store policies.

## Features

- Customer-facing chat UI with an assistant greeting and quick-start prompts.
- Local-only RAG pipeline using Ollama, LanceDB, MarkItDown, and Pydantic AI.
- Automatic knowledge-base indexing on server startup.
- Background folder watching for supported document changes.
- Safe index rebuilds that keep the previous index available until a new one
  is validated.
- Server-side model and document-folder configuration.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally
- A local Ollama chat model that supports tool calling, such as `qwen3:8b`
- The Ollama embedding model `nomic-embed-text`

## Setup

```bash
uv sync
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

For faster responses on lower-VRAM GPUs, you can also try a smaller
tool-capable model:

```bash
ollama pull qwen3:4b
```

## Documents

Add support documents to the `documents/` folder. The app indexes this folder
automatically when the server starts and refreshes the index when supported
files are added, edited, or deleted.

Supported formats:

- PDF
- Word: `.docx`
- PowerPoint: `.pptx`
- Excel: `.xlsx`
- Markdown: `.md`
- HTML
- CSV
- JSON

## Run

```bash
uv run uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The customer chat UI appears immediately. The chat input is enabled after the
knowledge base has been indexed.

## Configuration

The app hides setup controls from customers. Configure internal settings with
environment variables before starting the server.

```bash
CHATBOT_MODEL=qwen3:8b \
CHATBOT_DOCUMENTS_PATH=documents \
uv run uvicorn server:app --host 127.0.0.1 --port 8000
```

Settings:

- `CHATBOT_MODEL`: optional. If unset, the app chooses the first local
  tool-capable Ollama model, preferring configured defaults in code.
- `CHATBOT_DOCUMENTS_PATH`: optional. Defaults to `documents`.

## API

The customer UI uses these endpoints:

- `GET /api/status`: returns indexing and readiness state.
- `POST /api/chat`: sends a customer message and returns the answer plus search
  queries used by the RAG tool.
- `POST /api/chat/stream`: streams search and text events. This endpoint exists
  for future UI latency improvements.

Legacy internal endpoints still exist:

- `GET /api/models`
- `POST /api/index`

## Performance Notes

Response speed depends mostly on the local chat model and whether Ollama can use
GPU acceleration. On a 6GB GPU such as a GTX 1660 Super, smaller quantized
models like `qwen3:4b` may respond faster than `qwen3:8b`.

Useful checks:

```bash
nvidia-smi
```

Run `nvidia-smi` while the model is answering to confirm whether Ollama is using
the GPU.

Recommended latency improvements:

- Use `/api/chat/stream` in the UI for faster perceived response time.
- Reduce retrieved chunks in `core/agent.py` from 10 to 4 or 5.
- Keep answers concise in the system prompt.
- Use a smaller tool-capable model if quality remains acceptable.

## Project Structure

```text
customer-service-chatbot/
  server.py                  FastAPI app, startup indexing, folder watcher
  core/
    agent.py                 Pydantic AI RAG agent and document search tool
    document_loader.py       Document parsing, chunking, LanceDB indexing
    models.py                Ollama model discovery and validation
  documents/                 Store support documents to index
  static/
    index.html               Customer chat page
    app.js                   Chat UI behavior and status polling
    styles.css               Customer-facing chat design
  storage/lancedb/           Local vector database
```
