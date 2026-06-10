# Local Ollama RAG Chatbot Design

## Goal

Create a customer-service chatbot with retrieval-augmented generation using only local Ollama models. The implementation will use the neighboring `local-LLM-with-RAG` project as the technical reference, but replace its Streamlit interface with a vanilla HTML, CSS, and JavaScript web interface.

## Reference Summary

The reference project uses this flow:

1. Load documents from a folder with MarkItDown.
2. Split document text into overlapping chunks.
3. Store chunks and Ollama-generated embeddings in LanceDB.
4. Use Pydantic AI with an Ollama chat model.
5. Register a `search_documents` tool so the model can decide when to retrieve context.
6. Stream responses and show tool search activity in Streamlit.

The new project will preserve the RAG backend pattern and replace Streamlit with a FastAPI server plus static frontend files.

## Recommended Approach

Use a FastAPI backend and a vanilla static frontend.

FastAPI fits the Python and Pydantic stack already used by the reference project, supports typed request/response models, and can serve both JSON endpoints and static files. The browser interface stays framework-free while the backend owns Ollama, MarkItDown, LanceDB, and Pydantic AI integration.

## Architecture

The project will be structured as:

```text
customer-service-chatbot/
  core/
    __init__.py
    agent.py
    document_loader.py
    models.py
  static/
    index.html
    styles.css
    app.js
  documents/
  storage/
  server.py
  pyproject.toml
  README.md
```

### Core Modules

`core/document_loader.py`

- Load supported document formats from a configured folder.
- Convert files to markdown-like text with MarkItDown.
- Split text into overlapping chunks.
- Store chunks in LanceDB under `storage/lancedb`.
- Use Ollama `nomic-embed-text` embeddings through LanceDB's Ollama embedding registry.

Supported formats:

- PDF
- Word
- PowerPoint
- Excel
- Markdown
- HTML
- CSV
- JSON

`core/models.py`

- Query local Ollama models.
- Filter chat models to those that advertise tool-calling capability.
- Check whether the selected chat model and embedding model are available locally.
- Do not use cloud model providers or external LLM APIs.

`core/agent.py`

- Create a Pydantic AI agent backed by Ollama.
- Use a customer-service oriented system prompt.
- Register a `search_documents` tool that embeds the user query, searches LanceDB, and returns top matching chunks with source metadata.
- Maintain message history for follow-up questions.
- Stream answer text and expose search-tool events to the frontend when practical.

## Backend

`server.py` will expose a FastAPI app.

### Endpoints

`GET /`

- Serve the vanilla HTML chat interface.

`GET /api/models`

- Return local Ollama chat models that support tool calling.
- Prefer `qwen3:8b` or `qwen3:14b` in the UI when available.

`GET /api/status`

- Return current backend state:
  - selected model
  - documents folder
  - whether the folder is valid
  - whether the agent is initialized
  - indexed chunk count

`POST /api/index`

- Request body:

```json
{
  "model": "qwen3:8b",
  "documents_path": "documents",
  "reload": true
}
```

- Validate that the folder exists.
- Validate local Ollama connectivity.
- Ensure the selected chat model and `nomic-embed-text` are available locally.
- Load or rebuild the LanceDB index.
- Initialize or replace the active agent.

`POST /api/chat`

- Request body:

```json
{
  "message": "What is your refund policy?",
  "history": [
    {"role": "user", "content": "Earlier question"},
    {"role": "assistant", "content": "Earlier answer"}
  ]
}
```

- Use the active agent to answer the question.
- Return:

```json
{
  "answer": "Answer text",
  "searches": ["search query 1", "search query 2"]
}
```

`POST /api/chat/stream`

- Optional streaming endpoint using server-sent events.
- Stream search events and text deltas.
- If streaming proves too fragile for the first version, `POST /api/chat` is sufficient and the UI can show a pending state.

## Frontend

The frontend will use only HTML, CSS, and JavaScript.

### Layout

- Left settings panel:
  - model selector
  - documents folder input
  - index/re-index button
  - new chat button
  - status and chunk count
- Main chat area:
  - message transcript
  - assistant search status
  - input composer
  - disabled state until indexing succeeds

### Behavior

- Load model list and status on page startup.
- Let the user choose a local model and documents folder.
- Initialize the agent by indexing documents.
- Keep chat history in browser memory for the current session.
- Reset chat without clearing the indexed database.
- Show clear errors from backend responses.
- Avoid marketing-style content; the first screen is the usable chatbot interface.

## Prompt Behavior

The assistant should behave as a customer-service document assistant:

- Answer from the indexed documents when possible.
- Search documents for policy, product, workflow, pricing, return, troubleshooting, or support questions.
- Use complete search queries rather than pronouns.
- Run multiple focused searches for compound questions.
- If documents do not contain enough information, say that the answer is not available from the current documents.
- Do not claim access to external systems, live account data, order databases, or private customer information.

## Error Handling

The backend should return clear errors for:

- Ollama is not running.
- No local tool-capable chat model is available.
- The selected model is unavailable.
- `nomic-embed-text` is unavailable or cannot be pulled.
- The documents folder does not exist.
- No supported documents are found.
- Indexing fails for one or more files.
- The user asks a question before the agent is initialized.

File-specific document loading failures may be logged while allowing other valid files to index.

## Testing And Verification

Implementation should include at least:

- Python import/startup check.
- Document loader smoke check when feasible.
- FastAPI route smoke check.
- Manual browser verification with the local server running.

The final implementation report should include:

- Files changed.
- Commands run.
- Any tests or checks that could not be run.
- Local URL for trying the app.

## Scope Boundaries

This first version will not include:

- Cloud LLM providers.
- User authentication.
- Persistent multi-user chat history.
- Admin upload workflows.
- Production deployment.
- External customer database integrations.

Documents are read from a local folder path on the machine running the server.
