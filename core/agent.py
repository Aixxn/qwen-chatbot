import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import lancedb
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from core.document_loader import (
    DEFAULT_EMBEDDING_MODEL,
    get_db_connection,
    get_embedding_function,
    load_documents_into_database,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentDeps:
    """Dependencies passed to the RAG agent tools."""

    embedding_model: str
    vector_store: lancedb.table.Table


def _create_vector_store(
    embedding_model: str,
    documents_path: str = "documents",
    reload: bool = True,
) -> lancedb.table.Table:
    """Create or load the LanceDB vector store."""
    if reload:
        return load_documents_into_database(
            embedding_model, documents_path, reload=True
        )

    db = get_db_connection()
    if "documents" in db.table_names():
        return db.open_table("documents")
    return load_documents_into_database(embedding_model, documents_path, reload=True)


@dataclass
class CustomerServiceAgent:
    """Agentic RAG system for customer-service document queries."""

    llm_model: str
    embedding_model: str
    vector_store: lancedb.table.Table = field(
        default_factory=lambda: _create_vector_store(
            DEFAULT_EMBEDDING_MODEL, reload=False
        )
    )
    agent: Agent[AgentDeps, str] = field(init=False)
    search_log: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        model = OpenAIChatModel(
            model_name=self.llm_model,
            provider=OllamaProvider(base_url="http://localhost:11434/v1"),
        )

        self.agent = Agent(
            model,
            deps_type=AgentDeps,
            system_prompt=(
                "You are a customer-service document assistant. Answer questions "
                "using the indexed support documents, policies, product notes, "
                "troubleshooting guides, and operational documents.\n\n"
                "Always use the search_documents tool for questions about policies, "
                "pricing, returns, refunds, warranties, shipping, product details, "
                "troubleshooting, workflows, support procedures, or anything that "
                "could be answered by the documents. Do not rely on general model "
                "knowledge for document-specific answers.\n\n"
                "When searching, write complete self-contained search queries. Avoid "
                "pronouns such as 'this', 'that', or 'it'. For compound questions, "
                "run multiple focused searches instead of one broad query.\n\n"
                "If the documents do not contain enough information, say that the "
                "answer is not available from the current documents. Do not claim "
                "access to order systems, customer accounts, private customer data, "
                "or live business systems."
            ),
        )

        embedding_func = get_embedding_function(self.embedding_model)

        @self.agent.tool
        async def search_documents(ctx: RunContext[AgentDeps], query: str) -> str:
            """Search relevant customer-service document sections."""
            self.search_log.append(query)
            query_embedding = embedding_func.compute_query_embeddings(query)[0]
            results = ctx.deps.vector_store.search(query_embedding).limit(10).to_list()

            if not results:
                return "No relevant documents found."

            result_parts: list[str] = []
            for doc in results:
                source = doc.get("source", "Unknown")
                page = doc.get("page", "N/A")
                text = doc.get("text", "")
                result_parts.append(f"[Source: {source}, Page: {page}]")
                result_parts.append(text)

            return "\n\n".join(result_parts)

    def _deps(self) -> AgentDeps:
        return AgentDeps(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
        )

    def ask(
        self, question: str, message_history: list[ModelMessage] | None = None
    ) -> tuple[str, list[str]]:
        """Answer a question and return the answer plus search queries."""
        self.search_log = []
        result = self.agent.run_sync(
            question, deps=self._deps(), message_history=message_history
        )
        return result.output, self.search_log.copy()

    def stream(
        self, question: str, message_history: list[ModelMessage] | None = None
    ) -> Iterator[tuple[str, Any]]:
        """Stream an answer as text chunks after yielding tool-call events."""
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        self.search_log = []
        response = self.agent.run_stream_sync(
            question, deps=self._deps(), message_history=message_history
        )

        for message in response.new_messages():
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        args = part.args_as_dict()
                        query = str(args.get("query", "documents"))
                        if query not in self.search_log:
                            self.search_log.append(query)
                        yield ("search", {"query": query})

        last_text = ""
        for text in response.stream_text():
            new_text = text[len(last_text) :]
            if new_text:
                yield ("text", new_text)
            last_text = text


def create_customer_service_agent(
    llm_model: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    documents_path: str = "documents",
    reload: bool = False,
) -> CustomerServiceAgent:
    """Create a configured customer-service RAG agent."""
    vector_store = _create_vector_store(embedding_model, documents_path, reload=reload)
    return CustomerServiceAgent(
        llm_model=llm_model,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
