"""Retrieval pipeline for RAG."""

from typing import List, Optional
from app.config import TOP_K
from app.models.schemas import RAGResponse, RetrievedChunk
from app.rag.generator import generate_answer
from app.utils.logging import get_logger
from app.vectorstore.qdrant_store import QdrantVectorStore

logger = get_logger(__name__)

# Singleton instance
_RETRIEVER_INSTANCE = None


def get_retriever() -> QdrantVectorStore:
    """Get or create the retriever instance."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = QdrantVectorStore()
    return _RETRIEVER_INSTANCE


def retrieve_relevant_chunks(
    query: str,
    top_k: int = TOP_K,
    playlist_ids: Optional[List[str]] = None,
) -> List[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for a query.
    """
    logger.info(f"Retrieving top {top_k} chunks for query: '{query[:50]}...'")
    retriever = get_retriever()
    return retriever.search(query, limit=top_k, playlist_ids=playlist_ids)


def answer_query(query: str, playlist_ids: Optional[List[str]] = None) -> RAGResponse:
    """Answer a query using retrieved chunks and the configured LLM."""
    if not query.strip():
        return RAGResponse(answer="Please provide a question.", sources=[])

    chunks = retrieve_relevant_chunks(query, TOP_K, playlist_ids=playlist_ids)
    if not chunks:
        return RAGResponse(
            answer="I couldn't find relevant information in the processed videos to answer that.",
            sources=[],
        )

    return generate_answer(query, chunks)
