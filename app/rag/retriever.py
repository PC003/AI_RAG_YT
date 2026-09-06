"""Retrieval pipeline for RAG."""

from typing import List
from app.config import TOP_K
from app.models.schemas import RetrievedChunk
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


def retrieve_relevant_chunks(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for a query.
    """
    logger.info(f"Retrieving top {top_k} chunks for query: '{query[:50]}...'")
    retriever = get_retriever()
    return retriever.search(query, limit=top_k)