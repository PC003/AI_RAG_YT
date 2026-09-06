"""Compatibility orchestrator for the YouTube Playlist RAG system."""

from __future__ import annotations

from typing import Optional

from app.config import QDRANT_COLLECTION
from app.ingestion.processor import process_playlist as run_ingestion
from app.models.schemas import IngestionStats, RAGResponse
from app.rag.retriever import answer_query as run_answer_query
from app.vectorstore.qdrant_store import QdrantVectorStore
from app.utils.logging import get_logger

logger = get_logger(__name__)


class YouTubeRAGOrchestrator:
    """Small facade used by the richer Streamlit frontend."""

    def __init__(self):
        self.qdrant_store = QdrantVectorStore()
        logger.info("YouTubeRAGOrchestrator initialized")

    def process_playlist(self, playlist_url: str) -> IngestionStats:
        """Process a playlist through the canonical ingestion pipeline."""
        return run_ingestion(playlist_url)

    def answer_query(
        self,
        query: str,
        playlist_ids: Optional[list[str]] = None,
    ) -> RAGResponse:
        """Answer a query, optionally limited to one or more playlists."""
        return run_answer_query(query, playlist_ids=playlist_ids)

    def list_playlists(self) -> list[dict]:
        """Return playlists currently indexed in Qdrant."""
        return self.qdrant_store.list_playlists()

    def get_stats(self) -> dict:
        """Get statistics about the current Qdrant collection."""
        try:
            collection_info = self.qdrant_store.get_collection_info()
            playlists = self.qdrant_store.list_playlists()
            return {
                "collection_name": QDRANT_COLLECTION,
                "vector_count": collection_info.points_count if collection_info else 0,
                "playlist_count": len(playlists),
                "status": "ready" if collection_info else "error",
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}
