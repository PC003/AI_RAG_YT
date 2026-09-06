"""Generate text embeddings using SentenceTransformers."""

from __future__ import annotations
from typing import Optional
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Singleton
_embedder: Optional["Embedder"] = None


class Embedder:
    """Wrapper for sentence-transformers model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        # Verify dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded. Dimension: %d", self.dimension)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        # Return as plain float lists for Qdrant
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single search query."""
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()


def get_embedder() -> Embedder:
    """Return the cached embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder