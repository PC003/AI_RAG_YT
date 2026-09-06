"""Qdrant vector store for storing and retrieving chunks."""

from typing import List, Optional
from uuid import uuid4
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from app.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from app.embeddings.embedder import get_embedder
from app.models.schemas import Chunk, RetrievedChunk
from app.utils.logging import get_logger

logger = get_logger(__name__)


class QdrantVectorStore:
    """Handles Qdrant vector storage and retrieval operations."""

    def __init__(
        self,
        url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
        collection_name: str = QDRANT_COLLECTION,
    ):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.embedder = get_embedder()
        self.dimension = self.embedder.model.get_sentence_embedding_dimension()

        logger.info(f"Qdrant client initialized for collection '{collection_name}'")
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Create the collection if it does not exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection '{self.collection_name}' with dimension {self.dimension}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

    def upsert_chunks(self, chunks: List[Chunk]) -> bool:
        """
        Upsert chunks into Qdrant.
        Generates embeddings and stores them with payload.
        """
        if not chunks:
            logger.warning("No chunks to upsert")
            return True

        logger.info(f"Upserting {len(chunks)} chunks to Qdrant")

        # Extract texts for embedding
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed(texts)

        # Prepare points
        points = []
        for i, chunk in enumerate(chunks):
            # Use a deterministic ID based on video_id and timestamps to allow safe re-runs
            id_string = f"{chunk.video_id}_{chunk.start_time}_{chunk.end_time}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_string))

            point = PointStruct(
                id=point_id,
                vector=embeddings[i],
                payload={
                    "chunk_id": chunk.chunk_id,
                    "video_id": chunk.video_id,
                    "video_title": chunk.video_title,
                    "text": chunk.text,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "youtube_url": chunk.youtube_url,
                    "playlist_index": chunk.playlist_index,
                },
            )
            points.append(point)

        # Upsert in batches
        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Successfully upserted {len(points)} chunks")
            return True
        except Exception as e:
            logger.error(f"Error upserting chunks to Qdrant: {e}")
            return False

    def search(self, query: str, limit: int = 5) -> List[RetrievedChunk]:
        """
        Search for chunks similar to the query.
        Returns list of RetrievedChunk with scores.
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []

        logger.info(f"Searching Qdrant for: '{query[:50]}...' (limit={limit})")

        # Embed the query
        query_embedding = self.embedder.embed_query(query)

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            retrieved_chunks = []
            for point in results:
                payload = point.payload
                chunk = Chunk(
                    chunk_id=payload["chunk_id"],
                    video_id=payload["video_id"],
                    video_title=payload["video_title"],
                    text=payload["text"],
                    start_time=payload["start_time"],
                    end_time=payload["end_time"],
                    youtube_url=payload["youtube_url"],
                    playlist_index=payload.get("playlist_index"),
                )
                retrieved_chunks.append(RetrievedChunk(chunk=chunk, score=point.score))

            logger.info(f"Found {len(retrieved_chunks)} relevant chunks")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []

    def get_collection_info(self):
        """Get information about the collection."""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None

    def delete_collection(self) -> bool:
        """Delete the collection (use with caution)."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False