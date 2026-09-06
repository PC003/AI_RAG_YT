"""Qdrant vector store for storing and retrieving chunks."""

from typing import List, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

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
        self.client = QdrantClient(url=url, api_key=api_key or None)
        self.collection_name = collection_name
        self.embedder = None
        self.dimension = None

        logger.info(f"Qdrant client initialized for collection '{collection_name}'")

    def _ensure_collection_exists(self) -> None:
        """Create the collection if it does not exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            embedder = self._get_embedder()
            self.dimension = embedder.dimension
            logger.info(f"Creating collection '{self.collection_name}' with dimension {self.dimension}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = get_embedder()
            self.dimension = self.embedder.dimension
        return self.embedder

    def ensure_collection(self) -> None:
        """Public compatibility wrapper for collection creation."""
        self._ensure_collection_exists()

    def upsert_chunks(self, chunks: List[Chunk], embeddings: Optional[List[List[float]]] = None) -> bool:
        """
        Upsert chunks into Qdrant.
        Generates embeddings when they are not provided and stores playlist-aware payloads.
        """
        if not chunks:
            logger.warning("No chunks to upsert")
            return True

        self._ensure_collection_exists()
        logger.info(f"Upserting {len(chunks)} chunks to Qdrant")

        if embeddings is None:
            texts = [chunk.text for chunk in chunks]
            embeddings = self._get_embedder().embed(texts)

        # Prepare points
        points = []
        for i, chunk in enumerate(chunks):
            # Include playlist_id so the same video can safely appear in multiple playlists.
            playlist_id = chunk.playlist_id or "default"
            id_string = f"{playlist_id}_{chunk.video_id}_{chunk.start_time}_{chunk.end_time}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_string))

            point = PointStruct(
                id=point_id,
                vector=embeddings[i],
                payload={
                    "chunk_id": chunk.chunk_id,
                    "playlist_id": chunk.playlist_id,
                    "playlist_title": chunk.playlist_title,
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

    def search(
        self,
        query: str,
        limit: int = 5,
        playlist_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Search for chunks similar to the query.
        When playlist_ids is provided, results are limited to those playlists.
        Returns list of RetrievedChunk with scores.
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []

        logger.info(f"Searching Qdrant for: '{query[:50]}...' (limit={limit})")

        self._ensure_collection_exists()
        query_embedding = self._get_embedder().embed_query(query)

        try:
            query_filter = self._build_playlist_filter(playlist_ids)
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
                results = response.points
            else:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )

            retrieved_chunks = []
            for point in results:
                payload = point.payload
                chunk = Chunk(
                    chunk_id=payload["chunk_id"],
                    playlist_id=payload.get("playlist_id"),
                    playlist_title=payload.get("playlist_title"),
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

    def video_indexed(self, video_id: str, playlist_id: Optional[str] = None) -> bool:
        """Return whether at least one chunk exists for a video in the collection."""
        must = [
            FieldCondition(key="video_id", match=MatchValue(value=video_id)),
        ]
        if playlist_id:
            must.append(FieldCondition(key="playlist_id", match=MatchValue(value=playlist_id)))

        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=must),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(points) > 0
        except Exception as e:
            logger.warning(f"Error checking indexed video {video_id}: {e}")
            return False

    def list_playlists(self) -> list[dict]:
        """Return playlist IDs/titles currently represented in Qdrant payloads."""
        playlists: dict[str, dict] = {}
        offset = None

        try:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in points:
                    payload = point.payload or {}
                    playlist_id = payload.get("playlist_id")
                    if not playlist_id:
                        continue
                    playlists[playlist_id] = {
                        "playlist_id": playlist_id,
                        "playlist_title": payload.get("playlist_title") or playlist_id,
                    }

                if offset is None:
                    break

        except Exception as e:
            logger.warning(f"Error listing playlists: {e}")

        return sorted(playlists.values(), key=lambda p: p["playlist_title"])

    def delete_playlist(self, playlist_id: str) -> bool:
        """Delete all chunks for a playlist from the shared collection."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="playlist_id",
                                match=MatchValue(value=playlist_id),
                            )
                        ]
                    )
                ),
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting playlist {playlist_id}: {e}")
            return False

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

    @staticmethod
    def _build_playlist_filter(playlist_ids: Optional[List[str]]) -> Optional[Filter]:
        if not playlist_ids:
            return None

        playlist_ids = [pid for pid in playlist_ids if pid]
        if not playlist_ids:
            return None

        conditions = [
            FieldCondition(key="playlist_id", match=MatchValue(value=playlist_id))
            for playlist_id in playlist_ids
        ]
        return Filter(should=conditions)


QdrantStore = QdrantVectorStore
