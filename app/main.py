"""Main orchestrator for the YouTube Playlist RAG system."""

import logging
from typing import List, Optional
from app.config import *
from app.models.schemas import (
    VideoMetadata, Transcript, Chunk,
    IngestionStats, RAGResponse
)
from app.ingestion.playlist import extract_playlist_metadata
from app.ingestion.downloader import download_audio
from app.ingestion.transcriber import transcribe_audio
from app.chunking.chunker import create_chunks
from app.embeddings.embedder import embed_texts
from app.vectorstore.qdrant_store import QdrantVectorStore
from app.rag.retriever import retrieve_relevant_chunks
from app.rag.generator import generate_answer
from app.utils.logging import get_logger

logger = get_logger(__name__)


class YouTubeRAGOrchestrator:
    """Orchestrates the entire YouTube Playlist RAG pipeline."""

    def __init__(self):
        self.qdrant_store = QdrantVectorStore()
        logger.info("YouTubeRAGOrchestrator initialized")

    def process_playlist(self, playlist_url: str) -> IngestionStats:
        """
        Process a YouTube playlist: extract metadata, download audio,
        transcribe, chunk, embed, and store in Qdrant.
        """
        logger.info(f"Starting playlist ingestion: {playlist_url}")
        stats = IngestionStats()

        # Step 1: Extract playlist metadata
        videos = extract_playlist_metadata(playlist_url)
        if not videos:
            logger.error("No videos found in playlist")
            return stats

        stats.videos_found = len(videos)
        logger.info(f"Found {stats.videos_found} videos in playlist")

        # Step 2: Process each video
        for video in videos:
            try:
                logger.info(f"Processing video {video.playlist_index}/{len(videos)}: {video.title}")

                # Check if we already have this video fully processed
                if self._is_video_fully_processed(video.video_id):
                    logger.info(f"Video {video.video_id} already fully processed. Skipping.")
                    stats.videos_skipped += 1
                    continue

                # Download audio
                audio_path = download_audio(video.webpage_url, video.video_id)
                if not audio_path:
                    logger.error(f"Failed to download audio for {video.video_id}")
                    stats.videos_failed += 1
                    stats.failed_videos.append(video.video_id)
                    continue

                # Transcribe audio
                transcript = transcribe_audio(audio_path, video.video_id, video.title)
                if not transcript:
                    logger.error(f"Failed to transcribe audio for {video.video_id}")
                    stats.videos_failed += 1
                    stats.failed_videos.append(video.video_id)
                    continue

                # Create chunks
                chunks = create_chunks(transcript)
                if not chunks:
                    logger.warning(f"No chunks created for {video.video_id}")
                    stats.videos_failed += 1
                    stats.failed_videos.append(video.video_id)
                    continue

                # Upsert to Qdrant
                if self.qdrant_store.upsert_chunks(chunks):
                    stats.videos_processed += 1
                    stats.chunks_created += len(chunks)
                    logger.info(f"Successfully processed {video.video_id}: {len(chunks)} chunks")
                else:
                    logger.error(f"Failed to upsert chunks for {video.video_id}")
                    stats.videos_failed += 1
                    stats.failed_videos.append(video.video_id)

            except Exception as e:
                logger.error(f"Unexpected error processing video {video.video_id}: {e}", exc_info=True)
                stats.videos_failed += 1
                stats.failed_videos.append(video.video_id)

        logger.info(f"Ingestion complete: {stats}")
        return stats

    def _is_video_fully_processed(self, video_id: str) -> bool:
        """Check if a video has already been fully processed (transcript + chunks in Qdrant)."""
        # Check if transcript exists
        transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"
        if not transcript_path.exists():
            return False

        # Check if we have chunks in Qdrant for this video
        try:
            # Search for any chunk from this video
            results = self.qdrant_store.client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=video_id)
                        )
                    ]
                ),
                limit=1
            )
            return len(results[0]) > 0
        except Exception as e:
            logger.warning(f"Error checking if video {video_id} is processed: {e}")
            return False

    def answer_query(self, query: str) -> RAGResponse:
        """
        Answer a user query using the RAG pipeline.
        """
        logger.info(f"Answering query: '{query[:50]}...'")

        if not query.strip():
            return RAGResponse(
                answer="Please provide a question.",
                sources=[]
            )

        # Retrieve relevant chunks
        retrieved_chunks = retrieve_relevant_chunks(query, TOP_K)

        if not retrieved_chunks:
            return RAGResponse(
                answer="I couldn't find any relevant information in the processed videos to answer your question.",
                sources=[]
            )

        # Generate answer using LLM
        response = generate_answer(query, retrieved_chunks)
        logger.info(f"Generated answer with {len(response.sources)} sources")
        return response

    def get_stats(self) -> dict:
        """Get statistics about the current state."""
        try:
            collection_info = self.qdrant_store.get_collection_info()
            return {
                "collection_name": QDRANT_COLLECTION,
                "vector_count": collection_info.points_count if collection_info else 0,
                "status": "ready" if collection_info else "error"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}