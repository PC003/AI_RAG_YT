"""Orchestrate the full ingestion pipeline for a YouTube playlist."""

from __future__ import annotations
from typing import Callable, Optional
from app.ingestion.playlist import extract_playlist_metadata as extract_playlist
from app.ingestion.downloader import download_audio
from app.ingestion.transcriber import transcribe, transcript_exists, load_transcript
from app.chunking.chunker import chunk_transcript
from app.embeddings.embedder import get_embedder
from app.vectorstore.qdrant_store import QdrantStore
from app.models.schemas import IngestionStats, VideoMetadata
from app.utils.logging import get_logger

logger = get_logger(__name__)


def process_playlist(
    playlist_url: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> IngestionStats:
    """Run the end-to-end ingestion pipeline.

    *progress_callback(status_msg, current_index, total)* is called after
    each video so a UI can show progress.
    """
    stats = IngestionStats()

    # Step 1 – extract playlist metadata
    logger.info("Extracting playlist: %s", playlist_url)
    try:
        videos = extract_playlist(playlist_url)
    except Exception as e:
        logger.error("Playlist extraction failed: %s", e)
        raise

    stats.videos_found = len(videos)

    if not videos:
        logger.warning("No videos found in playlist")
        return stats

    # Initialise embedder and vector store once
    embedder = get_embedder()
    store = QdrantStore()
    store.ensure_collection()

    # Step 2 – process each video
    for idx, video in enumerate(videos):
        _notify(progress_callback, f"Processing: {video.title}", idx + 1, len(videos))
        try:
            _process_video(video, embedder, store, stats)
        except Exception as e:
            logger.error("Failed to process video %s (%s): %s", video.video_id, video.title, e)
            stats.videos_failed += 1
            stats.failed_videos.append(f"{video.video_id} – {video.title}")

    _notify(progress_callback, "Done", len(videos), len(videos))
    logger.info(
        "Ingestion complete: found=%d processed=%d skipped=%d failed=%d chunks=%d",
        stats.videos_found, stats.videos_processed, stats.videos_skipped,
        stats.videos_failed, stats.chunks_created,
    )
    return stats


def _process_video(
    video: VideoMetadata,
    embedder,
    store: QdrantStore,
    stats: IngestionStats,
) -> None:
    """Process a single video through the full pipeline."""
    # Skip if already fully indexed
    if transcript_exists(video.video_id) and store.video_indexed(video.video_id):
        logger.info("Skipping already indexed video: %s", video.title)
        stats.videos_skipped += 1
        return

    # Download audio
    audio_path = download_audio(video.video_id, video.webpage_url)

    # Transcribe
    transcript = transcribe(audio_path, video.video_id, video.title)

    # Chunk
    chunks = chunk_transcript(transcript, video)

    if not chunks:
        logger.warning("No chunks generated for %s", video.video_id)
        stats.videos_skipped += 1
        return

    # Embed
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)

    # Upsert into Qdrant
    store.upsert_chunks(chunks, embeddings)

    stats.videos_processed += 1
    stats.chunks_created += len(chunks)
    logger.info("Indexed %d chunks for: %s", len(chunks), video.title)


def _notify(cb, msg, current, total):
    if cb:
        try:
            cb(msg, current, total)
        except Exception:
            pass
