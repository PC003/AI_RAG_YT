"""Timestamp-aware semantic chunking for transcripts."""

import hashlib
from typing import List
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.models.schemas import Transcript, Chunk, VideoMetadata
from app.utils.logging import get_logger
from app.utils.timestamps import youtube_timestamp_url

logger = get_logger(__name__)


def chunk_transcript(transcript: Transcript, metadata: VideoMetadata = None, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """
    Groups transcript segments into chunks while preserving timestamp boundaries.
    Uses character-based sizing with sentence boundary awareness.
    """
    logger.info(f"Chunking transcript for {transcript.video_id}")

    if not transcript.segments:
        logger.warning(f"No segments found in transcript for {transcript.video_id}")
        return []

    chunks = []
    current_chunk_text = ""
    current_chunk_segments = []

    # Simple character-based estimation (1 token ≈ 4 characters)
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4

    i = 0
    while i < len(transcript.segments):
        seg = transcript.segments[i]

        # Add to current chunk
        if current_chunk_text:
            current_chunk_text += " " + seg.text
        else:
            current_chunk_text = seg.text

        current_chunk_segments.append(seg)

        # Check if chunk is full (or it's the last segment)
        if len(current_chunk_text) >= char_size or i == len(transcript.segments) - 1:
            # We have a complete chunk. Generate metadata based on its segments
            start_time = current_chunk_segments[0].start
            end_time = current_chunk_segments[-1].end

            # Generate stable chunk ID
            chunk_id_str = f"{transcript.video_id}_{start_time:.1f}_{end_time:.1f}"
            chunk_id = hashlib.md5(chunk_id_str.encode()).hexdigest()

            chunk = Chunk(
                chunk_id=chunk_id,
                video_id=transcript.video_id,
                video_title=transcript.video_title,
                text=current_chunk_text.strip(),
                start_time=start_time,
                end_time=end_time,
                youtube_url=youtube_timestamp_url(transcript.video_id, start_time),
                playlist_index=metadata.playlist_index if metadata else None
            )
            chunks.append(chunk)

            # Setup for next chunk with overlap
            if i < len(transcript.segments) - 1:
                # Go back slightly to create overlap, if we have enough segments
                overlap_text = ""
                overlap_segments = []

                # Walk backward to build the overlap
                for j in range(len(current_chunk_segments) - 1, -1, -1):
                    back_seg = current_chunk_segments[j]
                    if len(overlap_text) + len(back_seg.text) > char_overlap:
                        break
                    overlap_segments.insert(0, back_seg)
                    overlap_text = back_seg.text + " " + overlap_text if overlap_text else back_seg.text

                current_chunk_text = overlap_text.strip()
                current_chunk_segments = overlap_segments
            else:
                current_chunk_text = ""
                current_chunk_segments = []

        i += 1

    logger.info(f"Created {len(chunks)} chunks for {transcript.video_id}")
    return chunks
