"""Unit tests for YouTube RAG components."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.timestamps import seconds_to_hms, youtube_timestamp_url
from app.models.schemas import Transcript, TranscriptSegment, VideoMetadata, Chunk
from app.chunking.chunker import chunk_transcript


def test_seconds_to_hms():
    """Test timestamp formatting."""
    assert seconds_to_hms(45.5) == "00:45"
    assert seconds_to_hms(124.5) == "02:04"
    assert seconds_to_hms(3665.2) == "01:01:05"
    assert seconds_to_hms(0) == "00:00"


def test_youtube_timestamp_url():
    """Test YouTube URL generation with timestamp."""
    url = youtube_timestamp_url("dQw4w9WgXcQ", 124.5)
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=124s"

    url2 = youtube_timestamp_url("abc123", 0)
    assert url2 == "https://www.youtube.com/watch?v=abc123&t=0s"


def test_chunking_preserves_timestamps():
    """Test that chunking maintains timestamp boundaries."""
    metadata = VideoMetadata(
        video_id="test1",
        title="Test Video",
        webpage_url="http://youtube.com/watch?v=test1",
        playlist_index=1
    )

    transcript = Transcript(
        video_id="test1",
        video_title="Test Video",
        segments=[
            TranscriptSegment(start=0.0, end=5.0, text="First segment with some words here"),
            TranscriptSegment(start=5.0, end=10.0, text="Second segment with more content"),
            TranscriptSegment(start=10.0, end=15.0, text="Third segment final words"),
        ]
    )

    chunks = chunk_transcript(transcript, metadata, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 0, "Should create at least one chunk"

    # Verify first chunk has correct start time
    assert chunks[0].start_time == 0.0, "First chunk should start at 0.0"

    # Verify all chunks have required metadata
    for chunk in chunks:
        assert chunk.video_id == "test1"
        assert chunk.video_title == "Test Video"
        assert len(chunk.text) > 0
        assert chunk.start_time >= 0
        assert chunk.end_time > chunk.start_time
        assert "youtube.com" in chunk.youtube_url


def test_chunking_empty_transcript():
    """Test handling of empty transcript."""
    metadata = VideoMetadata(
        video_id="empty",
        title="Empty Video",
        webpage_url="http://youtube.com/watch?v=empty"
    )

    transcript = Transcript(
        video_id="empty",
        video_title="Empty Video",
        segments=[]
    )

    chunks = chunk_transcript(transcript, metadata)
    assert len(chunks) == 0, "Empty transcript should produce no chunks"


def test_video_metadata_model():
    """Test VideoMetadata pydantic model."""
    data = {
        "video_id": "abc123",
        "title": "Test Video Title",
        "webpage_url": "https://youtube.com/watch?v=abc123",
        "duration": 120.5,
        "uploader": "Test Channel",
        "playlist_index": 5
    }

    video = VideoMetadata(**data)
    assert video.video_id == "abc123"
    assert video.title == "Test Video Title"
    assert video.duration == 120.5
    assert video.playlist_index == 5


def test_transcript_segment_model():
    """Test TranscriptSegment pydantic model."""
    seg = TranscriptSegment(start=10.5, end=20.3, text="Hello world")

    assert seg.start == 10.5
    assert seg.end == 20.3
    assert seg.text == "Hello world"


def test_chunk_model():
    """Test Chunk pydantic model."""
    chunk = Chunk(
        chunk_id="abc123hash",
        video_id="vid1",
        video_title="My Video",
        text="Some transcript text",
        start_time=10.0,
        end_time=25.5,
        youtube_url="https://youtube.com/watch?v=vid1&t=10s",
        playlist_index=3
    )

    assert chunk.chunk_id == "abc123hash"
    assert chunk.video_id == "vid1"
    assert chunk.start_time == 10.0
    assert chunk.end_time == 25.5
    assert chunk.playlist_index == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
