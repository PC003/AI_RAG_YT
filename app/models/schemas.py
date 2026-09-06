"""Pydantic models for data flowing through the pipeline."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Metadata for a single YouTube video."""
    video_id: str
    title: str
    webpage_url: str
    playlist_id: Optional[str] = None
    playlist_title: Optional[str] = None
    duration: Optional[float] = None
    uploader: Optional[str] = None
    channel: Optional[str] = None
    playlist_index: Optional[int] = None
    upload_date: Optional[str] = None
    description: Optional[str] = None


class TranscriptSegment(BaseModel):
    """A single Whisper transcript segment with timestamps."""
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    """Full transcript for a video."""
    video_id: str
    video_title: str
    playlist_id: Optional[str] = None
    playlist_title: Optional[str] = None
    language: Optional[str] = None
    segments: list[TranscriptSegment] = Field(default_factory=list)


class Chunk(BaseModel):
    """A text chunk with timestamp and source metadata."""
    chunk_id: str
    playlist_id: Optional[str] = None
    playlist_title: Optional[str] = None
    video_id: str
    video_title: str
    text: str
    start_time: float
    end_time: float
    youtube_url: str
    playlist_index: Optional[int] = None


class RetrievedChunk(BaseModel):
    """A chunk returned from vector search, including its similarity score."""
    chunk: Chunk
    score: float


class Source(BaseModel):
    """A source reference shown in the final answer."""
    video_title: str
    video_id: str
    playlist_id: Optional[str] = None
    playlist_title: Optional[str] = None
    start_time: float
    end_time: float
    youtube_url: str
    timestamp_display: str = ""


class RAGResponse(BaseModel):
    """The final response returned to the user."""
    answer: str
    sources: list[Source] = Field(default_factory=list)


class IngestionStats(BaseModel):
    """Statistics from a playlist ingestion run."""
    videos_found: int = 0
    videos_processed: int = 0
    videos_skipped: int = 0
    videos_failed: int = 0
    chunks_created: int = 0
    failed_videos: list[str] = Field(default_factory=list)
