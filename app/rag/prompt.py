"""Prompt templates for RAG generation."""

from __future__ import annotations
from typing import List
from app.models.schemas import RetrievedChunk, Source


def build_context(chunks: List[RetrievedChunk]) -> str:
    """Build a context string from retrieved chunks for the LLM."""
    if not chunks:
        return "No relevant context found."

    context_parts = []
    for i, rc in enumerate(chunks, 1):
        chunk = rc.chunk
        timestamp_display = f"[{int(chunk.start_time // 60):02d}:{int(chunk.start_time % 60):02d}]"
        context_parts.append(
            f"SOURCE {i}:\n"
            f"Video: {chunk.video_title}\n"
            f"Timestamp: {timestamp_display}\n"
            f"Text: {chunk.text}\n"
        )

    return "\n---\n".join(context_parts)


def build_prompt(query: str, context: str) -> str:
    """Build the final prompt sent to the LLM."""
    return f"""You are a helpful assistant that answers questions based on YouTube video transcripts.

Use ONLY the provided context to answer the question. If the context doesn't contain the information needed to answer the question, say so clearly.

When answering:
- Be concise and accurate
- Cite specific sources by referencing the video title and timestamp
- If multiple sources support your answer, mention them
- Do not make up information not present in the context

Context:
{context}

QUESTION: {query}

ANSWER:"""


def format_sources(chunks: List[RetrievedChunk]) -> List[Source]:
    """Convert retrieved chunks to Source objects for display."""
    sources = []
    for rc in chunks:
        chunk = rc.chunk
        sources.append(Source(
            video_title=chunk.video_title,
            video_id=chunk.video_id,
            playlist_id=chunk.playlist_id,
            playlist_title=chunk.playlist_title,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            youtube_url=chunk.youtube_url,
            timestamp_display=f"{int(chunk.start_time // 60):02d}:{int(chunk.start_time % 60):02d}"
        ))
    return sources
