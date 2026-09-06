"""Abstract LLM generation for RAG answers."""

from __future__ import annotations
from typing import Optional
from app.config import LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from app.models.schemas import RAGResponse
from app.utils.logging import get_logger
from app.rag.prompt import build_prompt, format_sources, build_context

logger = get_logger(__name__)


def generate_answer(query: str, chunks: list) -> RAGResponse:
    """Generate an answer using the configured LLM provider."""
    if not chunks:
        return RAGResponse(
            answer="I couldn't find any relevant information in the video transcripts to answer your question.",
            sources=[]
        )

    # Build prompt
    context = build_context(chunks)
    prompt = build_prompt(query, context)

    try:
        if LLM_PROVIDER.lower() == "openai":
            return _generate_openai(prompt, chunks)
        elif LLM_PROVIDER.lower() == "anthropic":
            return _generate_anthropic(prompt, chunks)
        elif LLM_PROVIDER.lower() == "ollama":
            return _generate_ollama(prompt, chunks)
        else:
            raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        return RAGResponse(
            answer=f"I encountered an error while generating the answer: {str(e)}",
            sources=[]
        )


def _generate_openai(prompt: str, chunks: list) -> RAGResponse:
    """Generate answer using OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=LLM_API_KEY)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    answer = response.choices[0].message.content.strip()
    sources = format_sources(chunks)

    return RAGResponse(answer=answer, sources=sources)


def _generate_anthropic(prompt: str, chunks: list) -> RAGResponse:
    """Generate answer using Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key=LLM_API_KEY)

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1000,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.content[0].text.strip()
    sources = format_sources(chunks)

    return RAGResponse(answer=answer, sources=sources)


def _generate_ollama(prompt: str, chunks: list) -> RAGResponse:
    """Generate answer using Ollama API."""
    import requests
    import json

    url = f"{LLM_BASE_URL}/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 1000
        }
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    result = response.json()
    answer = result.get("response", "").strip()
    sources = format_sources(chunks)

    return RAGResponse(answer=answer, sources=sources)