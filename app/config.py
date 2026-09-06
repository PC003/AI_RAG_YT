"""Central configuration loaded from environment variables / .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# --- Paths ---
DATA_DIR = Path(_get("DATA_DIR", str(_PROJECT_ROOT / "data")))
AUDIO_DIR = Path(_get("AUDIO_DIR", str(DATA_DIR / "audio")))
TRANSCRIPTS_DIR = Path(_get("TRANSCRIPTS_DIR", str(DATA_DIR / "transcripts")))
METADATA_DIR = Path(_get("METADATA_DIR", str(DATA_DIR / "metadata")))

# Ensure directories exist
for d in (AUDIO_DIR, TRANSCRIPTS_DIR, METADATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Whisper ---
WHISPER_MODEL: str = _get("WHISPER_MODEL", "base")

# --- Embedding ---
EMBEDDING_MODEL: str = _get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Qdrant ---
QDRANT_URL: str = _get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str = _get("QDRANT_API_KEY", "")
QDRANT_COLLECTION: str = _get("QDRANT_COLLECTION", "youtube_chunks")

# --- Chunking ---
CHUNK_SIZE: int = int(_get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(_get("CHUNK_OVERLAP", "100"))

# --- Retrieval ---
TOP_K: int = int(_get("TOP_K", "5"))

# --- LLM ---
LLM_PROVIDER: str = _get("LLM_PROVIDER", "ollama")
LLM_MODEL: str = _get("LLM_MODEL", "llama3")
LLM_API_KEY: str = _get("LLM_API_KEY", "")
LLM_BASE_URL: str = _get("LLM_BASE_URL", "http://localhost:11434")

# --- Logging ---
LOG_LEVEL: str = _get("LOG_LEVEL", "INFO")
