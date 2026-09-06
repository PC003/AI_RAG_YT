"""Audio transcription using local Whisper model."""

import json
from pathlib import Path
from typing import Optional
import whisper
import torch

from app.config import TRANSCRIPTS_DIR, WHISPER_MODEL
from app.models.schemas import Transcript, TranscriptSegment
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Global model instance to avoid reloading
_WHISPER_MODEL_INSTANCE = None


def _get_whisper_model():
    """Lazy load the Whisper model."""
    global _WHISPER_MODEL_INSTANCE
    if _WHISPER_MODEL_INSTANCE is None:
        logger.info(f"Loading Whisper model '{WHISPER_MODEL}'... This might take a moment.")

        # Use MPS on Apple Silicon, CUDA on Nvidia, else CPU
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

        logger.info(f"Using device: {device}")
        _WHISPER_MODEL_INSTANCE = whisper.load_model(WHISPER_MODEL, device=device)

    return _WHISPER_MODEL_INSTANCE


def transcribe_audio(audio_path: Path, video_id: str, video_title: str) -> Optional[Transcript]:
    """
    Transcribe an audio file using Whisper.
    Saves the transcript to disk to avoid re-transcribing.
    """
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"

    # Check if already transcribed
    if transcript_path.exists():
        logger.info(f"Transcript for {video_id} already exists. Loading from disk.")
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Transcript(**data)
        except Exception as e:
            logger.warning(f"Failed to load existing transcript for {video_id}: {e}. Retranscribing.")

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return None

    logger.info(f"Transcribing {video_id} using Whisper...")

    try:
        model = _get_whisper_model()

        # Transcribe
        result = model.transcribe(str(audio_path))

        # Convert to our schema
        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip()
            ))

        transcript = Transcript(
            video_id=video_id,
            video_title=video_title,
            language=result.get("language"),
            segments=segments
        )

        # Save to disk
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript.model_dump_json(indent=2))

        logger.info(f"Successfully transcribed {video_id} ({len(segments)} segments)")
        return transcript

    except Exception as e:
        logger.error(f"Error transcribing {video_id}: {e}")
        return None
