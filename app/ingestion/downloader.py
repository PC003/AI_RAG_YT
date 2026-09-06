"""YouTube audio downloading using yt-dlp."""

import os
from pathlib import Path
from typing import Optional
import yt_dlp

from app.config import AUDIO_DIR
from app.utils.logging import get_logger

logger = get_logger(__name__)


def download_audio(video_url: str, video_id: str) -> Optional[Path]:
    """
    Download audio for a YouTube video.
    Returns the path to the downloaded audio file, or None if failed.
    """
    # Expected output path without extension
    out_tmpl = str(AUDIO_DIR / f"{video_id}")

    # Check if we already have it downloaded (any extension)
    existing_files = list(AUDIO_DIR.glob(f"{video_id}.*"))
    if existing_files:
        logger.info(f"Audio for {video_id} already exists. Skipping download.")
        return existing_files[0]

    logger.info(f"Downloading audio for {video_id}...")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{out_tmpl}.%(ext)s",
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])

            # Find the downloaded file
            downloaded = list(AUDIO_DIR.glob(f"{video_id}.*"))
            if downloaded:
                logger.info(f"Successfully downloaded audio for {video_id}")
                return downloaded[0]
            else:
                logger.error(f"Download completed but file not found for {video_id}")
                return None

        except Exception as e:
            logger.error(f"Error downloading audio for {video_id}: {e}")
            return None
