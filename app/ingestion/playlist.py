"""YouTube playlist extraction using yt-dlp."""

import yt_dlp
from typing import List
from app.models.schemas import VideoMetadata
from app.utils.logging import get_logger

logger = get_logger(__name__)


def extract_playlist_metadata(playlist_url: str) -> List[VideoMetadata]:
    """
    Extract metadata for all videos in a YouTube playlist.
    Does not download audio/video.
    """
    logger.info(f"Extracting metadata from playlist: {playlist_url}")

    # We only want metadata, no downloading
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,  # Skip private/deleted videos
    }

    videos = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(playlist_url, download=False)

            if not info:
                logger.error("Could not extract playlist information.")
                return []

            playlist_id = info.get('id')
            playlist_title = info.get('title')
            entries = info.get('entries', [])
            if not entries:
                # Might be a single video URL, not a playlist
                entries = [info]
                playlist_id = playlist_id or info.get('id')
                playlist_title = playlist_title or info.get('title')

            for idx, entry in enumerate(entries):
                if not entry:
                    continue

                video_id = entry.get('id')
                title = entry.get('title')

                if not video_id or not title:
                    continue

                # yt-dlp flat extraction might not have all fields, but has basics
                metadata = VideoMetadata(
                    video_id=video_id,
                    title=title,
                    webpage_url=entry.get('url') or f"https://www.youtube.com/watch?v={video_id}",
                    playlist_id=playlist_id,
                    playlist_title=playlist_title,
                    duration=entry.get('duration'),
                    uploader=entry.get('uploader'),
                    channel=entry.get('channel'),
                    playlist_index=idx + 1,
                )
                videos.append(metadata)

            logger.info(f"Successfully extracted metadata for {len(videos)} videos.")
            return videos

        except Exception as e:
            logger.error(f"Error extracting playlist metadata: {e}")
            return []
