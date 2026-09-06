"""Timestamp formatting and YouTube URL helpers."""


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS display string."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def youtube_timestamp_url(video_id: str, start_seconds: float) -> str:
    """Build a YouTube URL that starts playback at *start_seconds*."""
    t = int(start_seconds)
    return f"https://www.youtube.com/watch?v={video_id}&t={t}s"
