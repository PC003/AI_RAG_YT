"""Unit tests for utility functions."""

import pytest
from app.utils.timestamps import seconds_to_hms, youtube_timestamp_url


def test_seconds_to_hms():
    """Test seconds to HH:MM:SS conversion."""
    # Test MM:SS format (less than 1 hour)
    assert seconds_to_hms(65) == "01:05"
    assert seconds_to_hms(125) == "02:05"
    assert seconds_to_hms(3599) == "59:59"

    # Test HH:MM:SS format (1 hour or more)
    assert seconds_to_hms(3600) == "01:00:00"
    assert seconds_to_hms(3665) == "01:01:05"
    assert seconds_to_hms(7265) == "02:01:05"


def test_youtube_timestamp_url():
    """Test YouTube timestamp URL generation."""
    video_id = "dQw4w9WgXcQ"

    # Test basic timestamp
    url = youtube_timestamp_url(video_id, 65)
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=65s"

    # Test zero timestamp
    url = youtube_timestamp_url(video_id, 0)
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=0s"

    # Test larger timestamp
    url = youtube_timestamp_url(video_id, 3665)
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=3665s"


def test_empty_inputs():
    """Test edge cases with empty/minimal inputs."""
    assert seconds_to_hms(0) == "00:00"
    assert youtube_timestamp_url("test", 0.0) == "https://www.youtube.com/watch?v=test&t=0s"