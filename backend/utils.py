"""
Utility Functions
=================
Common helpers for logging, timing, and data processing.
"""

import logging
import time
from functools import wraps
from typing import Callable
import sys


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def time_it(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS string."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def calculate_fps(num_frames: int, elapsed_time: float) -> float:
    """Calculate frames per second."""
    if elapsed_time == 0:
        return 0.0
    return num_frames / elapsed_time


def normalize_pose(landmarks: "np.ndarray") -> "np.ndarray":
    """
    Normalize pose landmarks for scale and translation invariance.
    Centers pose at origin and scales to unit distance.
    """
    import numpy as np

    center = landmarks.mean(axis=0)
    centered = landmarks - center
    scale = np.linalg.norm(centered, axis=1).max()
    return centered / scale if scale > 0 else centered


def smooth_landmarks(landmark_history: list, window_size: int = 5) -> "np.ndarray":
    """Apply a moving average to reduce jitter in pose tracking."""
    import numpy as np

    if len(landmark_history) < window_size:
        return landmark_history[-1]
    return np.mean(landmark_history[-window_size:], axis=0)


def get_video_info(video_path: str) -> dict:
    """Return basic metadata for a video file."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "fps": fps,
        "total_frames": total_frames,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": total_frames / fps if fps > 0 else 0,
    }
    cap.release()
    return info


class PerformanceTimer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, logger=None):
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name} took {elapsed:.3f}s")
