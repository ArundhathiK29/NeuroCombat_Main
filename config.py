"""
Configuration
=============
Central settings for all NeuroCombat components.
Edit this file to tune detection, classification, and commentary behaviour.
"""

from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).parent
DATA_DIR    = PROJECT_ROOT / "data"
MODELS_DIR  = PROJECT_ROOT / "models"
OUTPUT_DIR  = PROJECT_ROOT / "output"

for _d in (DATA_DIR, MODELS_DIR, OUTPUT_DIR):
    _d.mkdir(exist_ok=True)


@dataclass
class PoseConfig:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float  = 0.5
    model_complexity: int           = 1   # 0=lite, 1=full, 2=heavy
    num_landmarks: int              = 33
    landmark_dimensions: int        = 3


@dataclass
class TrackerConfig:
    iou_threshold: float      = 0.3
    max_missing_frames: int   = 30
    history_size: int         = 10
    assignment_strategy: str  = "left_right"


@dataclass
class ClassifierConfig:
    window_size: int           = 15
    confidence_threshold: float = 0.6
    use_mock: bool             = True
    model_path: str            = str(MODELS_DIR / "move_classifier.pkl")
    move_classes: list         = field(default_factory=lambda: [
        "neutral", "jab", "cross", "uppercut", "front_kick", "roundhouse_kick"
    ])


@dataclass
class CommentaryConfig:
    min_time_between_comments: float = 2.0
    combo_window: float              = 3.0
    exchange_window: float           = 2.0
    enable_tts: bool                 = False
    tts_engine: str                  = "pyttsx3"
    tts_rate: int                    = 150


@dataclass
class VideoConfig:
    output_codec: str    = "mp4v"
    output_fps: int      = 30
    window_name: str     = "NeuroCombat"
    display_scale: float = 1.0
    skip_frames: int     = 0


@dataclass
class UIConfig:
    page_title: str  = "NeuroCombat — AI Fight Commentary"
    page_icon: str   = "🥊"
    layout: str      = "wide"
    primary_color: str   = "#FF6B6B"
    secondary_color: str = "#4ECDC4"
    max_upload_size_mb: int = 500
    allowed_video_formats: list = field(
        default_factory=lambda: ["mp4", "avi", "mov", "mkv"]
    )


POSE_CONFIG       = PoseConfig()
TRACKER_CONFIG    = TrackerConfig()
CLASSIFIER_CONFIG = ClassifierConfig()
COMMENTARY_CONFIG = CommentaryConfig()
VIDEO_CONFIG      = VideoConfig()
UI_CONFIG         = UIConfig()
