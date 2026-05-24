"""
Pose Extractor
==============
Extracts body pose landmarks for dual fighters using MediaPipe.
Handles per-frame detection and produces annotated visualizations.
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from .utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class PoseLandmarks:
    """Pose landmarks for a single detected person."""

    landmarks: np.ndarray       # Shape: (33, 3)
    visibility: np.ndarray      # Shape: (33,)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    player_id: int              # 1 or 2 (assigned by tracker)
    timestamp: float            # Frame timestamp in seconds


class PoseExtractor:
    """
    Extracts pose landmarks from video frames using MediaPipe Pose.
    Designed for dual-fighter MMA scenarios.
    """

    # Player colour coding: Player 1 = Red, Player 2 = Blue
    PLAYER_COLORS = {1: (0, 0, 255), 2: (255, 0, 0), 0: (0, 255, 0)}

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        logger.info(f"PoseExtractor ready (model_complexity={model_complexity})")

    def extract_poses_from_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> List[PoseLandmarks]:
        """
        Detect and return pose landmarks from a single BGR frame.
        Player ID is set to 0 here — the tracker assigns 1 or 2.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb)

        if not results.pose_landmarks:
            return []

        landmarks = self._landmarks_to_array(results.pose_landmarks)
        visibility = self._extract_visibility(results.pose_landmarks)
        bbox = self._calculate_bbox(results.pose_landmarks, frame.shape)

        return [
            PoseLandmarks(
                landmarks=landmarks,
                visibility=visibility,
                bbox=bbox,
                player_id=0,
                timestamp=timestamp,
            )
        ]

    def extract_poses_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> Tuple[List[List[PoseLandmarks]], Dict]:
        """
        Extract poses from an entire video file.

        Returns:
            (pose_sequences, metadata) where pose_sequences is a list
            of per-frame pose lists and metadata contains video info.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        metadata = {
            "fps": fps,
            "total_frames": total_frames,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "video_path": video_path,
        }

        logger.info(f"Processing {total_frames} frames @ {fps:.1f} FPS")

        all_poses = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            poses = self.extract_poses_from_frame(frame, frame_idx / fps)
            all_poses.append(poses)
            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info(f"  {frame_idx}/{total_frames} frames done")

        cap.release()
        logger.info(f"Extraction complete: {frame_idx} frames")
        return all_poses, metadata

    def draw_poses_on_frame(
        self,
        frame: np.ndarray,
        poses: List[PoseLandmarks],
    ) -> np.ndarray:
        """Draw pose skeletons and bounding boxes on a copy of the frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        for pose in poses:
            color = self.PLAYER_COLORS.get(pose.player_id, (0, 255, 0))

            # Bounding box + label
            x, y, bw, bh = pose.bbox
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            label = f"Player {pose.player_id}" if pose.player_id > 0 else "Untracked"
            cv2.putText(out, label, (x, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Convert normalised landmarks → pixel coords
            px_lm = []
            for lx, ly, lz in pose.landmarks:
                px_lm.append((int(np.clip(lx * w, 0, w - 1)),
                               int(np.clip(ly * h, 0, h - 1))))

            # Skeleton connections
            for start_idx, end_idx in self.mp_pose.POSE_CONNECTIONS:
                if start_idx >= len(px_lm) or end_idx >= len(px_lm):
                    continue
                sv = pose.visibility[start_idx]
                ev = pose.visibility[end_idx]
                if sv > 0.5 and ev > 0.5:
                    cv2.line(out, px_lm[start_idx], px_lm[end_idx], color, 2,
                             cv2.LINE_AA)

            # Keypoint dots
            for i, (px, py) in enumerate(px_lm):
                if pose.visibility[i] > 0.5:
                    cv2.circle(out, (px, py), 4, color, -1)

        return out

    # ------------------------------------------------------------------ helpers

    def _landmarks_to_array(self, landmarks) -> np.ndarray:
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

    def _extract_visibility(self, landmarks) -> np.ndarray:
        return np.array([lm.visibility for lm in landmarks.landmark])

    def _calculate_bbox(
        self,
        landmarks,
        frame_shape: Tuple[int, int, int],
        padding: int = 20,
    ) -> Tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]
        x0 = max(0, int(min(xs)) - padding)
        y0 = max(0, int(min(ys)) - padding)
        x1 = min(w, int(max(xs)) + padding)
        y1 = min(h, int(max(ys)) + padding)
        return (x0, y0, x1 - x0, y1 - y0)

    def close(self):
        self.pose_detector.close()
        logger.info("PoseExtractor closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
