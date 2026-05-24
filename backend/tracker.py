"""
Player Tracker
==============
Assigns and maintains Player 1 / Player 2 identities across frames
using bounding-box IoU matching.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import deque
from .pose_extractor import PoseLandmarks
from .utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class PlayerTrack:
    """State for a single tracked player."""

    player_id: int
    bbox_history: deque
    last_seen_frame: int
    confidence: float


class PlayerTracker:
    """
    Tracks player identities frame-to-frame via IoU-based bbox matching.
    On first detection, Player 1 is assigned to the left fighter and
    Player 2 to the right fighter.
    """

    def __init__(
        self,
        max_missing_frames: int = 30,
        iou_threshold: float = 0.3,
        history_size: int = 10,
    ):
        self.max_missing_frames = max_missing_frames
        self.iou_threshold = iou_threshold
        self.history_size = history_size

        self.tracks: Dict[int, PlayerTrack] = {}
        self.current_frame = 0

        logger.info(f"PlayerTracker ready (iou_threshold={iou_threshold})")

    def update(
        self,
        poses: List[PoseLandmarks],
        frame_idx: int,
    ) -> List[PoseLandmarks]:
        """
        Update tracker with detections from the current frame.
        Returns the same poses with player_id fields filled in.
        """
        self.current_frame = frame_idx
        self._remove_stale_tracks()

        if not self.tracks:
            return self._initialize_tracks(poses)
        return self._match_poses_to_tracks(poses)

    def _initialize_tracks(self, poses: List[PoseLandmarks]) -> List[PoseLandmarks]:
        """Assign player IDs by horizontal position (left = P1, right = P2)."""
        if not poses:
            return []

        sorted_poses = sorted(poses, key=lambda p: p.bbox[0])
        for idx, pose in enumerate(sorted_poses[:2]):
            pid = idx + 1
            pose.player_id = pid
            self.tracks[pid] = PlayerTrack(
                player_id=pid,
                bbox_history=deque([pose.bbox], maxlen=self.history_size),
                last_seen_frame=self.current_frame,
                confidence=1.0,
            )
            logger.info(f"Initialised Player {pid} at bbox {pose.bbox}")

        return sorted_poses[:2]

    def _match_poses_to_tracks(
        self,
        poses: List[PoseLandmarks],
    ) -> List[PoseLandmarks]:
        """Match detections to existing tracks using IoU."""
        matched = []
        remaining = list(poses)

        for pid, track in self.tracks.items():
            if not remaining:
                break

            last_bbox = track.bbox_history[-1]
            best_iou, best_idx = 0.0, -1

            for i, pose in enumerate(remaining):
                iou = self._iou(last_bbox, pose.bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou, best_idx = iou, i

            if best_idx >= 0:
                pose = remaining.pop(best_idx)
                pose.player_id = pid
                track.bbox_history.append(pose.bbox)
                track.last_seen_frame = self.current_frame
                track.confidence = min(1.0, track.confidence + 0.1)
                matched.append(pose)

        # Re-assign any unmatched detections to vacant player slots
        for pose in remaining:
            for pid in (1, 2):
                if pid not in self.tracks:
                    pose.player_id = pid
                    self.tracks[pid] = PlayerTrack(
                        player_id=pid,
                        bbox_history=deque([pose.bbox], maxlen=self.history_size),
                        last_seen_frame=self.current_frame,
                        confidence=0.5,
                    )
                    matched.append(pose)
                    logger.info(f"Reassigned Player {pid}")
                    break
            else:
                pose.player_id = 0  # couldn't assign

        return matched

    def _remove_stale_tracks(self):
        """Drop tracks that have been missing too long."""
        stale = [
            pid for pid, t in self.tracks.items()
            if self.current_frame - t.last_seen_frame > self.max_missing_frames
        ]
        for pid in stale:
            del self.tracks[pid]
            logger.warning(f"Lost track of Player {pid}")

    @staticmethod
    def _iou(b1: Tuple, b2: Tuple) -> float:
        """Intersection over Union for two (x, y, w, h) bounding boxes."""
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2

        ix0, iy0 = max(x1, x2), max(y1, y2)
        ix1, iy1 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0

        inter = (ix1 - ix0) * (iy1 - iy0)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union else 0.0

    def get_player_positions(self) -> Dict[int, Tuple[int, int]]:
        """Return current centre (x, y) for each tracked player."""
        return {
            pid: (t.bbox_history[-1][0] + t.bbox_history[-1][2] // 2,
                  t.bbox_history[-1][1] + t.bbox_history[-1][3] // 2)
            for pid, t in self.tracks.items()
            if t.bbox_history
        }

    def reset(self):
        self.tracks.clear()
        self.current_frame = 0
        logger.info("PlayerTracker reset")
