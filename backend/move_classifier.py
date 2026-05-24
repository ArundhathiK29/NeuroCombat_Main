"""
Move Classifier
===============
Classifies MMA moves from pose sequences using temporal features.
Supports: jab, cross, uppercut, front_kick, roundhouse_kick, neutral.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque
from .pose_extractor import PoseLandmarks
from .utils import setup_logging

logger = setup_logging(__name__)

MOVE_CLASSES = ["neutral", "jab", "cross", "uppercut", "front_kick", "roundhouse_kick"]


@dataclass
class MoveClassification:
    """A single classified move."""

    move_name: str
    confidence: float
    player_id: int
    timestamp: float
    duration: float = 0.0


class MoveClassifier:
    """
    Classifies combat moves from a sliding window of pose frames.
    Runs a rule-based mock classifier by default; swap in a trained
    sklearn model by setting use_mock=False and supplying model_path.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        window_size: int = 15,
        confidence_threshold: float = 0.6,
        use_mock: bool = True,
    ):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.use_mock = use_mock

        self.pose_buffers: Dict[int, deque] = {
            1: deque(maxlen=window_size),
            2: deque(maxlen=window_size),
        }

        self.model = None
        if model_path and Path(model_path).exists() and not use_mock:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded ML model from {model_path}")
        else:
            logger.info("Running in rule-based mock mode")

    # ------------------------------------------------------------------ public

    def classify_move(self, pose: PoseLandmarks) -> Optional[MoveClassification]:
        """
        Classify the current move for one player using a temporal window.
        Returns None if the window isn't full yet or confidence is too low.
        """
        pid = pose.player_id
        if pid not in self.pose_buffers:
            return None

        self.pose_buffers[pid].append(pose)
        if len(self.pose_buffers[pid]) < self.window_size:
            return None

        features = self._extract_features(list(self.pose_buffers[pid]))
        if self.use_mock or self.model is None:
            move, conf = self._rule_based_classify(pose)
        else:
            move, conf = self._model_classify(features)

        if conf < self.confidence_threshold:
            return None

        return MoveClassification(
            move_name=move,
            confidence=conf,
            player_id=pid,
            timestamp=pose.timestamp,
        )

    def classify_sequence(
        self, pose_sequence: List[PoseLandmarks]
    ) -> List[MoveClassification]:
        """Classify an entire pose sequence and deduplicate results."""
        results = [self.classify_move(p) for p in pose_sequence]
        results = [r for r in results if r is not None]
        return self._deduplicate(results)

    def reset(self):
        for buf in self.pose_buffers.values():
            buf.clear()
        logger.info("MoveClassifier reset")

    # ----------------------------------------------------------------- features

    def _extract_features(self, window: List[PoseLandmarks]) -> np.ndarray:
        """Build a feature vector from velocity, angles, and spatial cues."""
        feats = []

        for i in range(1, len(window)):
            delta = window[i].landmarks - window[i - 1].landmarks
            feats += [np.mean(np.abs(delta)), np.max(np.abs(delta)), np.std(delta)]

        lm = window[-1].landmarks
        feats += [
            self._angle(lm[12], lm[14], lm[16]),  # right arm
            self._angle(lm[11], lm[13], lm[15]),  # left arm
            self._angle(lm[24], lm[26], lm[28]),  # right leg
            self._angle(lm[23], lm[25], lm[27]),  # left leg
            lm[16][1] - lm[12][1],  # right wrist height vs shoulder
            lm[15][1] - lm[11][1],  # left wrist height vs shoulder
            lm[28][1] - lm[24][1],  # right foot height vs hip
            lm[27][1] - lm[23][1],  # left foot height vs hip
        ]

        return np.array(feats)

    @staticmethod
    def _angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Angle (degrees) at p2 in the p1–p2–p3 triplet."""
        v1, v2 = p1 - p2, p3 - p2
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    # ----------------------------------------------------------- classifiers

    def _rule_based_classify(self, pose: PoseLandmarks) -> Tuple[str, float]:
        """
        Lightweight heuristic classifier for demo / prototyping.
        Checks relative joint positions to identify the six move classes.
        """
        lm = pose.landmarks
        r_wrist, l_wrist = lm[16], lm[15]
        r_shoulder, l_shoulder = lm[12], lm[11]
        r_ankle, l_ankle = lm[28], lm[27]
        r_hip, l_hip = lm[24], lm[23]

        # Kick detection — foot raised above hip baseline
        if r_hip[1] - r_ankle[1] > 0.3 or l_hip[1] - l_ankle[1] > 0.3:
            lateral = abs(r_ankle[0] - r_hip[0])
            return ("front_kick", 0.85) if lateral < 0.2 else ("roundhouse_kick", 0.80)

        # Uppercut — wrist above shoulder
        if r_shoulder[1] - r_wrist[1] > 0.2 or l_shoulder[1] - l_wrist[1] > 0.2:
            return "uppercut", 0.75

        # Jab / cross — wrist extended horizontally away from shoulder
        if abs(r_wrist[0] - r_shoulder[0]) > 0.3:
            return "cross", 0.70
        if abs(l_wrist[0] - l_shoulder[0]) > 0.3:
            return "jab", 0.70

        return "neutral", 0.95

    def _model_classify(self, features: np.ndarray) -> Tuple[str, float]:
        """Classify using a pre-trained sklearn model."""
        probs = self.model.predict_proba(features.reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        return MOVE_CLASSES[idx], float(probs[idx])

    # ---------------------------------------------------------- post-processing

    @staticmethod
    def _deduplicate(classifications: List[MoveClassification]) -> List[MoveClassification]:
        """Merge consecutive identical moves within a 1-second window."""
        if not classifications:
            return []

        out = [classifications[0]]
        for curr in classifications[1:]:
            prev = out[-1]
            if curr.move_name == prev.move_name and curr.timestamp - prev.timestamp < 1.0:
                prev.duration = curr.timestamp - prev.timestamp
            elif curr.move_name != "neutral":
                out.append(curr)
        return out
