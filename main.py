"""
NeuroCombat — CLI Entry Point
==============================
Process an MMA fight video and generate commentary from the command line.

Usage:
    python main.py --video path/to/fight.mp4
    python main.py --video fight.mp4 --display --output results/
"""

import argparse
import cv2
import json
import sys
from pathlib import Path
from typing import Optional

from backend.pose_extractor import PoseExtractor
from backend.tracker import PlayerTracker
from backend.move_classifier import MoveClassifier
from backend.commentary_engine import CommentaryEngine
from backend.utils import setup_logging, PerformanceTimer, format_timestamp

logger = setup_logging(__name__)


class NeuroCombatPipeline:
    """Orchestrates the full pose → track → classify → commentary pipeline."""

    def __init__(
        self,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        classifier_window: int = 15,
        use_mock_classifier: bool = True,
        commentary_interval: float = 2.0,
    ):
        logger.info("Initialising NeuroCombat pipeline...")

        self.pose_extractor = PoseExtractor(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=1,
        )
        self.tracker = PlayerTracker(iou_threshold=0.3, max_missing_frames=30)
        self.move_classifier = MoveClassifier(
            window_size=classifier_window,
            confidence_threshold=0.6,
            use_mock=use_mock_classifier,
        )
        self.commentary_engine = CommentaryEngine(
            min_time_between_comments=commentary_interval,
        )
        logger.info("Pipeline ready")

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        display: bool = False,
        save_video: bool = True,
    ) -> dict:
        """
        Run the full pipeline on a video file.

        Returns a results dict with commentary events and fight stats.
        """
        vpath = Path(video_path)
        if not vpath.exists():
            raise FileNotFoundError(f"Video not found: {vpath}")

        cap = cv2.VideoCapture(str(vpath))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video: {total_frames} frames @ {fps:.1f} FPS ({width}x{height})")

        writer = None
        if save_video and output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{vpath.stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            logger.info(f"Saving annotated video to {out_path}")

        events = []
        frame_idx = 0

        with PerformanceTimer("Video processing", logger):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                ts = frame_idx / fps
                poses        = self.pose_extractor.extract_poses_from_frame(frame, ts)
                tracked      = self.tracker.update(poses, frame_idx)

                for pose in tracked:
                    classification = self.move_classifier.classify_move(pose)
                    if classification:
                        commentary = self.commentary_engine.generate_commentary(classification)
                        if commentary:
                            events.append(commentary)
                            logger.info(f"[{format_timestamp(ts)}] {commentary.text}")

                annotated = self.pose_extractor.draw_poses_on_frame(frame, tracked)
                if events:
                    self._overlay_commentary(annotated, events[-1].text)

                if display:
                    cv2.imshow("NeuroCombat", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if writer:
                    writer.write(annotated)

                frame_idx += 1
                if frame_idx % 100 == 0:
                    logger.info(f"  {frame_idx/total_frames*100:.1f}% done")

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        stats = self.commentary_engine.get_fight_summary()
        results = {
            "video_path": str(vpath),
            "total_frames": frame_idx,
            "fps": fps,
            "commentary_events": [
                {
                    "timestamp": e.timestamp,
                    "text": e.text,
                    "type": e.event_type,
                    "players": e.players_involved,
                }
                for e in events
            ],
            "fight_stats": stats,
        }

        if output_dir:
            results_path = Path(output_dir) / f"{vpath.stem}_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_path}")

        logger.info(f"Done — {len(events)} commentary events generated")
        return results

    @staticmethod
    def _overlay_commentary(frame: "np.ndarray", text: str):
        import cv2, numpy as np
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, text, (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def close(self):
        self.pose_extractor.close()


def main():
    parser = argparse.ArgumentParser(
        description="NeuroCombat — AI MMA Fight Commentary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", "-v", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--display", "-d", action="store_true", help="Show live view")
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--commentary-interval", type=float, default=2.0)
    parser.add_argument("--use-ml-model", action="store_true",
                        help="Use trained ML model instead of rule-based classifier")
    args = parser.parse_args()

    print("=" * 55)
    print("  NeuroCombat — AI Fight Commentary System")
    print("=" * 55)

    try:
        pipeline = NeuroCombatPipeline(
            detection_confidence=args.detection_confidence,
            use_mock_classifier=not args.use_ml_model,
            commentary_interval=args.commentary_interval,
        )
        results = pipeline.process_video(
            video_path=args.video,
            output_dir=args.output,
            display=args.display,
            save_video=not args.no_save_video,
        )

        print("\n--- Fight Summary ---")
        print(f"Commentary lines : {len(results['commentary_events'])}")
        print(f"Player 1 moves   : {results['fight_stats']['total_moves'].get(1, 0)}")
        print(f"Player 2 moves   : {results['fight_stats']['total_moves'].get(2, 0)}")
        pipeline.close()
        return 0

    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
