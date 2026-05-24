"""
NeuroCombat — Streamlit Web App
================================
Upload an MMA fight video, run the analysis pipeline, and view
annotated output with live commentary and fight statistics.

Run with:
    streamlit run app.py
"""

import streamlit as st
import cv2
import tempfile
from pathlib import Path
from typing import List

from backend.pose_extractor import PoseExtractor
from backend.tracker import PlayerTracker
from backend.move_classifier import MoveClassifier
from backend.commentary_engine import CommentaryEngine, CommentaryEvent
from backend.utils import setup_logging, format_timestamp

logger = setup_logging(__name__)

st.set_page_config(
    page_title="NeuroCombat — AI Fight Commentary",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    font-size: 2.8rem; font-weight: 700; text-align: center;
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 1.5rem;
}
.commentary-box {
    background: #1E1E1E; border-left: 4px solid #FF6B6B;
    padding: 0.8rem 1rem; margin: 0.4rem 0;
    border-radius: 6px; color: #FFFFFF; font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "processed_video": None,
        "commentary_events": [],
        "fight_stats": {},
        "processing_complete": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def run_pipeline(video_path: str, progress_bar, status_text) -> tuple:
    """Run the full analysis pipeline and return (events, stats, output_path)."""
    status_text.text("Setting up AI components...")

    pose_extractor   = PoseExtractor(min_detection_confidence=0.5, model_complexity=1)
    tracker          = PlayerTracker(iou_threshold=0.3, max_missing_frames=30)
    move_classifier  = MoveClassifier(window_size=15, confidence_threshold=0.6, use_mock=True)
    commentary_engine = CommentaryEngine(min_time_between_comments=2.0)

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    events: List[CommentaryEvent] = []
    frame_idx = 0
    status_text.text("Processing video frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / fps
        progress_bar.progress((frame_idx + 1) / total_frames)

        poses   = pose_extractor.extract_poses_from_frame(frame, ts)
        tracked = tracker.update(poses, frame_idx)

        for pose in tracked:
            clf = move_classifier.classify_move(pose)
            if clf:
                ev = commentary_engine.generate_commentary(clf)
                if ev:
                    events.append(ev)

        annotated = pose_extractor.draw_poses_on_frame(frame, tracked)
        if events:
            h = annotated.shape[0]
            cv2.putText(annotated, events[-1].text, (50, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(annotated)
        frame_idx += 1
        if frame_idx % 30 == 0:
            status_text.text(f"Processing... {frame_idx}/{total_frames} frames")

    cap.release()
    out.release()
    pose_extractor.close()
    status_text.text("Done!")

    return events, commentary_engine.get_fight_summary(), output_path


def sidebar() -> dict:
    st.sidebar.markdown("## ⚙️ Settings")
    detection_conf = st.sidebar.slider("Detection confidence", 0.1, 1.0, 0.5, 0.05)
    commentary_gap = st.sidebar.slider("Min. seconds between commentary", 1.0, 5.0, 2.0, 0.5)
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**NeuroCombat** detects fighter poses, classifies moves, "
        "and generates live commentary using AI."
    )
    return {"detection_conf": detection_conf, "commentary_gap": commentary_gap}


def show_results():
    st.markdown("---")
    st.markdown("## Fight Analysis Results")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Annotated Video")
        if st.session_state.processed_video:
            st.video(st.session_state.processed_video)

    with col2:
        st.markdown("### Stats")
        stats = st.session_state.fight_stats
        st.metric("Commentary lines", stats.get("total_commentary_lines", 0))
        st.metric("Player 1 moves",   stats.get("total_moves", {}).get(1, 0))
        st.metric("Player 2 moves",   stats.get("total_moves", {}).get(2, 0))

    st.markdown("### Live Commentary")
    events = st.session_state.commentary_events
    if events:
        for ev in events:
            ts = format_timestamp(ev.timestamp)
            st.markdown(
                f'<div class="commentary-box"><strong>[{ts}]</strong> {ev.text}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No commentary generated — try lowering the detection confidence.")


def main():
    init_state()
    sidebar()

    st.markdown('<h1 class="main-header">🥊 NeuroCombat 🥊</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#888;font-size:1.1rem;'>"
        "AI-Powered Real-Time MMA Fight Commentary</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("## Upload a Fight Video")
    uploaded = st.file_uploader(
        "Choose an MMA video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name

        st.success(f"Uploaded: {uploaded.name}")
        st.markdown("### Original Video")
        st.video(video_path)

        if st.button("Analyse Fight", type="primary"):
            with st.spinner("Running analysis..."):
                bar    = st.progress(0)
                status = st.empty()
                try:
                    events, stats, out_path = run_pipeline(video_path, bar, status)
                    st.session_state.commentary_events  = events
                    st.session_state.fight_stats        = stats
                    st.session_state.processed_video    = out_path
                    st.session_state.processing_complete = True
                    st.success("Analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(e, exc_info=True)

    if st.session_state.processing_complete:
        show_results()


if __name__ == "__main__":
    main()
