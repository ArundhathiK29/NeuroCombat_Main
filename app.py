"""
NeuroCombat — Cinematic Sports Broadcast UI
=============================================
Dark, premium MMA fight analysis app inspired by UFC broadcast aesthetics.
Features: side-by-side video, TTS audio commentary, live stats, and more.

Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import tempfile
import os
import time
from pathlib import Path
from typing import List, Optional

from backend.pose_extractor import PoseExtractor
from backend.tracker import PlayerTracker
from backend.move_classifier import MoveClassifier
from backend.commentary_engine import CommentaryEngine, CommentaryEvent
from backend.utils import setup_logging, format_timestamp

logger = setup_logging(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroCombat",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — cinematic matte-black / deep-crimson
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500&display=swap');

/* ── base ── */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0a0a0a;
    color: #e8e8e8;
}
.stApp { background-color: #0a0a0a; }

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* ── masthead ── */
.masthead {
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 1.2rem; margin-bottom: 2.5rem;
}
.masthead-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem; font-weight: 800; letter-spacing: 0.06em;
    text-transform: uppercase; color: #fff;
}
.masthead-title span { color: #c0392b; }
.masthead-badge {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: #888; border: 1px solid #2a2a2a; padding: 4px 10px;
    border-radius: 2px;
}

/* ── section labels ── */
.section-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.2em;
    text-transform: uppercase; color: #c0392b;
    margin-bottom: 0.6rem;
}
.section-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.5rem; font-weight: 700; letter-spacing: 0.04em;
    text-transform: uppercase; color: #fff;
    margin-bottom: 1.2rem;
}

/* ── upload zone ── */
.upload-zone {
    border: 1px dashed #2e2e2e;
    border-radius: 4px;
    padding: 3rem 2rem;
    text-align: center;
    background: #111;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #c0392b; }

/* ── stat cards ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #1a1a1a;
    border: 1px solid #1a1a1a;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.stat-card {
    background: #111;
    padding: 1.2rem 1.5rem;
}
.stat-card-label {
    font-size: 0.6rem; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; color: #555;
    margin-bottom: 0.4rem;
}
.stat-card-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem; font-weight: 700; color: #fff;
    line-height: 1;
}
.stat-card-value.accent { color: #c0392b; }

/* ── fighter comparison bars ── */
.fighter-row {
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 0.8rem;
}
.fighter-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #888; width: 60px;
}
.bar-track {
    flex: 1; height: 4px; background: #1e1e1e; border-radius: 2px;
    overflow: hidden;
}
.bar-fill-p1 { height: 100%; background: #c0392b; border-radius: 2px; }
.bar-fill-p2 { height: 100%; background: #2980b9; border-radius: 2px; }
.fighter-count {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.85rem; font-weight: 600; color: #ccc; width: 24px;
    text-align: right;
}

/* ── commentary feed ── */
.commentary-feed {
    background: #0d0d0d;
    border: 1px solid #1a1a1a;
    border-radius: 4px;
    overflow: hidden;
}
.commentary-header {
    background: #111;
    border-bottom: 1px solid #1a1a1a;
    padding: 0.8rem 1.2rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #c0392b;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
.commentary-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase; color: #888;
}
.commentary-item {
    padding: 0.85rem 1.2rem;
    border-bottom: 1px solid #151515;
    display: flex; align-items: flex-start; gap: 1rem;
    transition: background 0.15s;
}
.commentary-item:last-child { border-bottom: none; }
.commentary-item:hover { background: #131313; }
.commentary-ts {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.75rem; font-weight: 600;
    color: #c0392b; min-width: 44px; margin-top: 2px;
}
.commentary-text {
    font-size: 0.9rem; font-weight: 400;
    color: #ccc; line-height: 1.4;
}
.commentary-badge {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 2px 7px; border-radius: 2px;
    margin-left: auto; white-space: nowrap; align-self: center;
}
.badge-combo   { background: #3d1a1a; color: #e74c3c; }
.badge-exchange{ background: #1a2a3d; color: #3498db; }
.badge-move    { background: #1a1a1a; color: #666; }
.badge-opening { background: #1a2a1a; color: #27ae60; }

/* ── video panels ── */
.video-panel-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 6px 10px;
    display: inline-block;
    margin-bottom: 0.5rem;
}
.label-original  { background: #1a1a1a; color: #666; }
.label-annotated { background: #3d1a1a; color: #c0392b; }

/* ── analyse button ── */
.stButton > button {
    background: #c0392b !important;
    color: #fff !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 2rem !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #a93226 !important; }

/* ── progress bar ── */
.stProgress > div > div > div { background: #c0392b !important; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid #1a1a1a !important;
}

/* ── divider ── */
.divider { height: 1px; background: #1a1a1a; margin: 2rem 0; }

/* ── move breakdown table ── */
.move-table { width: 100%; border-collapse: collapse; }
.move-table td, .move-table th {
    padding: 0.5rem 0.8rem;
    font-size: 0.82rem;
    border-bottom: 1px solid #151515;
}
.move-table th {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: #555;
}
.move-table td { color: #aaa; }
.move-table td.move-name { color: #e8e8e8; font-weight: 500; }
.move-table td.p1-count { color: #c0392b; font-family: 'Barlow Condensed', sans-serif; font-size: 1rem; }
.move-table td.p2-count { color: #2980b9; font-family: 'Barlow Condensed', sans-serif; font-size: 1rem; }

/* ── audio section ── */
.audio-card {
    background: #111;
    border: 1px solid #1a1a1a;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "original_video": None,
        "annotated_video": None,
        "commentary_events": [],
        "fight_stats": {},
        "audio_path": None,
        "processing_complete": False,
        "uploaded_name": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# TTS — generate narrated audio from commentary events
# ─────────────────────────────────────────────────────────────────────────────
def generate_audio(events: List[CommentaryEvent], output_path: str) -> bool:
    """
    Synthesise commentary lines using gTTS and stitch them into one MP3
    with silent gaps positioned at the correct timestamps.
    Returns True on success.
    """
    try:
        from gtts import gTTS
        from pydub import AudioSegment
        import io

        if not events:
            return False

        total_duration_ms = int(events[-1].timestamp * 1000) + 5000
        track = AudioSegment.silent(duration=total_duration_ms)

        for ev in events:
            tts = gTTS(text=ev.text, lang="en", slow=False)
            mp3_buf = io.BytesIO()
            tts.write_to_fp(mp3_buf)
            mp3_buf.seek(0)
            speech = AudioSegment.from_mp3(mp3_buf)
            position_ms = int(ev.timestamp * 1000)
            track = track.overlay(speech, position=position_ms)

        track.export(output_path, format="mp3")
        return True

    except Exception as e:
        logger.warning(f"TTS generation failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    video_path: str,
    detection_conf: float,
    commentary_gap: float,
    progress_bar,
    status_text,
) -> tuple:
    """Full analysis pipeline. Returns (events, stats, annotated_path, audio_path)."""

    status_text.markdown(
        '<p style="color:#888;font-size:0.85rem;">Initialising AI components...</p>',
        unsafe_allow_html=True,
    )

    pose_extractor    = PoseExtractor(min_detection_confidence=detection_conf, model_complexity=1)
    tracker           = PlayerTracker(iou_threshold=0.3, max_missing_frames=30)
    move_classifier   = MoveClassifier(window_size=15, confidence_threshold=0.55, use_mock=True)
    commentary_engine = CommentaryEngine(min_time_between_comments=commentary_gap)

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    annotated_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

    events: List[CommentaryEvent] = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / fps
        progress_bar.progress(min((frame_idx + 1) / max(total_frames, 1), 1.0))

        poses   = pose_extractor.extract_poses_from_frame(frame, ts)
        tracked = tracker.update(poses, frame_idx)

        for pose in tracked:
            clf = move_classifier.classify_move(pose)
            if clf:
                ev = commentary_engine.generate_commentary(clf)
                if ev:
                    events.append(ev)

        annotated = pose_extractor.draw_poses_on_frame(frame, tracked)

        # Commentary overlay — bottom bar
        if events:
            h, w = annotated.shape[:2]
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, h - 70), (w, h), (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
            cv2.rectangle(annotated, (0, h - 70), (4, h), (192, 57, 43), -1)
            cv2.putText(annotated, events[-1].text, (16, h - 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
            ts_str = format_timestamp(events[-1].timestamp)
            cv2.putText(annotated, ts_str, (16, h - 50),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (192, 57, 43), 1, cv2.LINE_AA)

        out.write(annotated)
        frame_idx += 1

        if frame_idx % 30 == 0:
            pct = int(frame_idx / max(total_frames, 1) * 100)
            status_text.markdown(
                f'<p style="color:#888;font-size:0.85rem;">Analysing — {pct}% complete</p>',
                unsafe_allow_html=True,
            )

    cap.release()
    out.release()
    pose_extractor.close()

    stats = commentary_engine.get_fight_summary()

    # TTS audio
    status_text.markdown(
        '<p style="color:#888;font-size:0.85rem;">Generating audio commentary...</p>',
        unsafe_allow_html=True,
    )
    audio_path = tempfile.mktemp(suffix=".mp3")
    audio_ok = generate_audio(events, audio_path)

    status_text.markdown(
        '<p style="color:#c0392b;font-size:0.85rem;">Analysis complete.</p>',
        unsafe_allow_html=True,
    )

    return events, stats, annotated_path, (audio_path if audio_ok else None)


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_stat_cards(stats: dict, event_count: int):
    p1 = stats.get("total_moves", {}).get(1, 0)
    p2 = stats.get("total_moves", {}).get(2, 0)
    total = p1 + p2
    dominance = f"P{1 if p1 >= p2 else 2}"

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-card-label">Commentary Lines</div>
            <div class="stat-card-value accent">{event_count}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Player 1 Moves</div>
            <div class="stat-card-value">{p1}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Player 2 Moves</div>
            <div class="stat-card-value">{p2}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Dominant Fighter</div>
            <div class="stat-card-value accent">{dominance}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_fighter_bars(stats: dict):
    p1_moves = stats.get("player_1_moves", {})
    p2_moves = stats.get("player_2_moves", {})
    all_moves = ["jab", "cross", "uppercut", "front_kick", "roundhouse_kick"]
    move_labels = {
        "jab": "Jab", "cross": "Cross", "uppercut": "Uppercut",
        "front_kick": "Front Kick", "roundhouse_kick": "Roundhouse",
    }

    rows = ""
    for m in all_moves:
        c1 = p1_moves.get(m, 0)
        c2 = p2_moves.get(m, 0)
        mx = max(c1, c2, 1)
        rows += f"""
        <tr>
            <td class="move-name">{move_labels[m]}</td>
            <td class="p1-count">{c1}</td>
            <td style="width:120px">
                <div class="bar-track"><div class="bar-fill-p1" style="width:{int(c1/mx*100)}%"></div></div>
            </td>
            <td style="width:120px">
                <div class="bar-track"><div class="bar-fill-p2" style="width:{int(c2/mx*100)}%"></div></div>
            </td>
            <td class="p2-count">{c2}</td>
        </tr>"""

    st.markdown(f"""
    <table class="move-table">
        <thead>
            <tr>
                <th>Move</th>
                <th style="color:#c0392b">P1</th>
                <th colspan="2">Distribution</th>
                <th style="color:#2980b9">P2</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


def render_commentary_feed(events: List[CommentaryEvent]):
    badge_map = {
        "combo":    ("COMBO",    "badge-combo"),
        "exchange": ("EXCHANGE", "badge-exchange"),
        "opening":  ("OPENING",  "badge-opening"),
        "move":     ("MOVE",     "badge-move"),
    }

    items = ""
    for ev in reversed(events):
        ts  = format_timestamp(ev.timestamp)
        lbl, cls = badge_map.get(ev.event_type, ("—", "badge-move"))
        items += f"""
        <div class="commentary-item">
            <span class="commentary-ts">{ts}</span>
            <span class="commentary-text">{ev.text}</span>
            <span class="commentary-badge {cls}">{lbl}</span>
        </div>"""

    if not items:
        items = '<div class="commentary-item"><span class="commentary-text" style="color:#444;">No commentary generated yet.</span></div>'

    st.markdown(f"""
    <div class="commentary-feed">
        <div class="commentary-header">
            <div class="live-dot"></div>
            <span class="commentary-title">Live Commentary Feed</span>
        </div>
        {items}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("""
        <p style="font-family:'Barlow Condensed',sans-serif;font-size:0.65rem;
        letter-spacing:0.2em;text-transform:uppercase;color:#555;margin-bottom:1.5rem;">
        Analysis Settings</p>
        """, unsafe_allow_html=True)

        detection_conf = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05,
                                   help="Pose detection threshold")
        commentary_gap = st.slider("Commentary Frequency (s)", 0.5, 5.0, 1.5, 0.5,
                                   help="Min seconds between commentary lines")
        model_complexity = st.selectbox("Model Speed", ["Fast (Lite)", "Balanced", "Accurate (Heavy)"], index=1)
        complexity_map = {"Fast (Lite)": 0, "Balanced": 1, "Accurate (Heavy)": 2}

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size:0.75rem;color:#444;line-height:1.6;">
        NeuroCombat detects fighter poses using MediaPipe, classifies
        6 combat moves in real-time, and generates cinematic commentary
        with optional TTS audio.</p>
        """, unsafe_allow_html=True)

    return {
        "detection_conf": detection_conf,
        "commentary_gap": commentary_gap,
        "model_complexity": complexity_map[model_complexity],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    init_state()
    cfg = render_sidebar()

    # ── Masthead ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="masthead">
        <div class="masthead-title">NEURO<span>COMBAT</span></div>
        <div class="masthead-badge">AI Fight Analysis System</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    if not st.session_state.processing_complete:
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Upload Fight Video</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "", type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )

        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                video_path = tmp.name

            st.session_state.original_video = video_path
            st.session_state.uploaded_name  = uploaded.name

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="video-panel-label label-original">Original</div>',
                        unsafe_allow_html=True)
            st.video(video_path)

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

            col_btn, col_info = st.columns([2, 5])
            with col_btn:
                analyse = st.button("▶  Run Analysis", type="primary", use_container_width=True)
            with col_info:
                st.markdown(
                    f'<p style="color:#444;font-size:0.8rem;margin-top:0.6rem;">'
                    f'Loaded: <span style="color:#888">{uploaded.name}</span></p>',
                    unsafe_allow_html=True,
                )

            if analyse:
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                progress_bar = st.progress(0)
                status_text  = st.empty()

                try:
                    events, stats, annotated_path, audio_path = run_pipeline(
                        video_path,
                        cfg["detection_conf"],
                        cfg["commentary_gap"],
                        progress_bar,
                        status_text,
                    )
                    st.session_state.annotated_video    = annotated_path
                    st.session_state.commentary_events  = events
                    st.session_state.fight_stats        = stats
                    st.session_state.audio_path         = audio_path
                    st.session_state.processing_complete = True
                    st.rerun()

                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    logger.error(e, exc_info=True)

    # ── Results ───────────────────────────────────────────────────────────────
    else:
        events = st.session_state.commentary_events
        stats  = st.session_state.fight_stats

        # Reset button
        col_title, col_reset = st.columns([6, 1])
        with col_title:
            st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="section-title">{st.session_state.uploaded_name}</div>',
                unsafe_allow_html=True,
            )
        with col_reset:
            if st.button("↩ New Video"):
                st.session_state.processing_complete = False
                st.rerun()

        # ── Stat cards ────────────────────────────────────────────────────────
        render_stat_cards(stats, len(events))

        # ── Side-by-side videos ───────────────────────────────────────────────
        vid_col1, vid_col2 = st.columns(2)

        with vid_col1:
            st.markdown('<div class="video-panel-label label-original">Original</div>',
                        unsafe_allow_html=True)
            if st.session_state.original_video:
                st.video(st.session_state.original_video)

        with vid_col2:
            st.markdown('<div class="video-panel-label label-annotated">AI Annotated</div>',
                        unsafe_allow_html=True)
            if st.session_state.annotated_video:
                st.video(st.session_state.annotated_video)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ── Audio commentary ──────────────────────────────────────────────────
        if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Audio</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Commentary Audio</div>', unsafe_allow_html=True)

            audio_col1, audio_col2 = st.columns([3, 2])
            with audio_col1:
                st.markdown('<div class="audio-card">', unsafe_allow_html=True)
                st.markdown(
                    '<p style="font-size:0.75rem;color:#555;margin-bottom:0.8rem;">'
                    'Play inline — synced to fight timestamps</p>',
                    unsafe_allow_html=True,
                )
                st.audio(st.session_state.audio_path, format="audio/mp3")
                st.markdown('</div>', unsafe_allow_html=True)

            with audio_col2:
                st.markdown('<div class="audio-card">', unsafe_allow_html=True)
                st.markdown(
                    '<p style="font-size:0.75rem;color:#555;margin-bottom:0.8rem;">'
                    'Download MP3</p>',
                    unsafe_allow_html=True,
                )
                with open(st.session_state.audio_path, "rb") as f:
                    st.download_button(
                        label="↓  Download Commentary Audio",
                        data=f,
                        file_name="neurocombat_commentary.mp3",
                        mime="audio/mp3",
                        use_container_width=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)

        # ── Analytics ─────────────────────────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Move Breakdown</div>', unsafe_allow_html=True)

        analytics_col1, analytics_col2 = st.columns([3, 2])

        with analytics_col1:
            render_fighter_bars(stats)

        with analytics_col2:
            p1 = stats.get("total_moves", {}).get(1, 0)
            p2 = stats.get("total_moves", {}).get(2, 0)
            total = max(p1 + p2, 1)
            combos    = sum(1 for e in events if e.event_type == "combo")
            exchanges = sum(1 for e in events if e.event_type == "exchange")

            st.markdown(f"""
            <div style="display:flex;flex-direction:column;gap:1px;background:#1a1a1a;
            border:1px solid #1a1a1a;border-radius:4px;overflow:hidden;">
                <div style="background:#111;padding:1rem 1.2rem;">
                    <div class="stat-card-label">P1 Activity Share</div>
                    <div class="stat-card-value">{int(p1/total*100)}%</div>
                </div>
                <div style="background:#111;padding:1rem 1.2rem;">
                    <div class="stat-card-label">Combos Detected</div>
                    <div class="stat-card-value accent">{combos}</div>
                </div>
                <div style="background:#111;padding:1rem 1.2rem;">
                    <div class="stat-card-label">Exchanges</div>
                    <div class="stat-card-value">{exchanges}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Commentary feed ───────────────────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        render_commentary_feed(events)

        # ── Export annotated video ────────────────────────────────────────────
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        if st.session_state.annotated_video and os.path.exists(st.session_state.annotated_video):
            with open(st.session_state.annotated_video, "rb") as f:
                st.download_button(
                    label="↓  Download Annotated Video",
                    data=f,
                    file_name="neurocombat_annotated.mp4",
                    mime="video/mp4",
                )


if __name__ == "__main__":
    main()
