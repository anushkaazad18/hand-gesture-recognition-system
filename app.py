"""
app.py — Streamlit web application for Real-Time Hand Gesture Recognition.

Run with:
    streamlit run app.py

The app renders a live webcam feed inside the browser, runs gesture inference
on each frame, and displays the result alongside a confidence meter and
project documentation.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    GESTURE_DISPLAY,
    GESTURE_EMOJI,
    GESTURE_LABELS,
    FPSCounter,
    draw_landmarks_on_frame,
    draw_prediction_overlay,
    get_logger,
)

logger = get_logger("app")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🖐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── global typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── hero banner ── */
    .hero {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e0e0ff;
        letter-spacing: -0.5px;
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: #a0a0cc;
        font-size: 1rem;
        max-width: 600px;
        margin: 0 auto;
    }

    /* ── stat cards ── */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-card {
        flex: 1;
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-card .val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #7c6aff;
    }
    .stat-card .lbl {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── gesture badge ── */
    .gesture-badge {
        background: #1a1a2e;
        border: 2px solid #7c6aff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .gesture-badge .emoji { font-size: 3rem; }
    .gesture-badge .name  {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e0e0ff;
        margin-top: 0.3rem;
    }
    .gesture-badge .conf  { color: #a0a0cc; font-size: 0.9rem; }

    /* ── section headers ── */
    .section-header {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #7c6aff;
        margin: 1.5rem 0 0.5rem;
    }

    /* ── gesture grid ── */
    .gesture-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.6rem;
        margin-bottom: 1rem;
    }
    .gesture-chip {
        background: #12122a;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 0.5rem 0.6rem;
        font-size: 0.82rem;
        color: #c0c0e0;
        text-align: center;
    }

    /* ── Streamlit override: hide default header ── */
    header[data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 1.5rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        camera_id = st.number_input("Camera index", min_value=0, max_value=4, value=0, step=1)
        threshold = st.slider(
            "Confidence threshold",
            min_value=0.30, max_value=0.99,
            value=0.60, step=0.05,
            help="Predictions below this are shown as uncertain.",
        )
        max_hands = st.radio("Max hands to track", options=[1, 2], horizontal=True)
        show_skeleton = st.toggle("Show hand skeleton", value=True)

        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown(
            "This system detects hand gestures in real time using a webcam. "
            "It combines **MediaPipe** hand-landmark extraction with a "
            "**TensorFlow/Keras** dense neural network classifier."
        )

        st.markdown("---")
        st.markdown("### 🎓 Project Info")
        st.markdown(
            "**B.Tech Final Year Project**  \n"
            "Domain: Artificial Intelligence & ML  \n"
            "Stack: Python · OpenCV · MediaPipe · TF/Keras · Streamlit"
        )

        return camera_id, threshold, max_hands, show_skeleton


# ── hero banner ───────────────────────────────────────────────────────────────

def render_hero():
    st.markdown(
        """
        <div class="hero">
            <h1>🖐 Real-Time Hand Gesture Recognition</h1>
            <p>
                MediaPipe landmark extraction · TensorFlow/Keras classifier ·
                Live webcam inference with per-frame confidence scoring
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── stat cards ────────────────────────────────────────────────────────────────

def render_stats(fps: float, n_hands: int, gesture: str, confidence: float):
    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="val">{fps:.1f}</div>
                <div class="lbl">FPS</div>
            </div>
            <div class="stat-card">
                <div class="val">{n_hands}</div>
                <div class="lbl">Hands</div>
            </div>
            <div class="stat-card">
                <div class="val">{confidence * 100:.0f}%</div>
                <div class="lbl">Confidence</div>
            </div>
            <div class="stat-card">
                <div class="val">{GESTURE_EMOJI.get(gesture, "—")}</div>
                <div class="lbl">{GESTURE_DISPLAY.get(gesture, "—")}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── gesture reference panel ───────────────────────────────────────────────────

def render_gesture_reference():
    st.markdown('<div class="section-header">Supported Gestures</div>', unsafe_allow_html=True)
    chips = "".join(
        f'<div class="gesture-chip">'
        f'{GESTURE_EMOJI.get(g, "")} {GESTURE_DISPLAY.get(g, g)}'
        f'</div>'
        for g in GESTURE_LABELS
    )
    st.markdown(f'<div class="gesture-grid">{chips}</div>', unsafe_allow_html=True)


# ── model loader (cached) ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading gesture model…")
def load_predictor(max_hands: int = 2):
    """Load GesturePredictor once and cache for the session."""
    try:
        from predict import GesturePredictor
        return GesturePredictor(max_hands=max_hands)
    except FileNotFoundError as exc:
        return str(exc)   # Return error string so we can display it


# ── main app ──────────────────────────────────────────────────────────────────

def main():
    render_hero()
    camera_id, threshold, max_hands, show_skeleton = render_sidebar()

    # ── layout ────────────────────────────────────────────────────────────────
    col_feed, col_info = st.columns([3, 1], gap="large")

    with col_feed:
        st.markdown('<div class="section-header">Live Camera Feed</div>', unsafe_allow_html=True)
        run = st.toggle("▶ Start / Stop Camera", value=False, key="run_toggle")
        frame_placeholder = st.empty()

    with col_info:
        st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)
        badge_placeholder = st.empty()
        stats_placeholder = st.empty()
        render_gesture_reference()

    if not run:
        frame_placeholder.info("Toggle **Start / Stop Camera** to begin.")
        badge_placeholder.markdown(
            '<div class="gesture-badge">'
            '<div class="emoji">🖐</div>'
            '<div class="name">Waiting…</div>'
            '<div class="conf">Start the camera to see predictions</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── load model ────────────────────────────────────────────────────────────
    predictor = load_predictor(max_hands=max_hands)
    if isinstance(predictor, str):
        st.error(f"❌ {predictor}\n\nPlease run `collect_data.py → preprocess.py → train_model.py` first.")
        return

    # ── open camera ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(int(camera_id))
    if not cap.isOpened():
        st.error(f"Could not open camera {camera_id}. Check your device index in Settings.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_counter = FPSCounter(window=20)
    current_gesture, current_conf = "", 0.0

    try:
        while st.session_state.get("run_toggle", False):
            ret, frame = cap.read()
            if not ret:
                st.warning("Empty frame received. Check your camera.")
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            fps = fps_counter.tick()

            annotated, preds = predictor.process_frame(frame)

            if preds:
                best = max(preds, key=lambda p: p["confidence"])
                current_gesture = best["gesture"]
                current_conf    = best["confidence"]

                if show_skeleton:
                    draw_prediction_overlay(
                        annotated, best["display"], current_conf, fps, threshold
                    )
                else:
                    # Still draw the text overlay without the skeleton
                    draw_prediction_overlay(
                        annotated, best["display"], current_conf, fps, threshold
                    )
            else:
                current_gesture, current_conf = "", 0.0
                cv2.putText(
                    annotated,
                    "No hand detected",
                    (16, 44),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.1,
                    (120, 120, 120),
                    2,
                    cv2.LINE_AA,
                )

            # ── push frame to Streamlit ───────────────────────────────────────
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # ── badge ─────────────────────────────────────────────────────────
            if current_gesture:
                above_thresh = current_conf >= threshold
                badge_color  = "#7c6aff" if above_thresh else "#ff9500"
                badge_placeholder.markdown(
                    f'<div class="gesture-badge" style="border-color:{badge_color};">'
                    f'<div class="emoji">{GESTURE_EMOJI.get(current_gesture, "")}</div>'
                    f'<div class="name">{GESTURE_DISPLAY.get(current_gesture, current_gesture)}</div>'
                    f'<div class="conf">{current_conf * 100:.1f}% confidence'
                    + (" ✓" if above_thresh else " ⚠ low") +
                    "</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                badge_placeholder.markdown(
                    '<div class="gesture-badge">'
                    '<div class="emoji">👀</div>'
                    '<div class="name">No hand</div>'
                    '<div class="conf">Show your hand to the camera</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # ── stats row ────────────────────────────────────────────────────
            stats_placeholder.markdown(
                f"""
                <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.5rem;">
                    <div class="stat-card" style="flex:1;min-width:60px;">
                        <div class="val" style="font-size:1.3rem">{fps:.0f}</div>
                        <div class="lbl">FPS</div>
                    </div>
                    <div class="stat-card" style="flex:1;min-width:60px;">
                        <div class="val" style="font-size:1.3rem">{len(preds)}</div>
                        <div class="lbl">Hands</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Yield control so Streamlit can process the toggle
            time.sleep(0.001)

    finally:
        cap.release()
        logger.info("Camera released.")


if __name__ == "__main__":
    main()
