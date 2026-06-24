"""
utils.py
Compatible with Python 3.13 + MediaPipe 0.10.33+
"""

import logging
import time
from collections import deque
from typing import List

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# --------------------------------------------------
# Logging
# --------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------
# Constants
# --------------------------------------------------

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3

MODEL_PATH_TASK = "hand_landmarker.task"


# --------------------------------------------------
# MediaPipe Hand Detector
# --------------------------------------------------

def create_hand_detector(
    max_hands: int = 1,
    min_confidence: float = 0.7
):
    base_options = mp_python.BaseOptions(
        model_asset_path=MODEL_PATH_TASK
    )

    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=max_hands,
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return mp_vision.HandLandmarker.create_from_options(
        options
    )


# --------------------------------------------------
# Landmark Processing
# --------------------------------------------------

def extract_landmarks(hand_landmarks) -> np.ndarray:
    """
    Convert 21 landmarks to flat 63-feature vector.
    """

    coords = []

    for lm in hand_landmarks:
        coords.extend([
            lm.x,
            lm.y,
            lm.z
        ])

    return np.array(coords, dtype=np.float32)


def normalize_landmarks(
    landmarks: np.ndarray
) -> np.ndarray:

    coords = landmarks.reshape(NUM_LANDMARKS, 3)

    coords = coords - coords[0]

    max_val = np.max(np.abs(coords))

    if max_val > 0:
        coords = coords / max_val

    return coords.flatten().astype(np.float32)


# --------------------------------------------------
# Drawing Utilities
# --------------------------------------------------

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]


def draw_landmarks_on_frame(
    frame,
    hand_landmarks,
    label=""
):
    h, w = frame.shape[:2]

    points = []

    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    for a, b in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[a],
            points[b],
            (0, 255, 0),
            2
        )

    for i, (x, y) in enumerate(points):

        color = (0, 255, 255) if i == 0 else (0, 255, 0)

        cv2.circle(
            frame,
            (x, y),
            5,
            color,
            -1
        )

    if label:
        cv2.putText(
            frame,
            label,
            (points[0][0], points[0][1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    return frame


# --------------------------------------------------
# Overlay
# --------------------------------------------------

def draw_prediction_overlay(
    frame,
    gesture,
    confidence,
    fps,
    threshold=0.6
):

    h, w = frame.shape[:2]

    color = (
        (0, 255, 0)
        if confidence >= threshold
        else (0, 165, 255)
    )

    cv2.rectangle(
        frame,
        (0, 0),
        (w, 70),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 140, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return frame


# --------------------------------------------------
# FPS Counter
# --------------------------------------------------

class FPSCounter:

    def __init__(self, window=30):

        self._times = deque(maxlen=window)
        self._prev = time.perf_counter()

    def tick(self):

        now = time.perf_counter()

        self._times.append(
            now - self._prev
        )

        self._prev = now

        if len(self._times) < 2:
            return 0.0

        avg = sum(self._times) / len(self._times)

        return 1.0 / avg


# --------------------------------------------------
# Gesture Labels
# --------------------------------------------------

# --------------------------------------------------
# Gesture Labels
# --------------------------------------------------

GESTURE_LABELS = [
    "open_palm",
    "fist",
    "thumbs_up",
    "thumbs_down",
    "victory",
    "okay",
]

GESTURE_DISPLAY = {
    "open_palm": "Open Palm",
    "fist": "Fist",
    "thumbs_up": "Thumbs Up",
    "thumbs_down": "Thumbs Down",
    "victory": "Victory",
    "okay": "OK Sign",
}

GESTURE_EMOJI = {
    "open_palm": "🖐",
    "fist": "✊",
    "thumbs_up": "👍",
    "thumbs_down": "👎",
    "victory": "✌️",
    "okay": "👌",
}