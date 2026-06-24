"""
predict.py
Real-time gesture recognition using MediaPipe Tasks API

Usage:
python predict.py
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    create_hand_detector,
    draw_landmarks_on_frame,
    draw_prediction_overlay,
    extract_landmarks,
    normalize_landmarks,
    FPSCounter,
    GESTURE_DISPLAY,
    GESTURE_EMOJI,
    get_logger,
)

logger = get_logger("predict")

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "gesture_model.pkl"
LABEL_PATH = MODELS_DIR / "label_encoder.pkl"


class GesturePredictor:

    def __init__(self, max_hands=2):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing model file: {MODEL_PATH}"
            )

        if not LABEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing label encoder: {LABEL_PATH}"
            )

        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

        with open(LABEL_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)

        self.detector = create_hand_detector(
            max_hands=max_hands
        )

        logger.info("Model loaded successfully")

    def process_frame(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.detector.detect(
            mp_image
        )

        predictions = []

        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:

                draw_landmarks_on_frame(
                    frame,
                    hand_landmarks
                )

                raw = extract_landmarks(
                    hand_landmarks
                )

                features = normalize_landmarks(
                    raw
                )

                probs = self.model.predict_proba(
                    features.reshape(1, -1)
                )[0]

                idx = np.argmax(probs)

                gesture = self.label_encoder.inverse_transform(
                    [idx]
                )[0]

                confidence = float(
                    probs[idx]
                )

                predictions.append({
                    "gesture": gesture,
                    "display": GESTURE_DISPLAY.get(
                        gesture,
                        gesture
                    ),
                    "emoji": GESTURE_EMOJI.get(
                        gesture,
                        ""
                    ),
                    "confidence": confidence,
                })

        return frame, predictions

    def close(self):
        self.detector.close()


def run_webcam(
    camera_id=0,
    threshold=0.60,
    max_hands=2
):

    predictor = GesturePredictor(
        max_hands=max_hands
    )

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open webcam"
        )

    fps_counter = FPSCounter()

    try:

        while True:

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)

            fps = fps_counter.tick()

            frame, predictions = predictor.process_frame(
                frame
            )

            if predictions:

                best = max(
                    predictions,
                    key=lambda x: x["confidence"]
                )

                draw_prediction_overlay(
                    frame,
                    best["display"],
                    best["confidence"],
                    fps,
                    threshold
                )

                cv2.putText(
                    frame,
                    best["display"],
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "No Hand Detected",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "Hand Gesture Recognition",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        predictor.close()

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60
    )

    parser.add_argument(
        "--max-hands",
        type=int,
        default=2
    )

    args = parser.parse_args()

    run_webcam(
        camera_id=args.camera,
        threshold=args.threshold,
        max_hands=args.max_hands
    )

    predictor = GesturePredictor(
        max_hands=max_hands
    )

    cap = cv2.VideoCapture(
        camera_id
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open webcam"
        )

    fps_counter = FPSCounter()

    try:

        while True:

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.flip(
                frame,
                1
            )

            fps = fps_counter.tick()

            frame, predictions = predictor.process_frame(
                frame
            )

            if predictions:

               best = max(
                   predictions,
                    key=lambda x: x["confidence"]
                )
               draw_prediction_overlay(
                      frame,
                      best["display"],
                      best["confidence"],
                      fps,
                      threshold
                )

               cv2.putText(
                    frame,
                    best["display"],
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "No Hand Detected",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
               )
                cv2.putText(
                    frame,
                    best["emoji"],
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 255),
                    3
                )

            cv2.imshow(
                "Hand Gesture Recognition",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        predictor.close()

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60
    )

    parser.add_argument(
        "--max-hands",
        type=int,
        default=2
    )

    args = parser.parse_args()

    run_webcam(
        camera_id=args.camera,
        threshold=args.threshold,
        max_hands=args.max_hands
    )