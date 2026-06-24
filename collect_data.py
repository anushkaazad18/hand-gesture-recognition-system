"""
collect_data.py
Dataset collection using MediaPipe Tasks API

Usage:
python collect_data.py --gesture open_palm --samples 300
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    GESTURE_LABELS,
    MODEL_PATH_TASK,
    create_hand_detector,
    draw_landmarks_on_frame,
    extract_landmarks,
    normalize_landmarks,
    get_logger,
)

logger = get_logger("collect_data")

DATASET_DIR = Path("dataset")
CSV_PATH = DATASET_DIR / "landmarks.csv"

HEADER = []

for i in range(21):
    HEADER.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

HEADER.append("label")


def ensure_csv_exists():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)


def count_samples(label):
    if not CSV_PATH.exists():
        return 0

    count = 0

    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["label"] == label:
                count += 1

    return count


def save_sample(features, label):

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        row = list(features)
        row.append(label)

        writer.writerow(row)


def collect(gesture, target_samples, camera_id):

    if gesture not in GESTURE_LABELS:
        raise ValueError(
            f"Gesture must be one of {GESTURE_LABELS}"
        )

    if not Path(MODEL_PATH_TASK).exists():
        raise FileNotFoundError(
            f"{MODEL_PATH_TASK} not found"
        )

    ensure_csv_exists()

    existing = count_samples(gesture)

    logger.info(
        f"Existing samples: {existing}"
    )

    detector = create_hand_detector(
        max_hands=1
    )

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open webcam"
        )

    collected = 0

    last_save = 0

    SAVE_DELAY = 0.25

    try:

        while True:

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(
                mp_image
            )

            detected = False

            if result.hand_landmarks:

                detected = True

                hand_landmarks = result.hand_landmarks[0]

                draw_landmarks_on_frame(
                    frame,
                    hand_landmarks
                )

            total = existing + collected

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {total}/{target_samples}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            if detected:
                cv2.putText(
                    frame,
                    "Press SPACE to save",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "Gesture Data Collection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            now = time.perf_counter()

            if (
                key == ord(" ")
                and detected
                and now - last_save > SAVE_DELAY
            ):

                raw = extract_landmarks(
                    hand_landmarks
                )

                features = normalize_landmarks(
                    raw
                )

                save_sample(
                    features,
                    gesture
                )

                collected += 1

                last_save = now

                logger.info(
                    f"Saved sample {existing + collected}"
                )

                if (
                    existing + collected
                    >= target_samples
                ):
                    logger.info(
                        "Target reached"
                    )
                    break

            elif key == ord("q"):
                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        detector.close()


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gesture",
        required=True,
        choices=GESTURE_LABELS
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=300
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    collect(
        gesture=args.gesture,
        target_samples=args.samples,
        camera_id=args.camera
    )