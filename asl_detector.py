"""
ASL fingerspelling detector (static letters) using MediaPipe Hands + RandomForest.

Setup:
    pip install mediapipe==0.10.14 opencv-python scikit-learn joblib numpy pandas

Usage:
    python asl_detector.py collect   # hold a sign, press its letter key to save a sample
    python asl_detector.py train     # trains a classifier on collected samples
    python asl_detector.py run       # live detection from webcam

Note: J and Z involve motion, so a single-frame classifier can't handle them.
They are skipped here (24 letters).
"""
import csv
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

DATA_FILE = Path("asl_data.csv")
MODEL_FILE = Path("asl_model.joblib")
LETTERS = [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in "jz"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_features(hand_landmarks):
    """21 landmarks -> 63 numbers, made independent of hand position and size."""
    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark])
    pts -= pts[0]                      # wrist becomes the origin
    scale = np.abs(pts).max() or 1.0
    return (pts / scale).flatten()


def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam.")
    return cap


def collect():
    counts = Counter()
    new_file = not DATA_FILE.exists()
    with DATA_FILE.open("a", newline="") as f, \
         mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["label"] + [f"f{i}" for i in range(63)])
        cap = open_camera()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hand = result.multi_hand_landmarks[0] if result.multi_hand_landmarks else None
            if hand:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, "Press a letter key to save a sample (ESC to quit)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {sum(counts.values())}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Collect", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            ch = chr(key).lower() if 0 < key < 128 else ""
            if ch in LETTERS and hand:
                writer.writerow([ch] + extract_features(hand).tolist())
                counts[ch] += 1
                print(f"{ch.upper()}: {counts[ch]}")
        cap.release()
        cv2.destroyAllWindows()


def train():
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    if not DATA_FILE.exists():
        sys.exit("No data yet. Run `collect` first.")
    df = pd.read_csv(DATA_FILE)
    X, y = df.drop(columns="label").values, df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    print(classification_report(y_te, model.predict(X_te)))
    joblib.dump(model, MODEL_FILE)
    print(f"Saved {MODEL_FILE}")


def run():
    if not MODEL_FILE.exists():
        sys.exit("No model yet. Run `train` first.")
    model = joblib.load(MODEL_FILE)
    recent = deque(maxlen=10)          # majority vote smooths out flicker
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        cap = open_camera()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            label = ""
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                feats = extract_features(hand).reshape(1, -1)
                probs = model.predict_proba(feats)[0]
                if probs.max() > 0.6:  # ignore low-confidence guesses
                    recent.append(model.classes_[probs.argmax()])
            else:
                recent.clear()
            if recent:
                label = Counter(recent).most_common(1)[0][0].upper()

            cv2.putText(frame, label, (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        2.5, (0, 255, 0), 5)
            cv2.imshow("ASL Detector (ESC to quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cmds = {"collect": collect, "train": train, "run": run}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit("Usage: python asl_detector.py [collect|train|run]")
    cmds[sys.argv[1]]()