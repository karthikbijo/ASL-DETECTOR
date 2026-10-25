"""
ASL word detector: hello, me, you, i love you  (+ "nothing" class)
Records ~1 second clips, then classifies them with a RandomForest.

    python asl_words.py collect
    python asl_words.py train
    python asl_words.py run
"""
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

DATA_FILE = Path("words_data.npz")
MODEL_FILE = Path("words_model.joblib")
SEQ_LEN = 30          # frames per clip (~1 second)
N_FEAT = 129          # 63 right hand + 63 left hand + 3 nose
LABELS = {
    "h": "hello", "m": "me", "y": "you", "l": "i love you",
    "v": "peace", "g": "good", "o": "ok", "b": "goodbye",
    "t": "thank you", "p": "please", "x": "no",
    "n": "nothing",
}
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils


def frame_features(res):
    """Hand + nose positions relative to the body (shoulder center, shoulder width)."""
    if not res.pose_landmarks:
        return np.zeros(N_FEAT)
    pose = res.pose_landmarks.landmark
    ls, rs, nose = pose[11], pose[12], pose[0]
    origin = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2, (ls.z + rs.z) / 2])
    scale = np.hypot(ls.x - rs.x, ls.y - rs.y) or 1e-6

    def norm(p):
        return (np.array([p.x, p.y, p.z]) - origin) / scale

    def hand(h):
        if h is None:
            return np.zeros(63)
        return np.concatenate([norm(p) for p in h.landmark])

    return np.concatenate([hand(res.right_hand_landmarks),
                           hand(res.left_hand_landmarks),
                           norm(nose)])


def clip_features(seq):
    """30 frames -> one feature vector (10 sampled frames + how much each value moves)."""
    seq = np.asarray(seq)
    return np.concatenate([seq[::3].flatten(), seq.std(axis=0)])


def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam.")
    return cap


def draw(frame, res):
    mp_draw.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


def text(frame, msg, y, color=(0, 255, 0), size=0.7):
    cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)


def collect():
    X, y = [], []
    if DATA_FILE.exists():
        d = np.load(DATA_FILE, allow_pickle=True)
        X, y = list(d["X"]), list(d["y"])
    cap = open_camera()
    state, label, start, clip = "idle", None, 0.0, []
    with mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holo:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            res = holo.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw(frame, res)
            now = time.time()

            if state == "countdown" and now - start >= 1.5:
                state, clip = "recording", []
            elif state == "recording":
                clip.append(frame_features(res))
                if len(clip) == SEQ_LEN:
                    X.append(clip_features(clip))
                    y.append(label)
                    print(f"saved {label}  (total {Counter(y)[label]})")
                    state = "idle"

            if state == "idle":
                text(frame, "H=hello  M=me  Y=you  L=i love you  N=nothing  ESC=quit", 25)
            elif state == "countdown":
                text(frame, f"Get ready: {label}", 25, (0, 200, 255))
            else:
                text(frame, f"RECORDING: {label}", 25, (0, 0, 255))
            text(frame, "  ".join(f"{k}:{v}" for k, v in Counter(y).items()), 55, size=0.55)
            cv2.imshow("Collect", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            ch = chr(key).lower() if 0 < key < 128 else ""
            if state == "idle" and ch in LABELS:
                label, state, start = LABELS[ch], "countdown", now
    cap.release()
    cv2.destroyAllWindows()
    if X:
        np.savez(DATA_FILE, X=np.array(X), y=np.array(y))
        print(f"Saved {len(X)} clips to {DATA_FILE}")


def train():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    if not DATA_FILE.exists():
        sys.exit("No data yet. Run `collect` first.")
    d = np.load(DATA_FILE, allow_pickle=True)
    X, y = d["X"], d["y"]
    print(Counter(y))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    print(classification_report(y_te, model.predict(X_te)))
    model.fit(X, y)  # final model uses all data
    joblib.dump(model, MODEL_FILE)
    print(f"Saved {MODEL_FILE}")


def run():
    if not MODEL_FILE.exists():
        sys.exit("No model yet. Run `train` first.")
    model = joblib.load(MODEL_FILE)
    buf = deque(maxlen=SEQ_LEN)
    shown, shown_until, n = "", 0.0, 0
    cap = open_camera()
    with mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holo:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            res = holo.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw(frame, res)
            buf.append(frame_features(res))
            n += 1

            if len(buf) == SEQ_LEN and n % 3 == 0:
                probs = model.predict_proba([clip_features(buf)])[0]
                best = model.classes_[probs.argmax()]
                if best != "nothing" and probs.max() > 0.6:
                    shown, shown_until = best.upper(), time.time() + 1.5

            if time.time() < shown_until:
                text(frame, shown, 70, size=2, color=(0, 255, 0))
            cv2.imshow("ASL Words (ESC to quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cmds = {"collect": collect, "train": train, "run": run}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit("Usage: python asl_words.py [collect|train|run]")
    cmds[sys.argv[1]]()