import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import io
import time
import json
import os
import hashlib
import threading
import collections
import math
import html as html_lib
import re
import streamlit as st
import streamlit.components.v1 as components

# Real-time camera preview (optional). Keep this import ABOVE cv2, TensorFlow and DeepFace:
# PyAV and OpenCV each bundle their own video libraries, and loading PyAV after them can crash
# the whole process with a segmentation fault.
try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC_OK = True
except Exception:
    WEBRTC_OK = False

from PIL import Image, ImageOps
import numpy as np
import cv2
import pandas as pd
from deepface import DeepFace  # used ONLY to detect faces

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Facial Emotion Intelligence",
    page_icon="🫧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EMOJI_MAP = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😔",
    "surprise": "😲",
    "neutral": "😐",
}

# How "pleasant" each emotion is. Used for the mood trend line (-1 = unpleasant, +1 = pleasant).
VALENCE = {
    "happy": 1.0,
    "surprise": 0.25,
    "neutral": 0.0,
    "sad": -0.8,
    "fear": -0.7,
    "angry": -0.8,
    "disgust": -0.7,
}

MAX_PHOTOS = 5  # max photos for multi-photo averaging

EMOTION_PRESETS = {
    "happy": [
        "🎓 Cleared a Major Subject / Goal",
        "🎉 Good News or Progress Milestone",
        "🌿 In a calm, peaceful flow state",
    ],
    "sad": [
        "📚 Academic / Syllabus Overload",
        "🌧️ Feeling Mentally Drained",
        "💔 Personal or Internal Pressure",
    ],
    "angry": [
        "🛑 Frustrating Blocker or Inefficiency",
        "😤 Interpersonal Conflict",
        "⏳ Overwhelmed by Constraints",
    ],
    "fear": [
        "😰 Impending Deadlines or Exams",
        "⚡ Uncertainty About Next Steps",
        "🫣 Sensory / Work Overload",
    ],
    "disgust": [
        "🤢 Unfair or Chaotic Atmosphere",
        "😾 Mental Resistance to a Task",
        "🥗 Physically Uncomfortable Setting",
    ],
    "surprise": [
        "🎁 Unexpected Pivot or News",
        "🤯 Sudden Realization or Result",
        "⚡ Disrupted Routine / Schedule",
    ],
    "neutral": [
        "🧘 Mindful and Grounded Right Now",
        "💻 Locked in Deep Focus",
        "😐 Navigating a Steady Day",
    ],
}

if "history" not in st.session_state:
    st.session_state.history = []

if "cached_analysis" not in st.session_state:
    st.session_state.cached_analysis = None

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

if "active_mode" not in st.session_state:
    st.session_state.active_mode = "upload"

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "last_emotion_seen" not in st.session_state:
    st.session_state.last_emotion_seen = None

if "trigger_celebration" not in st.session_state:
    st.session_state.trigger_celebration = False

if st.session_state.trigger_celebration:
    st.balloons()
    st.session_state.trigger_celebration = False


# =============================================================================
# EMOTION MODELS (HSEmotion ensemble, trained on AffectNet)
# =============================================================================

# Output order of the 8-class HSEmotion models
HSE_LABELS = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise",
]

# Map to the 7 labels this app uses ("contempt" is dropped, then rescaled)
HSE_TO_APP = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}

# The first two are fast EfficientNet-B0 models (also used for the live preview).
# enet_b2_8 is a larger model added for the full analysis; it is skipped if it cannot be loaded.
HSE_MODEL_NAMES = ["enet_b0_8_best_vgaf", "enet_b0_8_best_afew", "enet_b2_8"]
CROP_PADDINGS = (0.08, 0.16, 0.24)  # three face-crop margins; results are averaged


@st.cache_resource(show_spinner="Loading emotion models...")
def load_hse_models():
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    loaded = []
    for name in HSE_MODEL_NAMES:
        try:
            loaded.append(HSEmotionRecognizer(model_name=name))
        except Exception:
            continue  # an unavailable optional model must never stop the app
    if not loaded:
        raise RuntimeError("No emotion model could be loaded. Check the hsemotion-onnx installation.")
    return loaded


load_hse_models()


def ensemble_probs(models, crops_rgb: list, mirror: bool = True) -> np.ndarray:
    """Average probabilities over models x crops (x original + mirrored when mirror=True).
    Takes the models as an argument so the live video thread never touches Streamlit."""
    collected = []
    for crop in crops_rgb:
        views = (crop, crop[:, ::-1]) if mirror else (crop,)
        for view in views:
            view = np.ascontiguousarray(view)
            for m in models:
                _, p = m.predict_emotions(view, logits=False)
                p = np.asarray(p, dtype=np.float64).flatten()
                if p.shape[0] == len(HSE_LABELS):
                    collected.append(p)
    if not collected:
        raise ValueError("no predictions")
    return np.mean(collected, axis=0)


def predict_emotion_probs(crops_rgb: list) -> np.ndarray:
    """Full-quality prediction used for uploaded and captured photos."""
    return ensemble_probs(load_hse_models(), crops_rgb, mirror=True)


def probs_to_scores(probs: np.ndarray) -> dict:
    """8 HSEmotion probabilities -> the app's 7 emotions as percentages (contempt dropped, rescaled)."""
    raw = dict(zip(HSE_LABELS, np.asarray(probs, dtype=np.float64).tolist()))
    emo = {HSE_TO_APP[k]: float(raw[k]) for k in HSE_TO_APP}
    total = sum(emo.values()) or 1.0
    return {k: v / total * 100 for k, v in emo.items()}


def enhance_webcam_frame(rgb_image: np.ndarray) -> np.ndarray:
    """Only brightens dark webcam frames. Well-lit photos stay untouched."""
    brightness = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).mean()
    if brightness >= 90:
        return rgb_image
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


@st.cache_data(show_spinner=False, max_entries=32)
def run_deepface_inference(image_bytes: bytes, width: int, height: int):
    """Detect faces with DeepFace, classify emotion with the HSEmotion ensemble.
    Returns (results, latency_ms, error). Each result has 'region' and 'emotion' (percentages)."""
    rgb_arr = np.frombuffer(image_bytes, np.uint8).reshape((height, width, 3))
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    t0 = time.time()
    faces = []

    # 1) Find faces
    for backend in ["yunet", "ssd", "opencv"]:
        try:
            found = DeepFace.extract_faces(
                img_path=bgr_arr,
                detector_backend=backend,
                enforce_detection=True,
                align=False,
            )
            found = [
                f for f in found
                if f["facial_area"]["w"] >= 40 and f["facial_area"]["h"] >= 40
                and not (f["facial_area"]["w"] >= width * 0.95
                         and f["facial_area"]["h"] >= height * 0.95)
            ]
            if found:
                faces = found
                break
        except Exception:
            continue

    if not faces:
        return [], (time.time() - t0) * 1000, \
            "No clear face found. Face the camera, good light, no heavy shadows."

    # 2) Classify each face (ensemble + test-time augmentation)
    results = []
    for f in faces:
        a = f["facial_area"]
        x, y, w, h = int(a["x"]), int(a["y"]), int(a["w"]), int(a["h"])

        crops = []
        for pad_frac in CROP_PADDINGS:
            pad = int(pad_frac * max(w, h))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
            c = rgb_arr[y1:y2, x1:x2]
            if c.size:
                crops.append(c)
        if not crops:
            continue

        try:
            probs = predict_emotion_probs(crops)
        except Exception:
            continue

        emo = probs_to_scores(probs)

        results.append({
            "region": {"x": x, "y": y, "w": w, "h": h},
            "emotion": emo,
        })

    latency_ms = (time.time() - t0) * 1000
    if results:
        return results, latency_ms, None
    return [], latency_ms, "Face found but could not be read. Retake the photo."


def assess_quality(rgb: np.ndarray, region: dict) -> list:
    """Heuristic checks: bad input photos are a top cause of wrong emotions."""
    x, y = max(0, int(region["x"])), max(0, int(region["y"]))
    w, h = int(region["w"]), int(region["h"])
    face = rgb[y:y + h, x:x + w]
    notes = []
    if face.size == 0:
        return notes

    if min(w, h) < 90:
        notes.append("📏 Face is small in the frame. Move closer for a more reliable reading.")

    gray = cv2.cvtColor(cv2.resize(face, (128, 128)), cv2.COLOR_RGB2GRAY)

    if cv2.Laplacian(gray, cv2.CV_64F).var() < 25:
        notes.append("🔍 Photo looks blurry. Hold still or use a sharper image.")

    mean_b = float(gray.mean())
    if mean_b < 70:
        notes.append("🌑 Face is too dark. Face a light source.")
    elif mean_b > 200:
        notes.append("☀️ Face is overexposed. Reduce direct glare.")

    if abs(float(gray[:, :64].mean()) - float(gray[:, 64:].mean())) > 45:
        notes.append("💡 Uneven lighting (one side of the face is in shadow). Shadows are often misread as sad or angry.")

    return notes


# =============================================================================
# IMAGE LOADING, FACE SCORING & CLASSIFICATION HELPERS
# =============================================================================

def load_rgb(file_obj, is_camera: bool) -> np.ndarray:
    """Open an uploaded / camera file as a resized RGB array (webcam shots are mirrored + brightened if dark)."""
    img = Image.open(io.BytesIO(file_obj.getvalue()))
    img = ImageOps.exif_transpose(img)

    max_dim = 1280
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.Resampling.BILINEAR,
        )

    arr = np.array(img.convert("RGB"))
    if is_camera:
        arr = cv2.flip(arr, 1)
        arr = enhance_webcam_frame(arr)
    return arr


def detect_and_score(image_np: np.ndarray) -> dict:
    """Run detection + emotion scoring for ALL faces in an image.
    'boxes' = real face boxes (left to right). 'faces' = boxes, or the single fallback result."""
    H, W = image_np.shape[:2]
    image_bytes = image_np.tobytes()
    results, latency, err = run_deepface_inference(image_bytes, W, H)

    boxes = []
    for r in results:
        reg = r.get("region", {})
        rx, ry, rw, rh = int(reg.get("x", 0)), int(reg.get("y", 0)), int(reg.get("w", 0)), int(reg.get("h", 0))
        is_fallback = (rx == 0 and ry == 0 and rw >= W * 0.95 and rh >= H * 0.95)
        if not is_fallback and rw >= 30 and rh >= 30:
            boxes.append(r)
    boxes.sort(key=lambda r: r["region"]["x"])  # number faces left to right

    faces = boxes if boxes else results[:1]
    return {
        "image": image_np, "bytes": image_bytes, "W": W, "H": H,
        "results": results, "boxes": boxes, "faces": faces,
        "latency": latency, "err": err,
    }


def face_area(face: dict) -> int:
    return int(face["region"]["w"]) * int(face["region"]["h"])


def classify_emotions(emo: dict) -> dict:
    """Turn raw emotion percentages into a label + flags (no artificial score multipliers)."""
    total = sum(emo.values()) or 1.0
    scores = {k: float(v) / total * 100 for k, v in emo.items()}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    dom, dom_conf = ranked[0][0].lower(), float(ranked[0][1])
    sec, sec_conf = ranked[1][0].lower(), float(ranked[1][1])
    margin = dom_conf - sec_conf

    # Gate out ambiguous / guessing predictions
    ambiguous = dom_conf < 35 or margin < 10
    # Two emotions nearly tied AND together dominant -> show them as a blend
    mixed = margin < 10 and (dom_conf + sec_conf) >= 70

    if mixed:
        title = f"{dom.capitalize()} + {sec.capitalize()}"
        emoji = EMOJI_MAP.get(dom, "") + EMOJI_MAP.get(sec, "")
        short = f"{dom.capitalize()}+{sec.capitalize()}"
    elif ambiguous:
        title = "Ambiguous / Inconclusive"
        emoji = "🤔"
        short = "Unclear"
    else:
        title = dom.capitalize()
        emoji = EMOJI_MAP.get(dom, "•")
        short = f"{dom.capitalize()} {dom_conf:.0f}%"

    mood = sum(scores.get(k, 0.0) / 100.0 * VALENCE.get(k, 0.0) for k in scores) * 100.0

    return {
        "scores": scores, "dominant": dom, "conf": dom_conf,
        "second": sec, "second_conf": sec_conf, "margin": margin,
        "ambiguous": ambiguous, "mixed": mixed,
        "title": title, "emoji": emoji, "short": short, "mood": mood,
    }


# =============================================================================
# LIVE (REAL-TIME) EMOTION PREVIEW
# =============================================================================

LIVE_MIN_INTERVAL = 0.02   # the models never run more often than this (seconds); they run as fast as they can
LIVE_MAX_FACES = 4         # label at most this many faces in the live view
LIVE_TAU = 0.45            # smoothing time constant in seconds (higher = steadier but slower to react)
LIVE_TAU_NEW = 0.12        # much faster smoothing during the first second of a new face, so it shows up at once
LIVE_DETECT_WIDTH = 960    # frames are shrunk to this width for face detection only (wider = sees smaller, farther faces)
LIVE_MIN_FACE = 32         # smallest face (pixels, in the detection frame) that is picked up
LIVE_BURST = 5             # sharpest recent frames averaged when you press Capture
LIVE_BUFFER = 12           # recent frames kept for the burst
LIVE_BUFFER_STEP = 0.12    # seconds between kept frames


def _iou(a, b) -> float:
    """Overlap of two (x, y, w, h) boxes, 0 to 1."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class SnapshotFile:
    """File-like wrapper so a live capture goes through the same pipeline as an upload."""

    def __init__(self, data: bytes, name: str = "live_capture.png"):
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data


class LiveState:
    """Shared by the video thread (writes) and the page (reads).
    The video callback only stores the newest frame and draws the latest result, so the video stays
    smooth. A worker thread tracks faces and runs the models as fast as it can, smoothing each
    face's reading over time. It also keeps a short buffer of recent frames for the burst capture."""

    def __init__(self, models):
        self.models = list(models)[:2]   # fast pair for the live preview; the full analysis uses all models
        self.lock = threading.Lock()
        self.raw = None                  # newest frame exactly as the camera sent it (RGB, not mirrored)
        self.mirrored = None             # same frame, mirrored like the on-screen preview
        self.frame_time = 0.0
        self.buffer = collections.deque(maxlen=LIVE_BUFFER)   # (sharpness, raw frame)
        self.last_buffered = 0.0
        self.faces, self.labels, self.main = [], [], 0
        self.scores = None               # smoothed percentages of the largest face
        self.tracks = []                 # worker thread only
        self.last_update = 0.0           # worker thread only
        self.worker = None
        self.profile = None              # side-view face detector (OpenCV 4 Haar cascade), optional
        try:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
            if not cascade.empty():
                self.profile = cascade
        except Exception:
            self.profile = None

    def push_frame(self, raw, mirrored):
        now = time.time()
        sharp = None
        if now - self.last_buffered >= LIVE_BUFFER_STEP:
            tiny = cv2.resize(mirrored, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            sharp = float(cv2.Laplacian(cv2.cvtColor(tiny, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
        with self.lock:
            self.raw, self.mirrored, self.frame_time = raw, mirrored, now
            if sharp is not None:
                self.buffer.append((sharp, raw))
                self.last_buffered = now
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(target=self._loop, daemon=True)
            self.worker.start()

    def _loop(self):
        while True:
            with self.lock:
                frame, ts = self.mirrored, self.frame_time
            if frame is None or time.time() - ts > 5:  # camera stopped: let the thread end
                break
            t0 = time.time()
            try:
                self.update(frame)
            except Exception:
                pass
            time.sleep(max(0.0, LIVE_MIN_INTERVAL - (time.time() - t0)))

    @staticmethod
    def detect(img) -> list:
        """Face boxes (x, y, w, h) with the same detectors as the full analysis (fast ones only)."""
        h, w = img.shape[:2]
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for backend in ("yunet", "opencv"):
            try:
                found = DeepFace.extract_faces(
                    img_path=bgr, detector_backend=backend, enforce_detection=True, align=False
                )
            except Exception:
                continue
            boxes = []
            for f in found:
                a = f["facial_area"]
                fx, fy, fw, fh = int(a["x"]), int(a["y"]), int(a["w"]), int(a["h"])
                if fw < LIVE_MIN_FACE or fh < LIVE_MIN_FACE:
                    continue
                if fw >= w * 0.95 and fh >= h * 0.95:  # "whole image" fallback, not a real face
                    continue
                boxes.append((fx, fy, fw, fh))
            if boxes:
                return boxes
        return []

    def detect_side(self, small, exclude) -> list:
        """Faces turned away from the camera (side view). Rough: it only shows that a face is there,
        and is never used to read an emotion."""
        if self.profile is None:
            return []
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
        width = gray.shape[1]
        found_boxes = []
        for flipped in (False, True):  # the cascade finds faces looking one way; mirror to find the other
            g = cv2.flip(gray, 1) if flipped else gray
            for (x, y, bw, bh) in self.profile.detectMultiScale(
                g, scaleFactor=1.1, minNeighbors=6, minSize=(LIVE_MIN_FACE, LIVE_MIN_FACE)
            ):
                if flipped:
                    x = width - x - bw
                found_boxes.append((int(x), int(y), int(bw), int(bh)))
        return [bx for bx in found_boxes if all(_iou(bx, e) < 0.2 for e in exclude)]

    def update(self, img):
        now = time.time()
        img = enhance_webcam_frame(img)  # only brightens dark frames; well-lit frames are untouched
        h, w = img.shape[:2]
        scale = min(1.0, LIVE_DETECT_WIDTH / float(w))
        small = img if scale == 1.0 else cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        front_small = self.detect(small)
        side_small = self.detect_side(small, front_small)

        def up(bx):
            return (int(bx[0] / scale), int(bx[1] / scale), int(bx[2] / scale), int(bx[3] / scale))

        boxes = sorted((up(bx) for bx in front_small), key=lambda bx: bx[2] * bx[3], reverse=True)
        boxes = boxes[:LIVE_MAX_FACES]
        side_boxes = [up(bx) for bx in side_small][:LIVE_MAX_FACES]

        dt = min(max(now - self.last_update, 0.02), 1.0) if self.last_update else 0.2
        self.last_update = now
        a_box = 1.0 - math.exp(-dt / 0.15)

        free = list(self.tracks)

        def match(bx):
            best, best_iou = None, 0.3
            for tr in free:
                v = _iou(tr["box"], bx)
                if v > best_iou:
                    best, best_iou = tr, v
            return best

        for box in boxes:
            best = match(box)
            x, y, bw, bh = box
            pad = int(0.12 * max(bw, bh))
            crop = img[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)]
            if crop.size == 0:
                continue
            try:
                scores = probs_to_scores(ensemble_probs(self.models, [crop], mirror=True))
            except Exception:
                continue
            if best is None:
                # a new face is shown from its very first reading, with no smoothing delay
                self.tracks.append(
                    {"box": [float(v) for v in box], "scores": scores, "seen": now, "born": now, "front": now, "hits": 1}
                )
            else:
                free.remove(best)
                best["box"] = [(1 - a_box) * o + a_box * n for o, n in zip(best["box"], box)]
                if best["scores"] is None:  # a side-view face has turned towards the camera: read it at once
                    best["scores"], best["born"] = scores, now
                else:
                    # time-based smoothing (independent of frame rate); quicker while the face is new
                    tau = LIVE_TAU_NEW if now - best["born"] < 1.0 else LIVE_TAU
                    a_score = 1.0 - math.exp(-dt / tau)
                    best["scores"] = {
                        k: (1 - a_score) * best["scores"].get(k, scores[k]) + a_score * scores[k] for k in scores
                    }
                best["seen"] = best["front"] = now
                best["hits"] += 1

        for box in side_boxes:
            best = match(box)
            if best is None:
                self.tracks.append(
                    {"box": [float(v) for v in box], "scores": None, "seen": now, "born": now, "front": 0.0, "hits": 1}
                )
            else:
                free.remove(best)
                best["box"] = [(1 - a_box) * o + a_box * n for o, n in zip(best["box"], box)]
                best["seen"] = now
                best["hits"] += 1

        self.tracks = [t for t in self.tracks if now - t["seen"] <= 1.0]
        # a side-only face must be seen twice before it is shown (filters one-off false alarms);
        # a short grace period avoids flicker when a detection drops for a moment
        live = [
            t for t in self.tracks
            if now - t["seen"] <= 0.6 and (t["scores"] is not None or t["hits"] >= 2)
        ]
        live.sort(key=lambda t: t["box"][0])  # left to right, like the full analysis
        readable = [i for i, t in enumerate(live) if t["scores"] is not None and now - t["front"] <= 0.5]
        faces = [
            {"region": {"x": int(t["box"][0]), "y": int(t["box"][1]), "w": int(t["box"][2]), "h": int(t["box"][3])}}
            for t in live
        ]
        labels = [
            classify_emotions(t["scores"])["short"] if i in readable else "Side view"
            for i, t in enumerate(live)
        ]
        main = max(readable, key=lambda i: live[i]["box"][2] * live[i]["box"][3]) if readable else 0
        with self.lock:
            self.faces, self.labels, self.main = faces, labels, main
            self.scores = dict(live[main]["scores"]) if readable else None

    def read(self) -> dict:
        with self.lock:
            return {
                "faces": list(self.faces),
                "labels": list(self.labels),
                "main": self.main,
                "scores": dict(self.scores) if self.scores else None,
            }

    def take_best(self, k: int) -> list:
        """The k sharpest of the recent frames, oldest first (copies, exactly as the camera sent them)."""
        with self.lock:
            items = list(self.buffer)
            if not items and self.raw is not None:
                items = [(0.0, self.raw)]
        if not items:
            return []
        top = sorted(range(len(items)), key=lambda i: items[i][0], reverse=True)[:k]
        return [items[i][1].copy() for i in sorted(top)]


def make_live_callback(state: LiveState):
    """Video callback: mirror the frame, draw the latest face boxes and labels, hand it back."""
    def callback(frame):
        raw = frame.to_ndarray(format="rgb24")
        mirrored = cv2.flip(raw, 1)
        state.push_frame(raw, mirrored)
        data = state.read()
        out = mirrored
        if data["faces"]:
            out = draw_clean_bounding(mirrored, data["faces"], True, data["labels"], data["main"])
        return av.VideoFrame.from_ndarray(out, format="rgb24")
    return callback


def _live_readout(state: LiveState, active: bool):
    """Live emotion percentages next to the video. Re-runs about twice a second while streaming."""
    if not active:
        st.markdown(
            '<div class="live-empty">Press START under the video to turn the camera on.</div>',
            unsafe_allow_html=True,
        )
        return
    data = state.read()
    scores = data["scores"]
    if not scores:
        msg = (
            "Face seen, but turned away. Look towards the camera for an emotion reading."
            if data["faces"]
            else "Looking for a face... face the camera in good light."
        )
        st.markdown(f'<div class="live-empty">{msg}</div>', unsafe_allow_html=True)
        return
    cls = classify_emotions(scores)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    rows = "".join(
        f'<div class="live-row"><div class="live-lab">{EMOJI_MAP.get(k, "")} {k.capitalize()}</div>'
        f'<div class="live-track"><div class="live-fill" style="width:{max(v, 1.5):.1f}%"></div></div>'
        f'<div class="live-val">{v:.1f}%</div></div>'
        for k, v in ranked
    )
    extra = ""
    if len(data["faces"]) > 1:
        extra = f'<div class="live-note">{len(data["faces"])} faces in view. The bars show the largest face.</div>'
    st.markdown(
        f'<div class="live-top"><span class="raw-emoji">{cls["emoji"]}</span>{cls["title"]} '
        f'<small>({cls["conf"]:.0f}%)</small></div>{rows}{extra}',
        unsafe_allow_html=True,
    )


if hasattr(st, "fragment"):  # Streamlit 1.37+: refresh only this panel, not the whole page
    live_readout_running = st.fragment(run_every=0.25)(_live_readout)
    live_readout_static = st.fragment(_live_readout)
else:
    live_readout_running = live_readout_static = _live_readout


# =============================================================================
# GROUNDED MENTAL WELLNESS COACH ENGINE
# =============================================================================

# --- Safety: crisis language is handled by a fixed, caring message (no AI guesswork) ---

CRISIS_REGEX = re.compile(
    r"(suicid\w*|kill(ing)? myself|end(ing)? my life|end it all|want(ed)? to die|wanna die|"
    r"don'?t want to (live|be here|exist)|no reason to live|better off (dead|without me)|"
    r"(hurt|harm|cut|cutting|hurting|harming) myself|self[- ]?harm|take my own life)",
    re.IGNORECASE,
)

CRISIS_REPLY = (
    "💙 I'm really glad you told me, and I'm taking it seriously. You don't have to carry this alone, "
    "and you deserve support from a real person right now.\n"
    "If you might act on these thoughts, please call your local emergency number (112 in India) or go to "
    "the nearest hospital, and try not to be alone. Reach out to a friend, a family member or someone you trust.\n"
    "In India, Tele-MANAS is free, confidential and open 24x7: call 14416 or 1-800-891-4416. "
    "Elsewhere, findahelpline.com lists free helplines by country.\n"
    "I'm an app, not a counsellor, but I'm here to keep talking with you too."
)

COACH_NOTE = (
    "Reflective check-in, not therapy or medical advice. Camera emotion reads are rough guesses, "
    "so your own words matter more. Photos are processed in memory and not saved by this app. "
    "If a Gemini key is set, only your typed messages (never your photo) are sent to Google Gemini. "
    "If things feel too heavy: Tele-MANAS (India) 14416, free and 24x7, or findahelpline.com for other countries."
)


def is_crisis_message(text: str) -> bool:
    return bool(CRISIS_REGEX.search(text or ""))


def render_chat_html(text: str) -> str:
    """Escape user / AI text before it goes into HTML, keep line breaks and **bold**."""
    safe = html_lib.escape(str(text))
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    return safe.replace("\n", "<br>")


# --- Quick resets shown in the "1-minute reset" drawer ---

EMOTION_RESETS = {
    "happy": ("✨ Lock in the win", [
        "Name one thing that went well today.",
        "Tell one person about it, since shared joy lasts longer.",
        "Note what helped, so you can repeat it on a harder day.",
    ]),
    "sad": ("🌧️ Be gentle with yourself", [
        "Put a hand on your chest and take five slow breaths.",
        "Say to yourself what you'd say to a friend feeling this way.",
        "Message or sit near one person you trust, even for five minutes.",
        "Pick one tiny task (water, fresh air) and stop there.",
    ]),
    "angry": ("🔥 Cool the heat", [
        "Unclench your jaw and drop your shoulders.",
        "Breathe in for 4, out for 6, five rounds.",
        "Write the problem in one sentence, then what you actually want from it.",
        "If you can, walk for 5 minutes before replying to anyone.",
    ]),
    "fear": ("🫧 5-4-3-2-1 grounding", [
        "Notice 5 things you can see.",
        "Notice 4 things you can touch.",
        "Notice 3 things you can hear.",
        "Notice 2 things you can smell.",
        "Notice 1 thing you can taste, then take a slow breath.",
    ]),
    "disgust": ("🌿 Reset your space", [
        "Step away from whatever feels off, even for a minute.",
        "Splash water on your face or change your surroundings.",
        "Name exactly what bothered you: a value, a behaviour, a smell?",
        "Decide: fix it, avoid it, or let it go.",
    ]),
    "surprise": ("⚡ Land after the jolt", [
        "Pause. You don't have to react instantly.",
        "Take three slow breaths.",
        "Ask yourself: is this good, bad, or just new?",
        "Choose only the next small step.",
    ]),
    "neutral": ("🧘 Use the calm", [
        "Do a quick body scan: forehead, jaw, shoulders, hands.",
        "Decide your one priority for the next hour.",
        "Drink some water and stretch for a minute.",
    ]),
}

# "box" = in 4, hold 4, out 4, hold 4   |   "relax" = in 4, out 6 (long exhale calms the body)
BREATHING_PATTERN = {
    "angry": "relax", "fear": "relax", "sad": "relax", "disgust": "relax",
    "surprise": "box", "neutral": "box", "happy": "box",
}


def render_breathing_widget(pattern: str, is_dark_mode: bool):
    """Animated breathing orb in the app's ambient colours."""
    if pattern == "box":
        phases = [["Breathe in", 4, 1.0], ["Hold", 4, 1.0], ["Breathe out", 4, 0.55], ["Hold", 4, 0.55]]
    else:
        phases = [["Breathe in", 4, 1.0], ["Breathe out", 6, 0.55]]
    payload = json.dumps(phases)
    text_color = "#FFFFFF" if is_dark_mode else "#0F172A"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      * {{ margin: 0; padding: 0; box-sizing: border-box;
           font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
      body {{
        background: transparent; color: {text_color};
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        height: 100vh; overflow: hidden;
      }}
      .stage {{ position: relative; width: 150px; height: 150px; display: flex; align-items: center; justify-content: center; }}
      .ring {{ position: absolute; inset: 0; border-radius: 50%; border: 1.5px solid rgba(56, 189, 248, 0.5); }}
      .orb {{
        width: 150px; height: 150px; border-radius: 50%;
        background: radial-gradient(circle at 35% 30%, rgba(255,255,255,0.85), rgba(56,189,248,0.55) 45%, rgba(192,132,252,0.5) 80%);
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.55), inset 0 0 18px rgba(255, 255, 255, 0.35);
        transform: scale(0.55);
      }}
      .label {{ margin-top: 10px; font-weight: 800; font-size: 1.05rem; }}
      .count {{ font-size: 0.85rem; opacity: 0.75; margin-top: 2px; font-variant-numeric: tabular-nums; }}
    </style>
    </head>
    <body>
      <div class="stage"><div class="ring"></div><div class="orb" id="orb"></div></div>
      <div class="label" id="label">Get comfortable...</div>
      <div class="count" id="count">&nbsp;</div>
      <script>
        const phases = {payload};
        const orb = document.getElementById("orb");
        const label = document.getElementById("label");
        const count = document.getElementById("count");
        const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

        async function run() {{
          await sleep(600);
          while (true) {{
            for (const [name, secs, scale] of phases) {{
              label.textContent = name;
              orb.style.transition = "transform " + secs + "s ease-in-out";
              orb.style.transform = "scale(" + scale + ")";
              for (let t = secs; t >= 1; t--) {{
                count.textContent = t;
                await sleep(1000);
              }}
            }}
          }}
        }}
        run();
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=240)


def get_initial_ai_greeting(
    emotion: str,
    conf: float,
    second: str = None,
    second_conf: float = 0.0,
    mixed: bool = False,
    ambiguous: bool = False,
) -> str:
    dom = emotion.lower()
    if mixed and second:
        return (
            f"🍃 Your expression looks like a mix of {dom} ({conf:.0f}%) and {second} ({second_conf:.0f}%). "
            f"A camera can only guess, and you know best. In your own words, how are you actually feeling right now?"
        )
    if ambiguous:
        return (
            "🫧 Your expression is hard to read right now, and that's completely fine. "
            "A camera can only guess. How are you actually feeling at this moment?"
        )
    if dom == "happy":
        return (
            f"🌿 You look genuinely at ease ({conf:.0f}% match). Hold onto this feeling without rushing past it. "
            f"What went well today?"
        )
    elif dom in ["sad", "fear"]:
        return (
            f"💙 Your expression looks heavy or tense ({conf:.0f}% match). Before fixing anything, let your shoulders "
            f"drop and unclench your jaw. What's weighing on your mind right now? I'm here to listen."
        )
    elif dom == "angry":
        return (
            f"🔥 I can see some tension and frustration in your expression ({conf:.0f}% match). That's valid, "
            f"since it often means something crossed a boundary. Take one slow breath with me. What's causing the friction?"
        )
    elif dom == "disgust":
        return (
            f"🌧️ Your expression looks like recoil or resistance ({conf:.0f}% match). Something may feel off to you. "
            f"What just crossed your path?"
        )
    elif dom == "surprise":
        return (
            f"⚡ You look caught off guard ({conf:.0f}% match). Give your mind a moment to catch up. "
            f"What took you by surprise?"
        )
    else:
        return (
            f"🍃 You look calm and composed ({conf:.0f}% match). A quiet mind is a good anchor. "
            f"How is your energy right now?"
        )


def generate_companion_reply(
    user_text: str,
    current_emotion: str,
    history: list = None,
    emotion_context: str = "",
) -> str:
    # 1) Safety first: fixed, caring response, no AI involved
    if is_crisis_message(user_text):
        return CRISIS_REPLY

    api_key = os.environ.get("GEMINI_API_KEY")
    context_line = emotion_context or current_emotion
    system_prompt = (
        "You are 'The Grounded Mental Wellness Coach', a calm, warm, emotionally steady companion inside a student "
        "project app. You are NOT a therapist: never diagnose, never label disorders, never give medical advice. "
        f"The camera made a rough, unreliable guess about the user's facial expression: {context_line}. "
        "Treat it as a gentle hint, never as fact. If the user says something different, believe them. "
        "GUIDELINES:\n"
        "- Reply in 2 to 4 short sentences, in plain text with no lists.\n"
        "- Reflect what they actually said in your own words; acknowledge specifics (studies, exams, burnout, wins).\n"
        "- Offer at most ONE small, concrete suggestion when it fits (a slow breath, a short break, writing one line "
        "down, talking to someone you trust).\n"
        "- End with at most one gentle, open question.\n"
        "- No toxic positivity, no clinical jargon, no customer-service phrases like 'How can I assist you today?'. "
        "Use 0 or 1 emoji.\n"
        "- If they mention self-harm or not wanting to live, respond with care and encourage reaching a trusted "
        "person, a local emergency number or a helpline."
    )

    if api_key:
        # Conversation memory: last 8 turns, must start with a user turn
        turns = []
        for m in (history or [])[-8:]:
            turns.append({"role": "user" if m["role"] == "user" else "model", "text": m["content"]})
        while turns and turns[0]["role"] != "user":
            turns.pop(0)
        turns.append({"role": "user", "text": user_text})

        model_names = [os.environ.get("GEMINI_MODEL", "").strip() or "gemini-2.5-flash", "gemini-2.0-flash"]

        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            contents = [{"role": t["role"], "parts": [{"text": t["text"]}]} for t in turns]
            for name in model_names:
                try:
                    resp = client.models.generate_content(
                        model=name,
                        contents=contents,
                        config={"system_instruction": system_prompt, "temperature": 0.7},
                    )
                    if resp and resp.text:
                        return resp.text.strip()
                except Exception:
                    continue
        except Exception:
            pass

        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=api_key)
            legacy_contents = [{"role": t["role"], "parts": [t["text"]]} for t in turns]
            model = legacy_genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_prompt,
            )
            resp = model.generate_content(legacy_contents)
            if resp and resp.text:
                return resp.text.strip()
        except Exception:
            pass

    # 2) Offline fallback (no key / API error): simple, safe rule-based replies
    txt = user_text.lower().strip()
    dom = current_emotion.lower()

    if any(w in txt for w in ["wsn", "wireless sensor", "subject", "syllabus", "portion", "chapter", "unit"]):
        if any(w in txt for w in ["done", "finished", "completed", "cleared", "over"]):
            return "Finishing a dense, technical subject like WSN takes serious cognitive endurance. Step away from the screen for at least fifteen minutes—let your mental cache clear before you jump to anything else. 🌿✨"
        return "Heavy technical topics naturally cause cognitive fatigue. Don't try to cram every protocol at once; break it into two small concepts, take five steady breaths, and proceed at a sustainable pace. ☕"

    if any(w in txt for w in ["selected", "got the job", "hired", "offer", "placed", "placement"]):
        return "That is a monumental milestone. Pause right here and let the relief settle into your bones—you carried the preparation and you earned this outcome. Be genuinely proud of your perseverance today. 🏆🌱"

    if any(w in txt for w in ["help me", "need help", "assist me", "stuck", "overwhelmed"]):
        return "I am right here with you. Take one deliberate four-second inhale, and let your breath out slowly. Tell me what single piece is feeling the heaviest right now, and let's untangle it together. 💙"

    if any(w in txt for w in ["tired", "exhausted", "burnout", "sleep", "drained"]):
        return "Listen to that signal without guilt. Rest is not a reward you have to earn after collapsing; it is essential maintenance for your nervous system. Unplug your mental tabs and give yourself permission to recharge. 🌿"

    if any(w in txt for w in ["exam", "test", "quiz", "marks", "grade", "cgpa"]):
        if dom == "happy" or any(w in txt for w in ["pass", "cleared", "good", "aced"]):
            return "That hard work paid its dividend. Savor this clarity and confidence—it proves you can navigate high-pressure demands when you stay consistent. 🌟"
        return "Remember that an exam is only an assessment of a snapshot in time, not a measurement of your intellect or capability. Focus only on the step right in front of you today. 💙"

    if any(w in txt for w in ["alone", "lonely", "nobody", "friend", "worth"]):
        return "Isolation makes problems feel deceptively permanent, but you are not alone in this moment. Treat yourself with the same gentleness you would offer a close friend navigating the dark. If it keeps feeling this heavy, talking to a counsellor or someone you trust can really help. 🫂"

    if dom == "happy":
        return "I love seeing you in this space of ease. Notice what contributed to this clarity so you can return to it when future days get noisy. 🌿🎈"
    elif dom in ["sad", "fear"]:
        return "It takes real self-awareness to admit when things feel heavy. You don't have to carry the whole mountain today—just focus on this single breath. 💙"
    elif dom == "angry":
        return "Acknowledge that tension, but don't let it consume your peace. Drop your shoulders, exhale the heat, and decide what action actually serves you best right now. 🔥💪"
    elif dom == "surprise":
        return "Give your mind a beat to process the shift. Life rarely follows our exact roadmap, but you have the resilience to adapt to whatever unfolds. ⚡"
    else:
        return "This balanced, quiet headspace is an asset. Protect this focus and move through your day with steady intention. ☕🫧"


# =============================================================================
# THEME PALETTES & COLOR VARIABLES
# =============================================================================

is_dark = st.session_state.theme == "Dark"

# Smooth mode switches off the heaviest visual effects (mouse ripples, blur, shimmer, floating bubbles)
# so the page and the live video stay fluid. It is on by default and can be turned off in the top bar.
if "smooth_mode" not in st.session_state:
    st.session_state.smooth_mode = True
smooth = bool(st.session_state.smooth_mode)
smooth_css = """
.amb-bubble, .bubble-ambient-container { display: none !important; }
* { backdrop-filter: none !important; -webkit-backdrop-filter: none !important; }
.ambient-glow-text, .ambient-bubble, .banner-title, .shimmer-txt, .stat-value, .feed-header-title, .btn-shimmer-text,
div[data-testid="stButton"] button p, div[data-testid="stButton"] button span,
div[data-testid="stDownloadButton"] button p, div[data-testid="stDownloadButton"] button span,
div[data-testid="stFormSubmitButton"] button p, div[data-testid="stFormSubmitButton"] button span {
    animation: none !important;
    background-position: 0% 50% !important;
}
""" if smooth else ""

if is_dark:
    app_bg = "radial-gradient(circle at 50% 12%, #081b33 0%, #030a16 50%, #01040a 100%)"
    card_bg = "rgba(6, 17, 36, 0.88)"
    capsule_border = "rgba(56, 189, 248, 0.65)"
    capsule_glow = "0 0 20px rgba(56, 189, 248, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.25)"
    banner_halo = "0 0 30px 4px rgba(37, 99, 235, 0.8), 0 0 65px 12px rgba(56, 189, 248, 0.4)"
    text_main = "#FFFFFF"
    text_muted = "#94A3B8"
    accent_tint = "#38BDF8"
    bubble_bg = "rgba(125, 211, 252, 0.35)"
    bubble_border = "rgba(255, 255, 255, 0.6)"
    dropzone_bg = "rgba(7, 18, 38, 0.75)"
    text_shimmer = "linear-gradient(90deg, #ffffff 0%, #38bdf8 30%, #c084fc 60%, #38bdf8 85%, #ffffff 100%)"
    chat_ai_bg = "rgba(14, 165, 233, 0.12)"
    chat_user_bg = "rgba(56, 189, 248, 0.22)"
else:
    app_bg = "radial-gradient(circle at 50% 12%, #BAE6FD 0%, #E0F2FE 45%, #F8FAFC 100%)"
    card_bg = "rgba(255, 255, 255, 0.92)"
    capsule_border = "rgba(2, 132, 199, 0.7)"
    capsule_glow = "0 0 18px rgba(2, 132, 199, 0.28), inset 0 1px 2px rgba(255, 255, 255, 0.8)"
    banner_halo = "0 0 28px 4px rgba(56, 189, 248, 0.5), 0 0 50px 10px rgba(37, 99, 235, 0.22)"
    text_main = "#0F172A"
    text_muted = "#64748B"
    accent_tint = "#0284C7"
    bubble_bg = "rgba(14, 165, 233, 0.3)"
    bubble_border = "rgba(2, 132, 199, 0.5)"
    dropzone_bg = "rgba(255, 255, 255, 0.88)"
    text_shimmer = "linear-gradient(90deg, #0f172a 0%, #0284c7 30%, #7c3aed 60%, #0284c7 85%, #0f172a 100%)"
    chat_ai_bg = "rgba(2, 132, 199, 0.10)"
    chat_user_bg = "rgba(2, 132, 199, 0.18)"

# =============================================================================
# CSS STYLING
# =============================================================================

st.markdown(
    f"""
<style>
:root, .stApp, div[data-testid="stFileUploader"] {{
    --secondary-background-color: {card_bg} !important;
    --background-color: {app_bg} !important;
}}

html, body, [class*="css"], .stApp {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif !important;
    background: {app_bg} !important;
    background-attachment: fixed !important;
    color: {text_main} !important;
    transition: background 0.25s ease, color 0.25s ease;
    -webkit-font-smoothing: antialiased;
}}

.block-container {{
    max-width: 980px !important;
    padding-top: 1rem !important;
    padding-bottom: 4rem !important;
    position: relative;
    z-index: 2;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

@keyframes ambientTextFlow {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

.ambient-glow-text,
.ambient-bubble,
.banner-title,
.badge-main .shimmer-txt,
.stat-value,
.feed-header-title,
.btn-shimmer-text,
div[data-testid="stButton"] button p,
div[data-testid="stButton"] button span:not(.button-raw-emoji):not(.raw-emoji),
div[data-testid="stDownloadButton"] button p,
div[data-testid="stDownloadButton"] button span:not(.button-raw-emoji):not(.raw-emoji),
div[data-testid="stFormSubmitButton"] button p,
div[data-testid="stFormSubmitButton"] button span:not(.button-raw-emoji):not(.raw-emoji) {{
    background: {text_shimmer} !important;
    background-size: 250% auto !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: ambientTextFlow 7s ease infinite !important;
    display: inline-block;
    white-space: nowrap !important;
}}

.raw-emoji,
.badge-main .raw-emoji,
.history-item .raw-emoji,
.button-raw-emoji,
span.button-raw-emoji {{
    background: none !important;
    -webkit-background-clip: border-box !important;
    -webkit-text-fill-color: initial !important;
    color: initial !important;
    animation: none !important;
    display: inline-block !important;
    filter: none !important;
    margin-right: 6px !important;
    font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif !important;
}}

.bubble-ambient-container {{
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    overflow: hidden;
    z-index: 1;
}}

.amb-bubble {{
    position: absolute;
    bottom: -100px;
    background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.7), {bubble_bg} 60%, transparent 95%);
    border: 1.5px solid {bubble_border};
    border-radius: 50%;
    box-shadow: inset 0 2px 6px rgba(255,255,255,0.6), 0 4px 15px rgba(56, 189, 248, 0.25);
    animation: floatAmbient 14s infinite ease-in;
}}

.ab1 {{ width: 55px; height: 55px; left: 8%;  animation-duration: 15s; animation-delay: 0s; }}
.ab2 {{ width: 32px; height: 32px; left: 24%; animation-duration: 11s; animation-delay: 3s; }}
.ab3 {{ width: 75px; height: 75px; left: 48%; animation-duration: 18s; animation-delay: 1s; }}
.ab4 {{ width: 42px; height: 42px; left: 72%; animation-duration: 13s; animation-delay: 4s; }}
.ab5 {{ width: 62px; height: 62px; left: 90%; animation-duration: 16s; animation-delay: 2s; }}

@keyframes floatAmbient {{
    0%   {{ transform: translateY(0) scale(0.85); opacity: 0; }}
    15%  {{ opacity: 0.85; }}
    85%  {{ opacity: 0.85; }}
    100% {{ transform: translateY(-120vh) scale(1.15); opacity: 0; }}
}}

.banner-capsule {{
    width: 100%;
    max-width: 820px;
    height: 94px;
    border-radius: 9999px;
    background: {card_bg};
    border: 2px solid {capsule_border};
    box-shadow: {banner_halo}, inset 0 1px 2px rgba(255, 255, 255, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.2rem;
    margin: 0.5rem auto 1.8rem auto;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}}

.bubble-logo {{
    position: relative;
    width: 38px;
    height: 38px;
    flex-shrink: 0;
}}

.bubble-logo .bubble {{
    position: absolute;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #ffffff 0%, #c084fc 40%, #818cf8 80%);
    box-shadow: 0 0 12px rgba(192, 132, 252, 0.7);
}}

.b-lg {{ width: 22px; height: 22px; top: 2px; right: 2px; }}
.b-md {{ width: 14px; height: 14px; bottom: 3px; left: 2px; }}
.b-sm {{ width: 9px;  height: 9px;  top: 13px; left: 0px; }}

.banner-title {{
    font-size: 2.15rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    filter: drop-shadow(0 2px 10px rgba(0, 0, 0, 0.6)) drop-shadow(0 0 16px rgba(56, 189, 248, 0.5));
}}

div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button,
div[data-testid="stCameraInput"] button,
section[data-testid="stFileUploadDropzone"] button,
div[data-testid="stFileUploader"] button,
div[data-testid="stFormSubmitButton"] button {{
    position: relative !important;
    border-radius: 9999px !important;
    padding: 0.65rem 1.4rem !important;
    background: {card_bg} !important;
    border: 1.5px solid {capsule_border} !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.25) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    overflow: hidden !important;
    white-space: nowrap !important;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

.theme-capsule-wrapper div[data-testid="stButton"] button {{
    min-width: 95px !important;
    padding: 0.55rem 1.1rem !important;
}}

div[data-testid="stButton"] button p,
div[data-testid="stButton"] button span,
div[data-testid="stButton"] button div,
div[data-testid="stDownloadButton"] button p,
div[data-testid="stDownloadButton"] button span,
div[data-testid="stFormSubmitButton"] button p,
div[data-testid="stFormSubmitButton"] button span {{
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    margin: 0 !important;
    line-height: normal !important;
    white-space: nowrap !important;
}}

div[data-testid="stButton"] button::before,
div[data-testid="stDownloadButton"] button::before,
section[data-testid="stFileUploadDropzone"] button::before,
div[data-testid="stFileUploader"] button::before,
div[data-testid="stFormSubmitButton"] button::before {{
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 48% !important;
    background: linear-gradient(to bottom, rgba(255, 255, 255, 0.22), transparent) !important;
    border-radius: 9999px 9999px 0 0 !important;
    pointer-events: none !important;
}}

div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover,
section[data-testid="stFileUploadDropzone"] button:hover,
div[data-testid="stFileUploader"] button:hover,
div[data-testid="stFormSubmitButton"] button:hover {{
    transform: translateY(-2px) !important;
    border-color: #38bdf8 !important;
    box-shadow: 0 0 24px rgba(56, 189, 248, 0.55), inset 0 1px 2px rgba(255, 255, 255, 0.4) !important;
}}

.active-capsule div[data-testid="stButton"] button {{
    border-color: #38bdf8 !important;
    box-shadow: 0 0 22px rgba(56, 189, 248, 0.5), inset 0 0 12px rgba(56, 189, 248, 0.25) !important;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.45), rgba(56, 189, 248, 0.3)) !important;
}}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
div[data-testid="stFileUploader"] section div[role="listitem"],
div[data-testid="stFileUploader"] ul li {{
    background: {card_bg} !important;
    background-color: {card_bg} !important;
    color: {text_main} !important;
    border: 1.5px solid {capsule_border} !important;
    border-radius: 9999px !important;
    box-shadow: {capsule_glow} !important;
    padding: 6px 16px !important;
}}

div[data-testid="stFileUploader"] small {{
    color: {text_muted} !important;
}}

div[data-testid="stFileUploader"] svg {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    fill: {accent_tint} !important;
    color: {accent_tint} !important;
}}

div[data-testid="stFileUploader"] div:has(> svg),
div[data-testid="stFileUploader"] span:has(> svg) {{
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

div[data-testid="stFileUploaderDeleteBtn"] button,
div[data-testid="stFileUploader"] button[aria-label*="delete" i],
div[data-testid="stFileUploader"] button[aria-label*="Remove" i] {{
    background: rgba(255, 255, 255, 0.12) !important;
    border: 1px solid {capsule_border} !important;
    border-radius: 50% !important;
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    min-height: 28px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}

div[data-testid="stFileUploaderDeleteBtn"] button:hover svg,
div[data-testid="stFileUploader"] button[aria-label*="delete" i]:hover svg {{
    fill: #ef4444 !important;
}}

div[data-testid="stFileUploader"] section {{
    background: {dropzone_bg} !important;
    border: 1.5px dashed {capsule_border} !important;
    border-radius: 20px !important;
    color: {text_main} !important;
    box-shadow: {capsule_glow} !important;
}}

div[data-testid="stCameraInput"] > div {{
    background: {dropzone_bg} !important;
    border: 1.5px solid {capsule_border} !important;
    border-radius: 20px !important;
    box-shadow: {capsule_glow} !important;
    padding: 15px !important;
}}

div[data-testid="stCameraInput"] video {{
    transform: scaleX(-1) !important;
    border-radius: 14px !important;
}}

.st-key-feed_panel,
.st-key-bars_panel,
.st-key-coach_panel,
.st-key-live_panel,
.st-key-insights_panel {{
    background: {card_bg};
    border: 1.5px solid {capsule_border};
    box-shadow: {capsule_glow};
    border-radius: 24px;
    padding: 1.4rem;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    margin-bottom: 1.2rem;
}}
.st-key-coach_panel {{
    padding: 20px 24px;
}}

div[data-testid="stExpander"] {{
    border: 1.5px solid {capsule_border} !important;
    border-radius: 18px !important;
    background: transparent !important;
    margin-top: 10px;
}}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p {{
    color: {text_main} !important;
    font-weight: 700 !important;
}}
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] li {{
    color: {text_main} !important;
}}

.clean-panel {{
    background: {card_bg};
    border: 1.5px solid {capsule_border};
    box-shadow: {capsule_glow};
    border-radius: 24px;
    padding: 1.4rem;
    backdrop-filter: blur(18px);
    margin-bottom: 1.2rem;
}}
.feed-header-title {{
    font-size: 1.05rem;
    font-weight: 800;
    margin-bottom: 8px;
    letter-spacing: -0.01em;
}}
.metric-badge {{
    background: {card_bg};
    border: 1.5px solid {capsule_border};
    box-shadow: {capsule_glow};
    border-radius: 22px;
    padding: 16px 20px;
    margin-bottom: 14px;
}}
.badge-main {{
    font-size: 1.65rem;
    font-weight: 800;
    margin-top: 4px;
}}
.stat-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 12px;
}}
.stat-row.four {{
    grid-template-columns: repeat(4, 1fr);
    margin-bottom: 12px;
}}
@media (max-width: 640px) {{
    .stat-row.four {{ grid-template-columns: repeat(2, 1fr); }}
}}
.stat-item {{
    padding: 9px 12px;
    border-radius: 14px;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid {capsule_border};
}}
.stat-label {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    color: {text_muted} !important;
}}
.stat-value {{
    font-size: 1.05rem;
    font-weight: 800;
}}

.chip-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 4px 0 14px 0;
}}
.chip {{
    padding: 4px 12px;
    border-radius: 9999px;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid {capsule_border};
    font-size: 0.78rem;
    font-weight: 700;
    color: {text_main};
}}

.face-row {{
    font-size: 0.85rem;
    padding: 5px 0;
    color: {text_main};
}}
.face-focus {{
    font-size: 0.65rem;
    font-weight: 800;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 9999px;
    border: 1px solid {accent_tint};
    color: {accent_tint};
    margin-left: 6px;
}}

.chat-capsule-container {{
    background: {card_bg};
    border: 1.5px solid {capsule_border};
    box-shadow: {capsule_glow};
    border-radius: 24px;
    padding: 20px 24px;
    margin-top: 6px;
    margin-bottom: 18px;
    backdrop-filter: blur(18px);
}}
.chat-bubble-ai {{
    background: {chat_ai_bg};
    border: 1px solid {capsule_border};
    border-radius: 16px 16px 16px 4px;
    padding: 13px 19px;
    font-size: 0.94rem;
    line-height: 1.55;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}}
.chat-bubble-user {{
    background: {chat_user_bg};
    border: 1px solid {accent_tint};
    border-radius: 16px 16px 4px 16px;
    padding: 13px 19px;
    font-size: 0.94rem;
    line-height: 1.55;
    margin-bottom: 10px;
    text-align: right;
    margin-left: auto;
    max-width: 80%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}}

.preset-title {{
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: {text_muted};
    margin: 8px 0 8px 0;
}}
.preset-grid div[data-testid="stButton"] button {{
    padding: 0.55rem 1rem !important;
    font-size: 0.85rem !important;
    border-radius: 9999px !important;
    white-space: normal !important;
    text-align: center !important;
    line-height: 1.3 !important;
    height: 100% !important;
    min-height: 44px !important;
}}

div[data-testid="stForm"] {{
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}}

.history-item {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 11px 16px;
    margin-bottom: 7px;
    border-radius: 16px;
    background: {card_bg};
    border: 1px solid {capsule_border};
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}}
.live-top {{
    font-size: 1.35rem;
    font-weight: 800;
    margin-bottom: 10px;
    color: {text_main};
}}
.live-top small {{
    font-size: 0.8rem;
    color: {text_muted};
    font-weight: 700;
}}
.live-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
}}
.live-lab {{
    width: 104px;
    font-size: 0.8rem;
    font-weight: 700;
    color: {text_main};
    text-align: right;
    white-space: nowrap;
}}
.live-track {{
    flex: 1;
    height: 10px;
    border-radius: 999px;
    background: rgba(51, 65, 85, 0.45);
    overflow: hidden;
    border: 1px solid {capsule_border};
}}
.live-fill {{
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #0284C7, {accent_tint});
    transition: width 0.3s ease;
}}
.live-val {{
    width: 52px;
    font-size: 0.8rem;
    font-weight: 800;
    color: {text_main};
    font-variant-numeric: tabular-nums;
}}
.live-note {{
    font-size: 0.74rem;
    color: {text_muted};
    margin-top: 10px;
    line-height: 1.5;
}}
.live-empty {{
    padding: 18px 4px;
    font-size: 0.9rem;
    color: {text_muted};
}}
.session-title {{
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.01em;
}}
{smooth_css}
</style>

<div class="bubble-ambient-container">
    <div class="amb-bubble ab1"></div>
    <div class="amb-bubble ab2"></div>
    <div class="amb-bubble ab3"></div>
    <div class="amb-bubble ab4"></div>
    <div class="amb-bubble ab5"></div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# JAVASCRIPT: EMOJI SHIELD & WATER WAKE
# =============================================================================

components.html(
    f"""
    <script>
    const SMOOTH = {str(smooth).lower()};
    let parentDoc;
    try {{
        parentDoc = window.parent.document;
    }} catch (e) {{
        parentDoc = document;
    }}

    function shieldButtonEmojis() {{
        if (!parentDoc) return;
        const buttons = parentDoc.querySelectorAll('div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button, div[data-testid="stFormSubmitButton"] button');
        buttons.forEach(btn => {{
            const txtNode = btn.querySelector('p') || btn.querySelector('span');
            if (txtNode && !txtNode.dataset.emojiShielded) {{
                const rawText = txtNode.innerText.trim();
                const chars = Array.from(rawText);
                let emojiBuf = [];
                let i = 0;

                const isEmojiChar = (ch) => /\\p{{Extended_Pictographic}}|\\uFE0F|\\u200D/u.test(ch);

                while (i < chars.length && (isEmojiChar(chars[i]) || chars[i] === ' ')) {{
                    if (isEmojiChar(chars[i])) emojiBuf.push(chars[i]);
                    i++;
                }}

                if (emojiBuf.length > 0) {{
                    const emojiStr = emojiBuf.join('');
                    const textStr = chars.slice(i).join('').trim();
                    txtNode.innerHTML = `<span class="button-raw-emoji">${{emojiStr}}</span><span class="btn-shimmer-text">${{textStr}}</span>`;
                    txtNode.dataset.emojiShielded = "true";
                }}
            }}
        }});
    }}

    function sanitizeUploaderColors() {{
        if (!parentDoc) return;
        const uploader = parentDoc.querySelector('div[data-testid="stFileUploader"]');
        if (!uploader) return;

        const allDivs = uploader.querySelectorAll('div, li, [role="listitem"]');
        allDivs.forEach(el => {{
            const bg = window.getComputedStyle(el).backgroundColor;
            if (bg.includes('255, 255, 255') || bg === 'white' || bg === '#ffffff') {{
                if (el.offsetWidth > 45) {{
                    el.style.setProperty('background', '{card_bg}', 'important');
                    el.style.setProperty('background-color', '{card_bg}', 'important');
                    el.style.setProperty('border', '1.5px solid {capsule_border}', 'important');
                    el.style.setProperty('border-radius', '9999px', 'important');
                    el.style.setProperty('box-shadow', '{capsule_glow}', 'important');
                }} else {{
                    el.style.setProperty('background', 'transparent', 'important');
                    el.style.setProperty('background-color', 'transparent', 'important');
                    el.style.setProperty('box-shadow', 'none', 'important');
                    el.style.setProperty('border', 'none', 'important');
                }}
            }}
        }});

        const texts = uploader.querySelectorAll('span, small, p');
        texts.forEach(t => {{
            const col = window.getComputedStyle(t).color;
            if (col.includes('0, 0, 0') || col.includes('49, 51, 63')) {{
                t.style.setProperty('color', '{text_main}', 'important');
            }}
        }});

        const svgs = uploader.querySelectorAll('svg');
        svgs.forEach(svg => {{
            svg.style.setProperty('background', 'transparent', 'important');
            svg.style.setProperty('fill', '{accent_tint}', 'important');
        }});
    }}

    if (window._shieldInterval) clearInterval(window._shieldInterval);
    window._shieldInterval = setInterval(() => {{
        shieldButtonEmojis();
        sanitizeUploaderColors();
    }}, SMOOTH ? 500 : 120);

    if (parentDoc) {{
        const oldWake0 = parentDoc.getElementById('water-swimming-halo');
        if (oldWake0) oldWake0.remove();
        if (window._swimMouseMove) parentDoc.removeEventListener('mousemove', window._swimMouseMove);
        if (window.spawnBubbleBurst) parentDoc.removeEventListener('click', window.spawnBubbleBurst, true);
    }}

    if (parentDoc && !SMOOTH) {{
        const oldWake = parentDoc.getElementById('water-swimming-halo');
        if (oldWake) oldWake.remove();

        const halo = parentDoc.createElement('div');
        halo.id = 'water-swimming-halo';
        halo.style.position = 'fixed';
        halo.style.top = '0px';
        halo.style.left = '0px';
        halo.style.width = '24px';
        halo.style.height = '24px';
        halo.style.borderRadius = '50%';
        halo.style.pointerEvents = 'none';
        halo.style.zIndex = '9999999';
        halo.style.background = 'radial-gradient(circle at 35% 35%, rgba(125, 211, 252, 0.45), rgba(2, 132, 199, 0.15) 70%, transparent 95%)';
        halo.style.boxShadow = '0 0 16px rgba(56, 189, 248, 0.45), inset 0 0 8px rgba(255, 255, 255, 0.4)';
        halo.style.border = '1px solid rgba(125, 211, 252, 0.35)';
        halo.style.opacity = '0';
        halo.style.transition = 'opacity 0.3s ease';
        halo.style.willChange = 'transform';

        parentDoc.body.appendChild(halo);

        let mouseX = window.innerWidth / 2;
        let mouseY = window.innerHeight / 2;
        let swimX = mouseX;
        let swimY = mouseY;
        let lastX = mouseX;
        let lastY = mouseY;
        let lastRippleTime = 0;

        const onMouseMove = (e) => {{
            halo.style.opacity = '0.85';
            mouseX = e.clientX;
            mouseY = e.clientY;

            const now = Date.now();
            const dist = Math.hypot(mouseX - lastX, mouseY - lastY);

            if (dist > 14 && now - lastRippleTime > 55) {{
                lastRippleTime = now;

                const ripple = parentDoc.createElement('div');
                ripple.style.position = 'fixed';
                ripple.style.left = (mouseX - 12) + 'px';
                ripple.style.top = (mouseY - 12) + 'px';
                ripple.style.width = '24px';
                ripple.style.height = '24px';
                ripple.style.borderRadius = '50%';
                ripple.style.pointerEvents = 'none';
                ripple.style.zIndex = '9999998';
                ripple.style.border = '1.5px solid rgba(125, 211, 252, 0.6)';
                ripple.style.background = 'radial-gradient(circle, rgba(56, 189, 248, 0.18) 0%, transparent 70%)';
                ripple.style.transition = 'transform 0.75s cubic-bezier(0.1, 0.8, 0.3, 1), opacity 0.75s ease-out';
                ripple.style.transform = 'scale(0.4)';
                ripple.style.opacity = '0.9';

                parentDoc.body.appendChild(ripple);

                requestAnimationFrame(() => {{
                    ripple.style.transform = 'scale(3.2)';
                    ripple.style.opacity = '0';
                }});

                setTimeout(() => {{
                    if (ripple.parentNode) ripple.parentNode.removeChild(ripple);
                }}, 800);
            }}

            lastX = mouseX;
            lastY = mouseY;
        }};

        parentDoc.removeEventListener('mousemove', window._swimMouseMove);
        window._swimMouseMove = onMouseMove;
        parentDoc.addEventListener('mousemove', onMouseMove);

        function animateWaterWake() {{
            const viscosity = 0.18;
            const vx = (mouseX - swimX) * viscosity;
            const vy = (mouseY - swimY) * viscosity;

            swimX += vx;
            swimY += vy;

            const speed = Math.min(Math.hypot(vx, vy), 25);
            const stretch = 1 + speed * 0.025;
            const squish = 1 - speed * 0.015;
            const angle = Math.atan2(vy, vx) * (180 / Math.PI);

            halo.style.transform = `translate3d(${{swimX - 12}}px, ${{swimY - 12}}px, 0) rotate(${{angle}}deg) scale(${{stretch}}, ${{squish}})`;

            requestAnimationFrame(animateWaterWake);
        }}
        animateWaterWake();

        parentDoc.removeEventListener('click', window.spawnBubbleBurst, true);
        window.spawnBubbleBurst = function(e) {{
            for (let i = 0; i < 5; i++) {{
                const bubble = parentDoc.createElement('div');
                const size = Math.floor(Math.random() * 24) + 14;
                const destX = (Math.random() - 0.5) * 90;
                const destY = -(Math.random() * 95 + 45);

                bubble.style.position = 'fixed';
                bubble.style.left = (e.clientX - size / 2) + 'px';
                bubble.style.top = (e.clientY - size / 2) + 'px';
                bubble.style.width = size + 'px';
                bubble.style.height = size + 'px';
                bubble.style.borderRadius = '50%';
                bubble.style.background = 'radial-gradient(circle at 30% 30%, rgba(255,255,255,0.9), rgba(56, 189, 248, 0.45) 60%, transparent 95%)';
                bubble.style.border = '1px solid rgba(255, 255, 255, 0.7)';
                bubble.style.boxShadow = 'inset 0 2px 4px rgba(255,255,255,0.6), 0 4px 12px rgba(56, 189, 248, 0.35)';
                bubble.style.pointerEvents = 'none';
                bubble.style.zIndex = '999999';
                bubble.style.transition = 'transform 0.85s cubic-bezier(0.25, 1, 0.5, 1), opacity 0.85s ease-out';
                bubble.style.transform = 'translate(0, 0) scale(0.6)';
                bubble.style.opacity = '1';

                parentDoc.body.appendChild(bubble);

                requestAnimationFrame(() => {{
                    bubble.style.transform = `translate(${{destX}}px, ${{destY}}px) scale(1.15)`;
                    bubble.style.opacity = '0';
                }});

                setTimeout(() => {{
                    if (bubble.parentNode) bubble.parentNode.removeChild(bubble);
                }}, 900);
            }}
        }};

        parentDoc.addEventListener('click', window.spawnBubbleBurst, true);
    }}
    </script>
    """,
    height=0,
    width=0,
)

# =============================================================================
# 1. TOP RIGHT: DARK / LIGHT SWITCH CAPSULES
# =============================================================================

_, col_smooth, col_dark, col_light = st.columns([4.2, 2.2, 1.8, 1.8])

with col_smooth:
    st.markdown('<div class="theme-capsule-wrapper">', unsafe_allow_html=True)
    if st.button(f"⚡ Smooth: {'On' if smooth else 'Off'}", key="smooth_btn"):
        st.session_state.smooth_mode = not smooth
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col_dark:
    is_d_active = "active-capsule theme-capsule-wrapper" if is_dark else "theme-capsule-wrapper"
    st.markdown(f'<div class="{is_d_active}">', unsafe_allow_html=True)
    if st.button("🌙 Dark", key="theme_dark_btn"):
        st.session_state.theme = "Dark"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col_light:
    is_l_active = "active-capsule theme-capsule-wrapper" if not is_dark else "theme-capsule-wrapper"
    st.markdown(f'<div class="{is_l_active}">', unsafe_allow_html=True)
    if st.button("🌼 Light", key="theme_light_btn"):
        st.session_state.theme = "Light"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 2. MAIN GLOWING TITLE BANNER CAPSULE
# =============================================================================

st.markdown(
    """
    <div class="banner-capsule">
        <div class="bubble-logo">
            <div class="bubble b-lg"></div>
            <div class="bubble b-md"></div>
            <div class="bubble b-sm"></div>
        </div>
        <div class="banner-title">Facial Emotion Intelligence</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 3. THE 2 CAPSULES: LIVE OPTICAL STREAM & UPLOAD STATIC PORTRAIT
# =============================================================================

col_space_l, col_btn_live, col_btn_upload, col_space_r = st.columns([1, 4, 4, 1])

with col_btn_live:
    live_cls = "active-capsule" if st.session_state.active_mode == "live" else ""
    st.markdown(f'<div class="{live_cls}">', unsafe_allow_html=True)
    if st.button("📷 Live Optical Stream", key="mode_live_btn"):
        st.session_state.active_mode = "live"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col_btn_upload:
    upload_cls = "active-capsule" if st.session_state.active_mode == "upload" else ""
    st.markdown(f'<div class="{upload_cls}">', unsafe_allow_html=True)
    if st.button("📁 Upload Static Portrait", key="mode_upload_btn"):
        st.session_state.active_mode = "upload"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def draw_clean_bounding(
    image_rgb: np.ndarray,
    detections: list,
    is_dark_mode: bool,
    labels: list = None,
    focus_idx: int = None,
) -> np.ndarray:
    """Corner-bracket boxes. If `labels` is given, each face gets its own emotion tag.
    With several faces the tags are numbered and the focus face is drawn in an accent colour."""
    out = image_rgb.copy()
    img_h, img_w = out.shape[:2]
    box_color = (125, 211, 252) if is_dark_mode else (2, 132, 199)
    focus_color = (192, 132, 252) if is_dark_mode else (124, 58, 237)
    text_color = (15, 23, 42) if is_dark_mode else (255, 255, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, img_w / 1100.0)
    font_thick = 2 if font_scale >= 0.8 else 1
    multi = len(detections) > 1

    for i, res in enumerate(detections):
        x, y, w, h = (int(res["region"][k]) for k in ("x", "y", "w", "h"))
        x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
        L = max(int(min(w, h) * 0.18), 16)

        is_focus = multi and focus_idx is not None and i == focus_idx
        col = focus_color if is_focus else box_color
        thick = 3 if is_focus else 2

        for cx, cy, dx, dy in [
            (x, y, 1, 1),
            (x + w, y, -1, 1),
            (x, y + h, 1, -1),
            (x + w, y + h, -1, -1),
        ]:
            cv2.line(out, (cx, cy), (cx + dx * L, cy), col, thick, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy + dy * L), col, thick, cv2.LINE_AA)

        if labels and i < len(labels):
            text = f"{i + 1}: {labels[i]}" if multi else labels[i]
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thick)
            pad = 5
            bx1 = max(0, min(x, img_w - tw - 2 * pad - 1))
            bx2 = bx1 + tw + 2 * pad
            by2 = y - 4
            by1 = by2 - th - 2 * pad
            if by1 < 0:  # no room above the face: tuck the tag just inside the box
                by1 = y + 4
                by2 = by1 + th + 2 * pad
            cv2.rectangle(out, (bx1, by1), (bx2, by2), col, -1)
            cv2.putText(out, text, (bx1 + pad, by2 - pad), font, font_scale, text_color, font_thick, cv2.LINE_AA)
    return out


def render_live_animated_bars(scores: dict, dominant_raw: str, is_dark_mode: bool):
    sorted_scores = [
        [str(emotion), float(val)]
        for emotion, val in sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)
    ]
    payload = json.dumps(sorted_scores)

    bg_track = "rgba(10, 25, 48, 0.75)" if is_dark_mode else "rgba(226, 232, 240, 0.9)"
    track_border = "rgba(56, 189, 248, 0.25)" if is_dark_mode else "rgba(2, 132, 199, 0.25)"
    text_color = "#FFFFFF" if is_dark_mode else "#0F172A"
    grid_color = "rgba(56, 189, 248, 0.12)" if is_dark_mode else "rgba(2, 132, 199, 0.14)"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        user-select: none;
        -webkit-font-smoothing: antialiased;
      }}
      body {{
        background: transparent;
        color: {text_color};
        padding: 8px 4px 6px 4px;
        overflow: hidden;
      }}

      @keyframes ambientTextFlow {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
      }}

      .chart-container {{
        display: flex;
        flex-direction: column;
        gap: 12px;
        position: relative;
      }}
      .grid-lines {{
        position: absolute;
        top: 0;
        bottom: 24px;
        left: 92px;
        right: 56px;
        display: flex;
        justify-content: space-between;
        pointer-events: none;
        z-index: 0;
      }}
      .grid-line {{
        width: 1px;
        background: {grid_color};
        height: 100%;
      }}
      .bar-row {{
        display: flex;
        align-items: center;
        position: relative;
        z-index: 1;
        height: 24px;
      }}
      .label {{
        width: 88px;
        font-size: 0.86rem;
        font-weight: 700;
        background: {text_shimmer};
        background-size: 250% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: ambientTextFlow 7s ease infinite;
        text-align: right;
        padding-right: 12px;
        letter-spacing: 0.01em;
      }}
      .track {{
        flex: 1;
        height: 13px;
        background: {bg_track};
        border-radius: 999px;
        overflow: hidden;
        position: relative;
        border: 1px solid {track_border};
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.4);
      }}
      .fill {{
        height: 100%;
        width: 0%;
        border-radius: 999px;
        transition: width 1.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
      }}
      .fill {{
        min-width: 6px;
        background: {text_shimmer};
        background-size: 250% auto;
        animation: ambientTextFlow 7s ease infinite;
      }}
      .fill.rank0 {{
        box-shadow: 0 0 14px {accent_tint};
      }}
      .fill.rank1 {{
        box-shadow: 0 0 8px {accent_tint}99;
      }}
      .fill.rank0::after {{
        content: '';
        position: absolute;
        right: 0;
        top: 0;
        bottom: 0;
        width: 6px;
        background: #ffffff;
        border-radius: 50%;
        box-shadow: 0 0 8px #ffffff;
      }}
      .val-text {{
        width: 54px;
        font-size: 0.84rem;
        font-weight: 800;
        text-align: right;
        padding-left: 8px;
        background: {text_shimmer};
        background-size: 250% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: ambientTextFlow 7s ease infinite;
        font-variant-numeric: tabular-nums;
      }}
      .axis-row {{
        display: flex;
        margin-left: 92px;
        margin-right: 56px;
        justify-content: space-between;
        font-size: 0.72rem;
        color: #94a3b8;
        font-weight: 700;
        padding-top: 6px;
      }}
    </style>
    </head>
    <body>
      <div class="chart-container" id="chart">
        <div class="grid-lines">
          <div class="grid-line"></div>
          <div class="grid-line"></div>
          <div class="grid-line"></div>
          <div class="grid-line"></div>
          <div class="grid-line"></div>
          <div class="grid-line"></div>
        </div>
      </div>
      <div class="axis-row">
        <span>0</span>
        <span>20</span>
        <span>40</span>
        <span>60</span>
        <span>80</span>
        <span>100</span>
      </div>

      <script>
        const data = {payload};
        const dominant = "{dominant_raw.lower()}";
        const container = document.getElementById("chart");

        data.forEach(([emotion, val], idx) => {{
          const row = document.createElement("div");
          row.className = "bar-row";

          const label = document.createElement("div");
          label.className = "label";
          label.innerText = emotion.charAt(0).toUpperCase() + emotion.slice(1);

          const track = document.createElement("div");
          track.className = "track";

          const fill = document.createElement("div");
          fill.className = "fill" + (idx === 0 ? " rank0" : (idx === 1 ? " rank1" : ""));
          // stagger each bar's light flow so the glow ripples down the chart
          fill.style.animationDelay = (-idx * 0.9) + "s";

          const valText = document.createElement("div");
          valText.className = "val-text";
          valText.innerText = "0.0%";

          track.appendChild(fill);
          row.appendChild(label);
          row.appendChild(track);
          row.appendChild(valText);
          container.appendChild(row);

          requestAnimationFrame(() => {{
            setTimeout(() => {{
              fill.style.width = Math.min(val, 100) + "%";
            }}, 50);
          }});

          const target = parseFloat(val);
          const duration = 1200;
          const startTime = performance.now();

          function step(currTime) {{
            const progress = Math.min((currTime - startTime) / duration, 1);
            const ease = 1 - Math.pow(1 - progress, 3);
            valText.innerText = (target * ease).toFixed(1) + "%";

            if (progress < 1) {{
              requestAnimationFrame(step);
            }} else {{
              valText.innerText = target.toFixed(1) + "%";
            }}
          }}
          requestAnimationFrame(step);
        }});
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=280)


def render_session_insights(history: list, is_dark_mode: bool):
    """Analytics panel: emotion mix, most frequent emotion, average confidence and mood trend."""
    df = pd.DataFrame(history)
    n = len(df)
    emo_keys = list(EMOJI_MAP.keys())
    emo_cols = [k.capitalize() for k in emo_keys]

    # History stores percentages as strings ("12.3%"). Turn them back into numbers.
    num = df[emo_cols].apply(
        lambda s: pd.to_numeric(s.astype(str).str.rstrip("%"), errors="coerce")
    ).fillna(0.0)

    dominant = num.idxmax(axis=1).str.lower()
    counts = dominant.value_counts()
    top_emotion = str(counts.index[0])
    top_count = int(counts.iloc[0])

    conf = pd.to_numeric(df["Confidence"].astype(str).str.rstrip("%"), errors="coerce")
    avg_conf = float(conf.mean())
    inconclusive = int((df["Classification"] == "Ambiguous / Inconclusive").sum())

    avg_scores = {k: float(num[k.capitalize()].mean()) for k in emo_keys}
    mood = sum(num[k.capitalize()] / 100.0 * VALENCE[k] for k in emo_keys) * 100.0
    mood = mood.reset_index(drop=True)

    with st.container(key="insights_panel"):
        st.markdown(
            '<div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">'
            '<span style="font-size:1.35rem;">📊</span>'
            '<div class="ambient-glow-text" style="font-weight:800; font-size:1.15rem;">Session Insights</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="stat-row four">'
            f'<div class="stat-item"><div class="stat-label">Readings</div><div class="stat-value">{n}</div></div>'
            f'<div class="stat-item"><div class="stat-label">Most frequent</div>'
            f'<div style="display:flex; align-items:center;">'
            f'<span class="raw-emoji">{EMOJI_MAP.get(top_emotion, "•")}</span>'
            f'<span class="stat-value">{top_emotion.capitalize()} ×{top_count}</span></div></div>'
            f'<div class="stat-item"><div class="stat-label">Avg confidence</div><div class="stat-value">{avg_conf:.1f}%</div></div>'
            f'<div class="stat-item"><div class="stat-label">Inconclusive</div><div class="stat-value">{inconclusive}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        chips = "".join(
            f'<span class="chip"><span class="raw-emoji" style="margin-right:4px !important;">{EMOJI_MAP.get(str(e), "•")}</span>'
            f'{str(e).capitalize()} ×{int(c)}</span>'
            for e, c in counts.items()
        )
        st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

        c_left, c_right = st.columns(2, gap="medium")

        with c_left:
            st.markdown('<div class="feed-header-title">Emotion mix (session average)</div>', unsafe_allow_html=True)
            render_live_animated_bars(avg_scores, top_emotion, is_dark_mode)

        with c_right:
            st.markdown('<div class="feed-header-title">Mood trend</div>', unsafe_allow_html=True)
            if n >= 2:
                trend = pd.DataFrame(
                    {"Mood score": mood.values},
                    index=pd.RangeIndex(1, n + 1, name="Reading"),
                )
                st.line_chart(trend, height=240)

                direction = ""
                if n >= 3:
                    change = float(np.polyfit(np.arange(n), mood.values, 1)[0]) * (n - 1)
                    if change > 15:
                        direction = "Overall trend: improving. "
                    elif change < -15:
                        direction = "Overall trend: dipping. "
                    else:
                        direction = "Overall trend: steady. "
                st.caption(
                    f"{direction}Mood score = emotion mix weighted by how pleasant each emotion is "
                    f"(+100 very positive, -100 very negative). A rough guide only."
                )
            else:
                st.info("Take at least two readings to see a mood trend.")


# =============================================================================
# ACTIVE VIEW CONTENT
# =============================================================================

photos = []  # list of (file, is_camera)
burst_mode = False  # True for a live burst: always averaged

if st.session_state.active_mode == "live":
    use_stream = False
    if WEBRTC_OK:
        cam_style = st.radio(
            "Camera mode",
            ["🎥 Real-time stream", "📷 Classic snapshot"],
            horizontal=True,
            key="cam_style",
            label_visibility="collapsed",
        )
        use_stream = cam_style.startswith("🎥")
    else:
        st.info(
            "Real-time mode needs two extra packages: pip install streamlit-webrtc av. "
            "Using the classic snapshot camera for now."
        )

    if use_stream:
        if "live_state" not in st.session_state:
            st.session_state.live_state = LiveState(load_hse_models())
        live_state = st.session_state.live_state

        col_vid, col_read = st.columns([1.35, 1], gap="medium")
        with col_vid:
            ctx = webrtc_streamer(
                key="live-emotion",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={
                    "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "frameRate": {"ideal": 30}},
                    "audio": False,
                },
                video_frame_callback=make_live_callback(live_state),
                async_processing=True,
            )
            playing = bool(ctx.state.playing)

            btn_cap, btn_clear = st.columns(2)
            with btn_cap:
                if st.button("📸 Capture & analyse", key="live_capture_btn", disabled=not playing):
                    n_faces = len([lab for lab in live_state.read()["labels"] if lab != "Side view"])
                    # one face: average a burst of the sharpest recent frames; a group: one sharpest frame
                    frames = live_state.take_best(1 if n_faces > 1 else LIVE_BURST)
                    encoded = []
                    for fr in frames:
                        ok, buf = cv2.imencode(
                            ".png", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1]
                        )
                        if ok:
                            encoded.append(buf.tobytes())
                    if encoded:
                        st.session_state.live_snapshots = encoded
                    else:
                        st.warning("No video frame received yet. Wait a second and try again.")
            with btn_clear:
                if st.session_state.get("live_snapshots"):
                    if st.button("🗑️ Discard capture", key="live_discard_btn"):
                        st.session_state.live_snapshots = None
                        st.rerun()
            if st.session_state.get("live_snapshots"):
                n_snap = len(st.session_state.live_snapshots)
                st.caption(
                    f"Below: the full analysis of your last capture ({n_snap} sharpest frames averaged)."
                    if n_snap > 1
                    else "Below: the full analysis of your last capture (sharpest recent frame)."
                )

        with col_read:
            with st.container(key="live_panel"):
                st.markdown(
                    '<div class="ambient-glow-text" style="font-weight:800; font-size:1.05rem; margin-bottom:8px;">'
                    'Live read</div>',
                    unsafe_allow_html=True,
                )
                (live_readout_running if playing else live_readout_static)(live_state, playing)
                st.markdown(
                    '<div class="live-note">Live preview: fast models, tracked and smoothed per face. '
                    'Press <b>Capture and analyse</b> for the full analysis: it keeps the sharpest recent frames, '
                    'runs every model and crop, and averages them, then opens the coach and history.</div>',
                    unsafe_allow_html=True,
                )

        snaps = st.session_state.get("live_snapshots")
        if snaps:
            photos = [(SnapshotFile(b, f"live_frame_{i + 1}.png"), True) for i, b in enumerate(snaps)]
            burst_mode = len(photos) > 1
    else:
        _, cam_col, _ = st.columns([1, 2.4, 1])
        with cam_col:
            camera_photo = st.camera_input("Capture Portrait", label_visibility="collapsed", key="cam_input_feed")
            if camera_photo is not None:
                photos = [(camera_photo, True)]
else:
    uploaded_files = st.file_uploader(
        f"Upload an image (JPG, PNG). Add up to {MAX_PHOTOS} photos of the same person to average them.",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="file_upload_feed",
    )
    if uploaded_files:
        if len(uploaded_files) > MAX_PHOTOS:
            st.warning(f"Only the first {MAX_PHOTOS} photos are used.")
        photos = [(f, False) for f in uploaded_files[:MAX_PHOTOS]]

# Several photos: average them (same person) or look at one photo at a time
average_mode = False
if len(photos) > 1 and burst_mode:
    average_mode = True  # a live burst is one person: average it
elif len(photos) > 1:
    view_choice = st.radio(
        "How should these photos be used?",
        ["🧮 Average all (same person)", "🖼️ View one photo"],
        horizontal=True,
        key="multi_view_mode",
    )
    average_mode = view_choice.startswith("🧮")
    if not average_mode:
        pick = st.selectbox(
            "Photo",
            list(range(len(photos))),
            format_func=lambda i: f"Photo {i + 1}: {photos[i][0].name}",
            key="single_photo_pick",
        )
        photos = [photos[pick]]

# =============================================================================
# INFERENCE PIPELINE
# =============================================================================

if not photos:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 35px; border-radius: 24px; border: 2px dashed {capsule_border}; box-shadow: {capsule_glow}; margin-top: 15px;">
            <div style="font-size: 2.4rem; margin-bottom: 8px;"><span class="ambient-bubble">🫧</span></div>
            <div class="ambient-glow-text" style="font-weight: 800; font-size: 1.25rem;">Awaiting Facial Capture</div>
            <div style="font-size: 0.85rem; color: {text_muted}; margin-top: 6px;">Glide your cursor to swim through the water, or capture a photo to begin.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    ready = False
    view_mode = "single"

    # Filled by whichever branch runs below
    primary_cls = None
    img_hash = ""
    source_label = ""
    latency = 0.0
    stat_items = []
    extra_note = ""
    rendered_frame = None
    face_list_html = ""
    gallery = []
    quality_target = None  # (image, region) for photo-quality hints

    if average_mode:
        # ---------------- MULTI-PHOTO AVERAGING ----------------
        per_photo = []
        skipped = 0
        total_latency = 0.0

        for n_idx, (f_obj, is_cam) in enumerate(photos):
            img = load_rgb(f_obj, is_cam)
            d = detect_and_score(img)
            total_latency += d["latency"]
            if d["err"] or not d["faces"]:
                skipped += 1
                continue
            face = max(d["faces"], key=face_area)  # same person: use the largest face
            per_photo.append({
                "idx": n_idx + 1,
                "image": img,
                "bytes": d["bytes"],
                "face": face,
                "boxed": bool(d["boxes"]),
                "cls": classify_emotions(face["emotion"]),
            })

        if not per_photo:
            st.error("Inference Alert: No clear face found in any of the photos. Use well-lit photos facing the camera.")
        else:
            ready = True
            view_mode = "average"
            k = len(per_photo)

            if skipped:
                st.warning(f"{skipped} of {len(photos)} photos had no readable face and were skipped.")

            avg_scores = {
                e: float(np.mean([p["cls"]["scores"][e] for p in per_photo]))
                for e in EMOJI_MAP
            }
            primary_cls = classify_emotions(avg_scores)
            agree = sum(1 for p in per_photo if p["cls"]["dominant"] == primary_cls["dominant"])

            img_hash = hashlib.md5(
                "".join(hashlib.md5(p["bytes"]).hexdigest() for p in per_photo).encode()
            ).hexdigest()
            source_label = f"Live burst ({k} frames averaged)" if burst_mode else f"Average of {k} photos"
            latency = total_latency
            extra_note = f"Averaged over {k} photo{'s' if k != 1 else ''}: {agree}/{k} agree on {primary_cls['dominant']}"
            stat_items = [
                ("Photos used", f"{k}/{len(photos)}"),
                ("Latency", f"{latency:.0f} ms"),
                ("Agreement", f"{agree}/{k}"),
            ]
            for p in per_photo:
                frame = draw_clean_bounding(
                    p["image"], [p["face"]] if p["boxed"] else [], is_dark, [p["cls"]["short"]]
                )
                gallery.append((frame, f"Photo {p['idx']}: {p['cls']['short']}"))
            biggest = max(per_photo, key=lambda p: face_area(p["face"]))
            quality_target = (biggest["image"], biggest["face"]["region"])

    else:
        # ---------------- SINGLE PHOTO (ONE OR MANY FACES) ----------------
        file_obj, is_cam = photos[0]
        image_np = load_rgb(file_obj, is_cam)
        d = detect_and_score(image_np)

        if d["err"]:
            st.error(f"Inference Alert: {d['err']}")
        elif not d["faces"]:
            st.warning("No face clearly recognized. Center your face in good lighting and retake.")
        else:
            ready = True
            faces = d["faces"]
            face_cls = [classify_emotions(f["emotion"]) for f in faces]
            img_hash = hashlib.md5(d["bytes"]).hexdigest()

            focus_idx = int(np.argmax([face_area(f) for f in faces]))
            if len(faces) > 1:
                focus_idx = st.selectbox(
                    "🎯 Focus face (drives the coach, history and bars)",
                    list(range(len(faces))),
                    index=focus_idx,
                    format_func=lambda i: f"Face {i + 1}: {face_cls[i]['short']}",
                    key=f"focus_{img_hash[:8]}",
                )

            primary_cls = face_cls[focus_idx]
            latency = d["latency"]
            labels = [c["short"] for c in face_cls] if len(faces) > 1 else None
            rendered_frame = (
                draw_clean_bounding(image_np, d["boxes"], is_dark, labels, focus_idx)
                if d["boxes"] else image_np
            )

            if len(faces) > 1:
                source_label = f"Group photo ({len(faces)} faces, face {focus_idx + 1})"
                face_list_html = "".join(
                    f'<div class="face-row"><span class="raw-emoji">{c["emoji"]}</span>'
                    f'<strong>Face {i + 1}</strong>: {c["title"]} ({c["conf"]:.0f}%)'
                    + ('<span class="face-focus">focus</span>' if i == focus_idx else "")
                    + '</div>'
                    for i, c in enumerate(face_cls)
                )
            else:
                source_label = "Single photo"

            num_faces_label = str(len(d["boxes"])) if d["boxes"] else "1 (Portrait)"
            stat_items = [
                ("Faces", num_faces_label),
                ("Latency", f"{latency:.0f} ms"),
                ("Resolution", f"{d['W']}x{d['H']}"),
            ]
            quality_target = (image_np, faces[focus_idx]["region"])

    # ---------------------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------------------
    if ready:
        # Photo quality hints (blur, darkness, small face, uneven light)
        if quality_target is not None:
            for note in assess_quality(quality_target[0], quality_target[1]):
                st.info(note)

        calibrated_emotions = primary_cls["scores"]
        dominant_label = primary_cls["dominant"]
        dominant_conf = primary_cls["conf"]
        second_label = primary_cls["second"].capitalize()
        runner_up_conf = primary_cls["second_conf"]
        is_ambiguous = primary_cls["ambiguous"]
        is_mixed = primary_cls["mixed"]
        classification_title = primary_cls["title"]
        classification_emoji = primary_cls["emoji"]

        # Short description of the camera read, passed to the coach as a soft hint
        emotion_context = (
            f"{dominant_label} {dominant_conf:.0f}%, then {second_label.lower()} {runner_up_conf:.0f}%"
            + (" (mixed or unclear)" if is_ambiguous else "")
        )

        # Log to session history (one row per reading). Changing the focus face
        # of the same photo updates that row instead of adding a new one.
        cache_key = f"{img_hash}|{source_label}"
        if st.session_state.cached_analysis != cache_key:
            same_photo = (
                st.session_state.cached_analysis is not None
                and st.session_state.cached_analysis.split("|")[0] == img_hash
                and len(st.session_state.history) > 0
            )
            st.session_state.cached_analysis = cache_key
            entry = {
                "Timestamp": time.strftime("%H:%M:%S"),
                "Classification": classification_title,
                "Confidence": f"{dominant_conf:.1f}%",
                "User_Reflection": "None",
                "Source": source_label,
                **{k.capitalize(): f"{float(v):.1f}%" for k, v in calibrated_emotions.items()},
            }
            if same_photo:
                entry["User_Reflection"] = st.session_state.history[-1].get("User_Reflection", "None")
                st.session_state.history[-1] = entry
            else:
                st.session_state.history.append(entry)

            if st.session_state.last_emotion_seen != dominant_label:
                st.session_state.last_emotion_seen = dominant_label
                st.session_state.chat_messages = [
                    {"role": "ai", "content": get_initial_ai_greeting(
                        dominant_label, dominant_conf, second_label.lower(), runner_up_conf, is_mixed, is_ambiguous
                    )}
                ]

        # -------------------------------------------------------------
        # 1. OPTICAL FEED & GRAPH
        # -------------------------------------------------------------
        col_img, col_graph = st.columns([1.05, 1.0], gap="medium")

        with col_img:
            with st.container(key="feed_panel"):
                if view_mode == "average":
                    st.markdown(
                        '<div class="feed-header-title">Photos Used (largest face in each)</div>',
                        unsafe_allow_html=True,
                    )
                    per_row = 3
                    for start in range(0, len(gallery), per_row):
                        row_items = gallery[start:start + per_row]
                        g_cols = st.columns(per_row)
                        for g_col, (frame, cap) in zip(g_cols, row_items):
                            with g_col:
                                st.image(frame, caption=cap, use_container_width=True)
                else:
                    st.markdown(
                        '<div class="feed-header-title">Optical Feed Result</div>',
                        unsafe_allow_html=True,
                    )
                    st.image(rendered_frame, use_container_width=True)
                    if face_list_html:
                        st.markdown(face_list_html, unsafe_allow_html=True)

                stat_html = "".join(
                    f'<div class="stat-item"><div class="stat-label">{lbl}</div><div class="stat-value">{val}</div></div>'
                    for lbl, val in stat_items
                )
                st.markdown(f'<div class="stat-row">{stat_html}</div>', unsafe_allow_html=True)

        with col_graph:
            display_title = classification_title
            display_emoji = classification_emoji

            # Extra info lines under the badge. Hide the runner-up line when it is basically
            # zero (under 5%). Never leave a blank line here: a blank line would break the
            # HTML block in markdown.
            also_parts = []
            if runner_up_conf >= 5:
                also_parts.append(
                    f'<div style="font-size:0.78rem; color:{text_muted}; margin-top:4px;">'
                    f'Also detected: {second_label} ({runner_up_conf:.1f}%)</div>'
                )
            if extra_note:
                also_parts.append(
                    f'<div style="font-size:0.78rem; color:{text_muted}; margin-top:4px;">{html_lib.escape(extra_note)}</div>'
                )
            also_html = "".join(also_parts) if also_parts else "<div></div>"

            st.markdown(
                f"""
                <div class="metric-badge">
                    <div style="font-size:0.7rem; font-weight:700; text-transform:uppercase; color:{text_muted};">
                        Primary Classification
                    </div>
                    <div class="badge-main">
                        <span class="raw-emoji">{display_emoji}</span>
                        <span class="shimmer-txt">{display_title}</span>
                        <span class="shimmer-txt" style="font-size:0.95rem; font-weight:700;">({dominant_conf:.1f}%)</span>
                    </div>
                    {also_html}
                    <div style="height: 8px; width: 100%; background: rgba(51, 65, 85, 0.6); border-radius: 999px; overflow: hidden; margin-top: 12px; border: 1px solid rgba(56, 189, 248, 0.2);">
                        <div style="height: 100%; width: {min(dominant_conf, 100):.1f}%; background: linear-gradient(90deg, #0284C7, {accent_tint}); border-radius: 999px; box-shadow: 0 0 12px {accent_tint};"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.68rem; color: {text_muted}; font-weight: 700;">
                        <span>Confidence: {dominant_conf:.1f}%</span>
                        <span>Gate: 35%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.container(key="bars_panel"):
                render_live_animated_bars(calibrated_emotions, dominant_label, is_dark)

        # -------------------------------------------------------------
        # 2. WELLNESS COACH
        # -------------------------------------------------------------
        with st.container(key="coach_panel"):
            st.markdown(
                f"""
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
                        <span style="font-size: 1.35rem;">🌿</span>
                        <div class="ambient-glow-text" style="font-weight: 800; font-size: 1.15rem;">
                            Grounded Mental Wellness Coach
                        </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            for msg in st.session_state.chat_messages:
                if msg["role"] == "ai":
                    st.markdown(
                        f'<div class="chat-bubble-ai"><strong>Coach:</strong> {render_chat_html(msg["content"])}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chat-bubble-user"><strong>You:</strong> {render_chat_html(msg["content"])}</div>',
                        unsafe_allow_html=True,
                    )

            presets = EMOTION_PRESETS.get(dominant_label, EMOTION_PRESETS["neutral"])
            st.markdown(
                f'<div class="preset-title">🌱 Check In (Or Type Below):</div>',
                unsafe_allow_html=True,
            )

            p_cols = st.columns(3)
            selected_preset = None

            for p_idx, preset_text in enumerate(presets):
                with p_cols[p_idx]:
                    if st.button(preset_text, key=f"preset_btn_primary_{p_idx}"):
                        selected_preset = preset_text

            if selected_preset:
                st.session_state.chat_messages.append({"role": "user", "content": selected_preset})
                if len(st.session_state.history) > 0:
                    st.session_state.history[-1]["User_Reflection"] = selected_preset
                bot_reply = generate_companion_reply(
                    selected_preset,
                    dominant_label,
                    history=st.session_state.chat_messages[:-1],
                    emotion_context=emotion_context,
                )
                st.session_state.chat_messages.append({"role": "ai", "content": bot_reply})
                if dominant_label in ["happy", "surprise"]:
                    st.session_state.trigger_celebration = True
                st.rerun()

            with st.form("wellness_coach_chat_form", clear_on_submit=True):
                chat_col_input, chat_col_btn = st.columns([4.2, 1.1])
                with chat_col_input:
                    user_chat_text = st.text_input(
                        "Message Coach",
                        placeholder="Check in or share what's on your mind...",
                        label_visibility="collapsed",
                    )
                with chat_col_btn:
                    submitted = st.form_submit_button("Reflect 🌿")
                    if submitted:
                        text_to_send = user_chat_text.strip()
                        if text_to_send:
                            st.session_state.chat_messages.append({"role": "user", "content": text_to_send})
                            crisis_flag = is_crisis_message(text_to_send)
                            if len(st.session_state.history) > 0:
                                st.session_state.history[-1]["User_Reflection"] = (
                                    "[private message]" if crisis_flag else text_to_send
                                )
                            bot_reply = generate_companion_reply(
                                text_to_send,
                                dominant_label,
                                history=st.session_state.chat_messages[:-1],
                                emotion_context=emotion_context,
                            )
                            st.session_state.chat_messages.append({"role": "ai", "content": bot_reply})
                            if dominant_label in ["happy", "surprise"] and not crisis_flag:
                                st.session_state.trigger_celebration = True
                            st.rerun()

            with st.expander("🧰 1-minute reset for this moment"):
                reset_title, reset_steps = EMOTION_RESETS.get(dominant_label, EMOTION_RESETS["neutral"])
                st.markdown(f"**{reset_title}**")
                st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(reset_steps, 1)))
                render_breathing_widget(BREATHING_PATTERN.get(dominant_label, "box"), is_dark)

            st.markdown(
                f'<div style="font-size:0.72rem; color:{text_muted}; margin-top:10px; line-height:1.5;">{COACH_NOTE}</div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# SESSION INSIGHTS + HISTORY
# =============================================================================

if len(st.session_state.history) > 0:
    st.markdown("---")

    render_session_insights(st.session_state.history, is_dark)

    col_title, col_actions = st.columns([2.4, 2.6])
    with col_title:
        st.markdown('<div class="session-title ambient-glow-text">Session Expression History</div>', unsafe_allow_html=True)
    with col_actions:
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            df_history = pd.DataFrame(st.session_state.history)
            st.download_button(
                label="📥 Export CSV",
                data=df_history.to_csv(index=False),
                file_name="facial_emotion_session_report.csv",
                mime="text/csv",
                key="dl_btn_history_csv",
            )
        with btn_col2:
            if st.button("🗑️ Clear Logs", key="clear_logs_btn"):
                st.session_state.history = []
                st.session_state.cached_analysis = None
                st.session_state.chat_messages = []
                st.session_state.last_emotion_seen = None
                st.rerun()

    timeline = " → ".join(
        EMOJI_MAP.get(h["Classification"].lower().split(" ")[0], "🤔")
        for h in st.session_state.history[-10:]
    )
    st.markdown(
        f'<div style="margin: 6px 0 12px 0; font-size:0.8rem; color:{text_muted}; font-weight:700;">'
        f'Mood timeline: <span class="raw-emoji" style="font-size:1.15rem;">{timeline}</span></div>',
        unsafe_allow_html=True,
    )

    for item in st.session_state.history[-5:][::-1]:
        classification = item["Classification"]
        reflection = item.get("User_Reflection", "None")
        emoji = EMOJI_MAP.get(classification.lower().split(" ")[0], "🤔")
        source_txt = item.get("Source", "")
        time_line = item["Timestamp"] + (f" · {html_lib.escape(source_txt)}" if source_txt else "")
        reflection_markup = (
            f'<div style="font-size:0.75rem; color:{accent_tint}; margin-top:2px;">Check-In: "{html_lib.escape(reflection)}"</div>'
            if reflection != "None"
            else ""
        )
        card_html = (
            f'<div class="history-item">'
            f'<div style="display:flex; align-items:center; gap:10px;">'
            f'<span class="raw-emoji" style="font-size:1.3rem;">{emoji}</span>'
            f'<div>'
            f'<div class="ambient-glow-text" style="font-weight:800; font-size:0.92rem;">{classification}</div>'
            f'<div style="font-size:0.7rem; color:#94A3B8;">{time_line}</div>'
            f'{reflection_markup}'
            f'</div>'
            f'</div>'
            f'<div class="ambient-glow-text" style="font-weight:800; font-size:0.92rem;">{item["Confidence"]}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)


