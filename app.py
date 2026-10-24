import io
import time
import json
import os
import hashlib
import base64
import tempfile
from pathlib import Path
import html as html_lib
import re

# DeepFace needs the legacy Keras (tf-keras) on TensorFlow 2.16+. Set it BEFORE TensorFlow is imported,
# and never read a variable that may not exist (that raised a KeyError on Streamlit Cloud).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st
import streamlit.components.v1 as components
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
# LIVE CAMERA (runs in the visitor's browser: no extra Python packages needed)
# =============================================================================

# The live video, face boxes and emotion bars run inside the browser (face-api.js), so nothing is
# streamed to the server while you watch. Pressing "Take photo" sends a few frames to Python, which
# runs the full, more accurate analysis on them. If the browser cannot load the live engine, the
# camera and Take photo still work.
LIVE_CAM_HTML = r'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --card: rgba(6, 17, 36, 0.88); --border: rgba(56, 189, 248, 0.65); --text: #FFFFFF; --muted: #94A3B8; --accent: #38BDF8; }
  * { box-sizing: border-box; margin: 0; padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif;
      -webkit-font-smoothing: antialiased; }
  html, body { background: transparent; color: var(--text); overflow: hidden; }
  #wrap { display: flex; flex-direction: column; gap: 12px; padding: 3px; }
  #stageBox { position: relative; border: 1.5px solid var(--border); border-radius: 24px; overflow: hidden; background: #000;
              box-shadow: 0 0 20px rgba(56, 189, 248, 0.25); }
  #stage { position: relative; width: 100%; aspect-ratio: 4 / 3; }
  video, canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
  video { object-fit: fill; transform: scaleX(-1); background: #000; }
  #flash { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: #fff; opacity: 0; pointer-events: none; transition: opacity 0.35s ease-out; }
  #status { position: absolute; left: 12px; right: 12px; bottom: 12px; padding: 8px 14px; border-radius: 9999px; font-size: 0.82rem;
            font-weight: 700; text-align: center; background: var(--card); border: 1px solid var(--border); color: var(--text); }
  #status.err { border-color: #F87171; }
  #status.hide { display: none; }
  #panel { display: none; }
  #btnRow { display: flex; justify-content: center; }
  #shot { min-width: 220px; padding: 14px 34px; border-radius: 9999px; border: 2px solid var(--border); background: var(--card);
          color: var(--text); font-size: 1.05rem; font-weight: 800; cursor: pointer; box-shadow: 0 0 18px rgba(56, 189, 248, 0.35);
          -webkit-tap-highlight-color: transparent; transition: transform 0.15s ease, box-shadow 0.15s ease; }
  #shot:active { transform: scale(0.96); box-shadow: 0 0 28px rgba(56, 189, 248, 0.6); }
  #head { font-weight: 800; font-size: 1.02rem; margin-bottom: 8px; color: var(--accent); }
  #top { font-size: 1.35rem; font-weight: 800; margin-bottom: 10px; min-height: 1.8rem; }
  #top small { font-size: 0.8rem; color: var(--muted); font-weight: 700; }
  .row { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
  .lab { width: 104px; font-size: 0.8rem; font-weight: 700; text-align: right; white-space: nowrap; }
  .track { flex: 1; height: 10px; border-radius: 999px; background: rgba(51, 65, 85, 0.45); overflow: hidden; border: 1px solid var(--border); }
  .fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #0284C7, var(--accent)); width: 1.5%; transition: width 0.25s ease; }
  .val { width: 52px; font-size: 0.8rem; font-weight: 800; font-variant-numeric: tabular-nums; }
  #note { font-size: 0.74rem; color: var(--muted); margin-top: 10px; line-height: 1.5; }
  #empty { padding: 14px 2px; font-size: 0.9rem; color: var(--muted); }
</style>
</head>
<body>
<div id="wrap">
  <div id="stageBox">
    <div id="stage">
      <video id="v" autoplay muted playsinline></video>
      <canvas id="c"></canvas>
      <div id="flash"></div>
      <div id="status">Starting camera...</div>
    </div>
  </div>
  <div id="btnRow"><button id="shot" type="button">&#128248; Take photo</button></div>
  <div id="panel">
    <div id="head">Live read</div>
    <div id="top"></div>
    <div id="bars"><div id="empty">Waiting for the camera...</div></div>
    <div id="note">Live preview runs on your device. Press <b>Take photo</b> for the full, more accurate analysis, coach and history.</div>
  </div>
</div>

<script>
(function () {
  // ----- where the live emotion engine (face-api.js) and its small models come from -----
  var SCRIPT_URLS = [
    "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js",
    "https://unpkg.com/face-api.js@0.22.2/dist/face-api.min.js"
  ];
  var MODEL_URLS = [
    "https://justadudewhohacks.github.io/face-api.js/models",
    "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights",
    "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights"
  ];
  var MAX_FACES = 4;

  var EMOJI = { angry: "😠", disgust: "🤢", fear: "😨", happy: "😊", sad: "😔", surprise: "😲", neutral: "😐" };
  var MAP = { angry: "angry", disgusted: "disgust", fearful: "fear", happy: "happy", sad: "sad", surprised: "surprise", neutral: "neutral" };

  function $(id) { return document.getElementById(id); }
  var video = $("v"), canvas = $("c"), g = canvas.getContext("2d");
  var args = __ARGS__;
  var tracks = [], lastUpdate = 0;
  var camReady = false, engineReady = false, usingCpu = false;

  var sleep = function (ms) { return new Promise(function (r) { setTimeout(r, ms); }); };
  function setHeight() {
    try {
      var box = $("stageBox"), ratio = video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : 4 / 3;
      var avail = $("wrap").clientWidth;
      box.style.width = Math.min(avail, Math.round(440 * ratio)) + "px";   // never taller than ~440px
      box.style.margin = "0 auto";
    } catch (e) {}
  }
  function applyTheme(a) {
    var s = document.documentElement.style;
    if (a.card) s.setProperty("--card", a.card);
    if (a.border) s.setProperty("--border", a.border);
    if (a.text) s.setProperty("--text", a.text);
    if (a.muted) s.setProperty("--muted", a.muted);
    if (a.accent) s.setProperty("--accent", a.accent);
  }
  applyTheme(args);
  if (window.ResizeObserver) new ResizeObserver(setHeight).observe($("wrap"));

  // Hand the captured frames to the app: they are placed in its hidden file box, exactly as if chosen by hand.
  function deliver(frames) {
    try {
      var input = window.parent.document.querySelector('.st-key-live_capture_box input[type="file"]');
      if (!input) return false;
      var dt = new DataTransfer();
      frames.forEach(function (u, i) {
        var bin = atob(u.split(",")[1]), arr = new Uint8Array(bin.length);
        for (var j = 0; j < bin.length; j++) arr[j] = bin.charCodeAt(j);
        dt.items.add(new File([arr], "live_frame_" + (i + 1) + ".jpg", { type: "image/jpeg" }));
      });
      input.files = dt.files;
      input.dispatchEvent(new Event("change", { bubbles: true }));
      return true;
    } catch (e) { return false; }
  }

  // ----- status pill -----
  var statusTimer = null;
  function setStatus(text, isErr) {
    var el = $("status");
    el.textContent = text || "";
    el.className = (isErr ? "err" : "") + (text ? "" : " hide");
  }

  // ----- classification rules (same as the full analysis) -----
  function classify(scores) {
    var ranked = Object.keys(scores).map(function (k) { return [k, scores[k]]; }).sort(function (a, b) { return b[1] - a[1]; });
    var dom = ranked[0][0], domConf = ranked[0][1], sec = ranked[1][0], secConf = ranked[1][1];
    var margin = domConf - secConf;
    var cap = function (s) { return s.charAt(0).toUpperCase() + s.slice(1); };
    var mixed = margin < 10 && (domConf + secConf) >= 70;
    var ambiguous = domConf < 35 || margin < 10;
    if (mixed) return { title: cap(dom) + " + " + cap(sec), emoji: EMOJI[dom] + EMOJI[sec], short: cap(dom) + "+" + cap(sec), conf: domConf };
    if (ambiguous) return { title: "Ambiguous / Inconclusive", emoji: "🤔", short: "Unclear", conf: domConf };
    return { title: cap(dom), emoji: EMOJI[dom], short: cap(dom) + " " + Math.round(domConf) + "%", conf: domConf };
  }
  function toScores(ex) {
    var raw = {}, tot = 0, k;
    for (k in MAP) { var v = Number(ex[k]) || 0; raw[MAP[k]] = v; tot += v; }
    var out = {};
    for (k in raw) out[k] = raw[k] / (tot || 1) * 100;
    return out;
  }

  // ----- tracking + smoothing (time based, quicker for a brand-new face) -----
  function iou(a, b) {
    var iw = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    var ih = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    var inter = iw * ih, uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0 ? inter / uni : 0;
  }
  function updateTracks(dets, now) {
    var dt = lastUpdate ? Math.min(Math.max(now - lastUpdate, 0.02), 1.0) : 0.2;
    lastUpdate = now;
    var aBox = 1 - Math.exp(-dt / 0.15);
    var free = tracks.slice();
    dets.sort(function (a, b) { return b.box.w * b.box.h - a.box.w * a.box.h; });
    dets.slice(0, MAX_FACES).forEach(function (d) {
      var best = null, bi = 0.3;
      free.forEach(function (t) { var v = iou(t.box, d.box); if (v > bi) { best = t; bi = v; } });
      if (!best) {
        tracks.push({ box: { x: d.box.x, y: d.box.y, w: d.box.w, h: d.box.h }, scores: d.scores, seen: now, born: now });
      } else {
        free.splice(free.indexOf(best), 1);
        ["x", "y", "w", "h"].forEach(function (k) { best.box[k] = (1 - aBox) * best.box[k] + aBox * d.box[k]; });
        var tau = (now - best.born < 1.0) ? 0.12 : 0.45;
        var a = 1 - Math.exp(-dt / tau);
        for (var k in d.scores) {
          var prev = best.scores[k] === undefined ? d.scores[k] : best.scores[k];
          best.scores[k] = (1 - a) * prev + a * d.scores[k];
        }
        best.seen = now;
      }
    });
    tracks = tracks.filter(function (t) { return now - t.seen <= 1.0; });
  }
  function liveTracks(now) {
    var vw = video.videoWidth || 1;
    return tracks.filter(function (t) { return now - t.seen <= 0.6; })
      .sort(function (a, b) { return (vw - (a.box.x + a.box.w)) - (vw - (b.box.x + b.box.w)); });  // left to right as you see it
  }

  // ----- drawing (the video is mirrored by CSS, so boxes are mirrored here too) -----
  function draw(now) {
    var vw = video.videoWidth, vh = video.videoHeight;
    if (!vw) return;
    if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }
    g.clearRect(0, 0, vw, vh);
    var live = liveTracks(now), multi = live.length > 1;
    var mainIdx = -1, bestA = 0;
    live.forEach(function (t, i) { var a = t.box.w * t.box.h; if (a > bestA) { bestA = a; mainIdx = i; } });
    var dark = args.dark !== false;
    var boxCol = dark ? "#7DD3FC" : "#0284C7", focusCol = dark ? "#C084FC" : "#7C3AED", txtCol = dark ? "#0F172A" : "#FFFFFF";
    var fs = Math.max(16, Math.round(vw / 30));
    g.font = "700 " + fs + "px -apple-system, BlinkMacSystemFont, Roboto, sans-serif";
    g.textBaseline = "middle";
    live.forEach(function (t, i) {
      var w = t.box.w, h = t.box.h, x = vw - (t.box.x + w), y = t.box.y;
      var col = (multi && i === mainIdx) ? focusCol : boxCol;
      var L = Math.max(Math.round(Math.min(w, h) * 0.18), 14);
      var lw = Math.max(3, Math.round(vw / 110));
      g.strokeStyle = col; g.lineWidth = (multi && i === mainIdx) ? Math.round(lw * 1.3) : lw; g.lineCap = "round";
      [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]].forEach(function (c) {
        g.beginPath(); g.moveTo(c[0] + c[2] * L, c[1]); g.lineTo(c[0], c[1]); g.lineTo(c[0], c[1] + c[3] * L); g.stroke();
      });
      var text = classify(t.scores).short;
      if (multi) text = (i + 1) + ": " + text;
      var tw = g.measureText(text).width, pad = 6, bh = fs + 2 * pad;
      var bx = Math.max(0, Math.min(x, vw - tw - 2 * pad - 1));
      var by = y - bh - 4; if (by < 0) by = y + 4;
      g.fillStyle = col; g.fillRect(bx, by, tw + 2 * pad, bh);
      g.fillStyle = txtCol; g.fillText(text, bx + pad, by + bh / 2);
    });
  }

  // ----- side panel: all seven emotions of the largest face -----
  function updatePanel(now) {
    var live = liveTracks(now);
    if (!live.length) {
      $("top").innerHTML = "";
      $("bars").innerHTML = '<div id="empty">Looking for a face... face the camera in good light.</div>';
      setStatus(engineReady ? "Looking for a face..." : "", false);
      return;
    }
    setStatus("", false);
    var main = live.reduce(function (a, b) { return (b.box.w * b.box.h > a.box.w * a.box.h) ? b : a; });
    var c = classify(main.scores);
    $("top").innerHTML = c.emoji + " " + c.title + " <small>(" + Math.round(c.conf) + "%)</small>";
    var ranked = Object.keys(main.scores).map(function (k) { return [k, main.scores[k]]; }).sort(function (a, b) { return b[1] - a[1]; });
    var html = ranked.map(function (r) {
      var name = r[0].charAt(0).toUpperCase() + r[0].slice(1);
      return '<div class="row"><div class="lab">' + EMOJI[r[0]] + " " + name + '</div><div class="track"><div class="fill" style="width:' +
        Math.max(r[1], 1.5).toFixed(1) + '%"></div></div><div class="val">' + r[1].toFixed(1) + "%</div></div>";
    }).join("");
    if (live.length > 1) html += '<div id="note" style="margin-top:8px">' + live.length + " faces in view. The bars show the largest face.</div>";
    $("bars").innerHTML = html;
  }

  // ----- camera -----
  async function startCamera() {
    setStatus("Starting camera...", false);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("This browser cannot use the camera here (it needs a secure https page).", true);
      return false;
    }
    try {
      var stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false
      });
      video.srcObject = stream;
      try { await video.play(); } catch (e) {}
      await new Promise(function (r) { if (video.videoWidth) r(); else video.onloadedmetadata = function () { r(); }; });
      $("stage").style.aspectRatio = video.videoWidth + " / " + video.videoHeight;
      camReady = true;
      setHeight();
      return true;
    } catch (e) {
      setStatus("Camera blocked or unavailable. Allow camera access for this site, then reload.", true);
      return false;
    }
  }

  // ----- live emotion engine (optional: Take photo works even if this cannot load) -----
  function loadScript(url) {
    return new Promise(function (res, rej) {
      var s = document.createElement("script");
      s.src = url; s.onload = res; s.onerror = function () { rej(new Error("script")); };
      document.head.appendChild(s);
    });
  }
  async function loadEngine() {
    for (var i = 0; i < SCRIPT_URLS.length && !window.faceapi; i++) {
      try { await loadScript(SCRIPT_URLS[i]); } catch (e) {}
    }
    if (!window.faceapi) throw new Error("engine script");
    for (var j = 0; j < MODEL_URLS.length; j++) {
      try {
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URLS[j]);
        await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URLS[j]);
        return;
      } catch (e) {}
    }
    throw new Error("models");
  }
  async function useCpu() {
    if (usingCpu || !window.faceapi || !faceapi.tf) return false;
    try { usingCpu = true; await faceapi.tf.setBackend("cpu"); await faceapi.tf.ready(); return true; } catch (e) { return false; }
  }
  async function loop() {
    var opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.45 });
    var lastPanel = 0;
    while (true) {
      if (document.hidden || video.readyState < 2) { await sleep(150); continue; }
      try {
        var res = await faceapi.detectAllFaces(video, opts).withFaceExpressions();
        var now = performance.now() / 1000;
        var dets = res.map(function (r) {
          return { box: { x: r.detection.box.x, y: r.detection.box.y, w: r.detection.box.width, h: r.detection.box.height }, scores: toScores(r.expressions) };
        });
        updateTracks(dets, now);
        draw(now);
        if (now - lastPanel > 0.08) { updatePanel(now); lastPanel = now; }
      } catch (e) {
        if (!(await useCpu())) { setStatus("Live emotion stopped. You can still press Take photo.", true); return; }
      }
      await new Promise(function (r) { requestAnimationFrame(r); });
    }
  }

  // ----- Take photo: grab frames exactly as the camera sends them (not mirrored) -----
  function grab() {
    var vw = video.videoWidth, vh = video.videoHeight, s = Math.min(1, 1280 / vw);
    var c = document.createElement("canvas");
    c.width = Math.round(vw * s); c.height = Math.round(vh * s);
    c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.92);
  }
  var busy = false;
  async function capture() {
    if (busy) return;
    busy = true;
    if (!camReady) { setStatus("The camera is not ready yet.", true); return; }
    var now = performance.now() / 1000, n = liveTracks(now).length;
    var count = 1;   // one photo is enough: the analysis already shows every emotion as a percentage
    var f = $("flash"); f.style.transition = "none"; f.style.opacity = "0.85";
    setTimeout(function () { f.style.transition = "opacity 0.35s ease-out"; f.style.opacity = "0"; }, 30);
    setStatus("Hold still...", false);
    var frames = [];
    for (var i = 0; i < count; i++) { frames.push(grab()); if (i < count - 1) await sleep(110); }
    var ok = deliver(frames);
    setStatus(ok ? "Photo taken. Analysing below..." : "Could not pass the photo to the app. Reload the page and try again.", !ok);
    busy = false;
    clearTimeout(statusTimer);
    statusTimer = setTimeout(function () { setStatus("", false); }, 2500);
  }

  $("shot").addEventListener("click", function () { capture(); });
  setHeight();

  // ----- go -----
  (async function main() {
    if (!(await startCamera())) return;
    setStatus("Loading emotion engine...", false);
    try {
      await loadEngine();
      engineReady = true;
      setStatus("Looking for a face...", false);
      loop();
    } catch (e) {
      setStatus("Live emotion could not load. Take photo still works.", true);
    }
  })();
})();
</script>
</body>
</html>
'''


class SnapshotFile:
    """File-like wrapper so a live capture goes through the same pipeline as an upload."""

    def __init__(self, data: bytes, name: str = "live_capture.jpg"):
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data


def live_cam_html(theme: dict) -> str:
    """The camera page with the app's colours filled in."""
    return LIVE_CAM_HTML.replace("__ARGS__", json.dumps(theme))


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

if is_dark:
    app_bg = "radial-gradient(circle at 50% 12%, #081b33 0%, #030a16 50%, #01040a 100%)"
    card_bg = "rgba(6, 17, 36, 0.88)"
    capsule_border = "rgba(56, 189, 248, 0.65)"
    capsule_glow = "0 0 20px rgba(56, 189, 248, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.25)"
    banner_halo = "0 0 30px 4px rgba(37, 99, 235, 0.8), 0 0 65px 12px rgba(56, 189, 248, 0.4)"
    text_main = "#FFFFFF"
    text_muted = "#94A3B8"
    accent_tint = "#38BDF8"
    media_solid = "#071a30"
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
    media_solid = "#EAF5FE"
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

.st-key-live_capture_box {{
    position: absolute !important;
    left: -9999px !important;
    top: 0;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0;
}}

.st-key-media_zone {{
    background: {media_solid};
    border-radius: 26px;
    padding: 0.5rem;
    position: relative;
    z-index: 3;
    margin-bottom: 0.8rem;
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
    }}, 120);

    if (parentDoc) {{
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

_, col_dark, col_light = st.columns([6.4, 1.8, 1.8])

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
    st.session_state.setdefault("live_reset", 0)

    with st.container(key="media_zone"):
        components.html(
            live_cam_html(
                dict(dark=bool(is_dark), card=card_bg, border=capsule_border, text=text_main, muted=text_muted, accent=accent_tint)
            ),
            height=560,
            scrolling=False,
        )

    # The camera page drops its captured frames into this box (hidden by CSS); the app reads them from here.
    with st.container(key="live_capture_box"):
        live_files = st.file_uploader(
            "Live capture",
            type=["jpg", "jpeg"],
            accept_multiple_files=True,
            key=f"live_capture_{st.session_state.live_reset}",
            label_visibility="collapsed",
        )

    if live_files:
        _, discard_col, _ = st.columns([1, 2, 1])
        with discard_col:
            if st.button("🗑️ Discard photo", key="live_discard_btn"):
                st.session_state.live_reset += 1
                st.rerun()
        photos = [(f, True) for f in live_files[:MAX_PHOTOS]]
        burst_mode = len(photos) > 1
        st.caption(
            f"Full analysis of your photo ({len(photos)} frames averaged for accuracy)."
            if burst_mode
            else "Full analysis of your photo."
        )
else:
    with st.container(key="media_zone"):
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

