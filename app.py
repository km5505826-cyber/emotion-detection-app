import io
import hashlib
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from deepface import DeepFace

# ============================================================
# PAGE + UNDERWATER THEME (keeps the project's visual identity)
# ============================================================
st.set_page_config(
    page_title="Facial Emotion Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { color-scheme: dark; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(ellipse at 18% 0%, rgba(20,104,150,.35), transparent 42%),
    radial-gradient(ellipse at 85% 20%, rgba(0,190,210,.12), transparent 35%),
    linear-gradient(155deg, #12304a 0%, #071522 44%, #030a12 75%, #02060b 100%);
  color: #eaf7ff;
}
[data-testid="stHeader"] { background: rgba(2,10,18,.25); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(9,32,49,.96), rgba(3,12,21,.98));
  border-right: 1px solid rgba(110,210,255,.18);
}
.block-container { padding-top: 1.6rem; max-width: 1450px; }
.hero {
  padding: 1.3rem 1.5rem; border-radius: 24px; margin-bottom: 1.25rem;
  background: linear-gradient(120deg, rgba(20,105,150,.32), rgba(10,34,52,.64));
  border: 1px solid rgba(126,220,255,.25);
  box-shadow: 0 12px 45px rgba(0,0,0,.16), inset 0 1px rgba(255,255,255,.06);
}
.hero h1 { margin: 0; color: #e8f8ff; letter-spacing: -.04em; font-size: clamp(1.7rem, 4vw, 2.7rem); }
.hero p { color: #a9d2e7; margin: .55rem 0 0; }
.glass {
  background: linear-gradient(145deg, rgba(18,54,76,.66), rgba(5,21,34,.78));
  border: 1px solid rgba(120,210,245,.17);
  border-radius: 20px; padding: 1rem 1.1rem; margin-bottom: .85rem;
  box-shadow: inset 0 1px rgba(255,255,255,.04);
}
.kpi { font-size: 1.7rem; font-weight: 800; color: #e8f8ff; }
.muted { color: #9fc3d7; font-size: .9rem; }
.pill {
 display:inline-block; border:1px solid rgba(110,220,255,.35);
 border-radius:999px; padding:.28rem .7rem; color:#bfeeff;
 background:rgba(35,145,190,.13); font-size:.78rem;
}
div[data-testid="stMetric"] {
  background: rgba(10,35,52,.62); border:1px solid rgba(120,210,245,.16);
  padding: .85rem; border-radius: 16px;
}
.stButton > button, .stDownloadButton > button {
  border-radius: 12px; border: 1px solid rgba(100,205,245,.35);
  background: linear-gradient(110deg, #12658a, #0b425f); color: white;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: #83e0ff; color: white; background: #145a78;
}
div[data-testid="stFileUploader"] {
  background: rgba(8,31,47,.48); border-radius: 16px;
  border: 1px dashed rgba(112,206,242,.3); padding: .5rem;
}
hr { border-color: rgba(130,210,245,.16); }
footer { visibility: hidden; }
@media (prefers-reduced-motion: no-preference) {
  .hero { animation: riseIn .65s ease-out both; }
  @keyframes riseIn { from {opacity:0; transform:translateY(8px)} to {opacity:1; transform:translateY(0)} }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <span class="pill">UNDERWATER AI LAB · COMPUTER VISION</span>
  <h1>🌊 Facial Emotion Intelligence</h1>
  <p>Face detection and facial-expression analysis, with clearer confidence reporting and safer handling of uncertain predictions.</p>
</div>
""", unsafe_allow_html=True)

EMOTION_LABELS = {
    "angry": "Angry", "disgust": "Disgust", "fear": "Fear",
    "happy": "Happy", "sad": "Sad", "surprise": "Surprise",
    "neutral": "Neutral"
}
EMOTION_COLORS = {
    "happy": "#38d9a9", "sad": "#74b9ff", "angry": "#ff7675",
    "fear": "#b197fc", "surprise": "#ffd166", "disgust": "#a8d672",
    "neutral": "#b2c6d6"
}

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "last_analysis_id" not in st.session_state:
    st.session_state.last_analysis_id = None

# ----------------------- Helpers ----------------------------
def load_image(uploaded_file):
    """Read an uploaded image, apply EXIF orientation, return RGB PIL + BGR ndarray."""
    pil_img = Image.open(uploaded_file)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    rgb = np.asarray(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return pil_img, bgr

def safe_float(value, default=0.0):
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default

def normalize_emotions(raw):
    """Normalize DeepFace emotion scores into a 0-100 distribution."""
    scores = {str(k).lower(): max(0.0, safe_float(v)) for k, v in (raw or {}).items()}
    scores = {k: v for k, v in scores.items() if k in EMOTION_LABELS}
    total = sum(scores.values())
    if not scores:
        return {}
    if total > 0 and total > 1.5:  # DeepFace commonly returns percentage values
        scores = {k: v / total * 100 for k, v in scores.items()}
    elif total > 0:                # Some versions/models may return fractions
        scores = {k: v / total * 100 for k, v in scores.items()}
    return scores

def analyze_image(image_bgr, detector_backend, min_face_size):
    """
    DeepFace's pretrained emotion classifier is used for inference.
    RetinaFace usually gives stronger face localization than the basic OpenCV
    detector, but is slower. No synthetic rule forces a label to 'happy'/'sad'.
    """
    results = DeepFace.analyze(
        img_path=image_bgr,
        actions=["emotion"],
        detector_backend=detector_backend,
        enforce_detection=True,
        align=True,
        silent=True,
    )
    if isinstance(results, dict):
        results = [results]

    cleaned = []
    h, w = image_bgr.shape[:2]
    for result in results:
        area = result.get("region") or result.get("facial_area") or {}
        x = int(area.get("x", 0)); y = int(area.get("y", 0))
        fw = int(area.get("w", area.get("width", 0)))
        fh = int(area.get("h", area.get("height", 0)))
        if fw < min_face_size or fh < min_face_size:
            continue
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + fw), min(h, y + fh)
        if x2 <= x1 or y2 <= y1:
            continue

        scores = normalize_emotions(result.get("emotion", {}))
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        cleaned.append({
            "label": top_label,
            "score": top_score,
            "margin": top_score - second_score,
            "scores": scores,
            "box": (x1, y1, x2, y2),
            "face_area": fw * fh,
        })
    return sorted(cleaned, key=lambda item: item["face_area"], reverse=True)

def annotate_image(image_rgb, faces):
    canvas = image_rgb.copy()
    for index, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = face["box"]
        key = face["label"]
        color_hex = EMOTION_COLORS.get(key, "#8be9fd")
        color = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        text_label = f'{index}. {EMOTION_LABELS.get(key, key.title())} · {face["score"]:.0f}%'
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.62, 2
        (tw, th), baseline = cv2.getTextSize(text_label, font, scale, thickness)
        ty = max(th + 10, y1 - 8)
        cv2.rectangle(canvas, (x1, ty - th - 8), (min(canvas.shape[1]-1, x1 + tw + 10), ty + baseline - 3), color, -1)
        cv2.putText(canvas, text_label, (x1 + 5, ty - 4), font, scale, (4, 15, 23), thickness, cv2.LINE_AA)
    return canvas

def confidence_note(face, min_score, min_margin):
    if face["score"] < min_score:
        return "Uncertain result", "The leading emotion score is low. Try a clearer, front-facing image with better lighting."
    if face["margin"] < min_margin:
        return "Mixed / ambiguous expression", "Two or more emotion scores are close, so the model cannot confidently separate them."
    return EMOTION_LABELS.get(face["label"], face["label"].title()), "This is the model's best facial-expression estimate, not a confirmed statement about how the person feels."

# ---------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ Analysis settings")
    detector = st.selectbox(
        "Face detector",
        ["retinaface", "opencv", "mediapipe", "ssd"],
        index=0,
        help="RetinaFace is the recommended starting point for face localization. OpenCV is lighter/faster."
    )
    min_face_size = st.slider("Ignore faces smaller than (px)", 30, 160, 55, 5)
    min_score = st.slider("Minimum leading score to show a clear label", 20, 70, 38, 1)
    min_margin = st.slider("Minimum gap from second-place emotion", 0, 25, 5, 1)
    st.caption("These settings control detection and uncertainty messaging. They do not retrain the model or increase its underlying accuracy.")
    st.markdown("---")
    st.markdown("### Tips for better predictions")
    st.markdown("""
- Use a sharp, front-facing photo.
- Keep the face large enough in the frame.
- Prefer even light; avoid strong backlighting.
- Avoid filters, masks, extreme angles, and motion blur.
- For camera input, hold still and use a natural expression.
""")
    st.markdown("---")
    st.caption("Expression estimates can be wrong and should not be used for medical, hiring, security, or other high-stakes decisions.")

# ------------------------ Main UI ---------------------------
tab_upload, tab_camera, tab_history = st.tabs(["🫧 Upload image", "📷 Camera snapshot", "📊 History"])

def run_analysis(pil_img, image_bgr, source_name):
    image_rgb = np.asarray(pil_img.convert("RGB"))
    digest = hashlib.sha256(image_rgb.tobytes()).hexdigest()
    analysis_id = f"{digest}:{detector}:{min_face_size}:{min_score}:{min_margin}"
    if st.session_state.last_analysis_id == analysis_id:
        st.info("This exact image and these settings were already analyzed. Change a setting or upload a different image to run it again.")
        return
    try:
        with st.spinner("Finding faces and analyzing expressions… first run may take a while while AI models load."):
            faces = analyze_image(image_bgr, detector, min_face_size)
        st.session_state.last_analysis_id = analysis_id
    except Exception as exc:
        st.error("The analysis failed. Try another detector or use a clearer image.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")
        st.info("If the error mentions a missing package/model, install the dependencies from requirements.txt and restart Streamlit.")
        return

    if not faces:
        st.warning("No usable face was detected. Try a closer, brighter, front-facing image. If using RetinaFace, try OpenCV as a fallback.")
        return

    annotated = annotate_image(image_rgb, faces)
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="glass"><b>Face detection output</b><div class="muted">Bounding boxes show detected face regions.</div></div>', unsafe_allow_html=True)
        st.image(annotated, use_container_width=True)
        st.download_button(
            "⬇️ Download annotated image",
            data=Image.fromarray(annotated).tobytes() if False else image_to_png_bytes(annotated),
            file_name="emotion_analysis.png",
            mime="image/png",
            key=f"download_{analysis_id[:12]}",
        )
    with right:
        st.markdown('<div class="glass"><b>Analysis summary</b></div>', unsafe_allow_html=True)
        st.metric("Faces detected", len(faces))
        counts = {}
        for f in faces:
            counts[f["label"]] = counts.get(f["label"], 0) + 1
        for i, face in enumerate(faces, start=1):
            label, note = confidence_note(face, min_score, min_margin)
            actual = EMOTION_LABELS.get(face["label"], face["label"].title())
            color = EMOTION_COLORS.get(face["label"], "#8be9fd")
            st.markdown(f"""
            <div class="glass">
              <span class="pill">FACE {i}</span>
              <h3 style="margin:.55rem 0 .2rem;color:{color}">{label}</h3>
              <div class="muted">Top model label: {actual} · Leading score: {face['score']:.1f}% · Lead over next: {face['margin']:.1f} points</div>
              <p style="margin:.55rem 0 0">{note}</p>
            </div>
            """, unsafe_allow_html=True)
            score_rows = [{"Emotion": EMOTION_LABELS.get(k, k.title()), "Score (%)": round(v, 2)} for k, v in sorted(face["scores"].items(), key=lambda kv: kv[1], reverse=True)]
            st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)
            chart_df = pd.DataFrame(score_rows).sort_values("Score (%)", ascending=True)
            st.bar_chart(chart_df.set_index("Emotion"), x_label="Score (%)", horizontal=True, use_container_width=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, face in enumerate(faces, start=1):
        st.session_state.analysis_history.append({
            "Time": now, "Source": source_name, "Face": i,
            "Predicted label": EMOTION_LABELS.get(face["label"], face["label"].title()),
            "Top score (%)": round(face["score"], 2),
            "Lead (%)": round(face["margin"], 2),
            "Detector": detector,
            **{f"{EMOTION_LABELS.get(k, k.title())} (%)": round(v, 2) for k, v in face["scores"].items()}
        })
    st.success("Analysis complete. Remember: facial-expression recognition is an estimate and can misclassify subtle or mixed expressions.")

def image_to_png_bytes(image_array):
    output = io.BytesIO()
    Image.fromarray(image_array).save(output, format="PNG")
    return output.getvalue()

with tab_upload:
    st.markdown('<div class="glass"><b>Upload a photo</b><div class="muted">JPG, JPEG, PNG, and WEBP are supported.</div></div>', unsafe_allow_html=True)
    upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"], key="upload_img")
    if upload is not None:
        try:
            pil_img, bgr_img = load_image(upload)
            st.image(pil_img, caption="Image selected for analysis", use_container_width=True)
            if st.button("🔎 Analyze facial expressions", type="primary", key="analyze_upload"):
                run_analysis(pil_img, bgr_img, upload.name)
        except Exception as exc:
            st.error(f"Could not open this image: {exc}")

with tab_camera:
    st.markdown('<div class="glass"><b>Take a camera snapshot</b><div class="muted">This analyzes a single captured frame, not continuous live video.</div></div>', unsafe_allow_html=True)
    camera_file = st.camera_input("Capture a photo", key="camera_snapshot")
    if camera_file is not None:
        try:
            pil_img, bgr_img = load_image(camera_file)
            if st.button("🔎 Analyze camera snapshot", type="primary", key="analyze_camera"):
                run_analysis(pil_img, bgr_img, "Camera snapshot")
        except Exception as exc:
            st.error(f"Could not read the camera image: {exc}")

with tab_history:
    st.markdown('<div class="glass"><b>Previous analyses</b><div class="muted">History is kept for this session only.</div></div>', unsafe_allow_html=True)
    if st.session_state.analysis_history:
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Export history as CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="emotion_analysis_history.csv",
            mime="text/csv",
        )
        if st.button("Clear session history"):
            st.session_state.analysis_history = []
            st.session_state.last_analysis_id = None
            st.rerun()
    else:
        st.info("No analyses yet. Upload a photo or capture a camera snapshot to get started.")

st.markdown("---")
st.markdown(
    '<p class="muted" style="text-align:center">Facial Emotion Intelligence · Underwater UI · AI predictions are estimates, not facts about a person’s internal emotional state.</p>',
    unsafe_allow_html=True
)
