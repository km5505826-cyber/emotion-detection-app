import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
from deepface import DeepFace

# Page Setup
st.set_page_config(
    page_title="AI Facial Expression Analytics",
    page_icon="🧠",
    layout="wide"
)

# Emoji mapping for the 7 standard facial emotions
EMOJI_MAP = {
    "angry": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

# Custom Theme Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #070B19 0%, #0F172A 50%, #1E1B4B 100%);
        color: #F8FAFC;
    }
    .animated-title-box {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }
    .animated-emoji {
        font-size: 2.8rem;
    }
    .title-text {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .header-sub {
        font-size: 1rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .result-badge {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.9);
        backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .dominant-text {
        font-size: 1.55rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
    }
    .progress-label {
        font-size: 0.95rem;
        color: #CBD5E1;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="animated-title-box">
    <span class="animated-emoji">🎭</span>
    <span class="title-text">AI Facial Expression Analytics</span>
</div>
<div class="header-sub">Real-time facial capture & emotion recognition (Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust)</div>
""", unsafe_allow_html=True)

# Direct Webcam Input (Replaces File Uploader)
camera_photo = st.camera_input("📸 Take a photo using your webcam")

if camera_photo is not None:
    raw_image = Image.open(camera_photo)
    raw_image = ImageOps.exif_transpose(raw_image)
    image_np = np.array(raw_image.convert('RGB'))
    display_image = image_np.copy()

    with st.spinner("Detecting face and analyzing affective cues..."):
        try:
            results = DeepFace.analyze(
                img_path=image_np,
                actions=['emotion'],
                enforce_detection=False
            )

            valid_results = [
                r for r in results 
                if r['region']['w'] > 60 and r['region']['h'] > 60
            ]

            if not valid_results:
                st.warning("No face detected with sufficient clarity. Please retake the photo.")
            else:
                col_img, col_metrics = st.columns([1.1, 0.9], gap="large")

                for idx, res in enumerate(valid_results):
                    region = res['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    dominant_raw = res['dominant_emotion'].lower()
                    dominant_name = dominant_raw.capitalize()

                    cv2.rectangle(display_image, (x, y), (x + w, y + h), (56, 189, 248), 3)
                    label = f"Face #{idx + 1}: {dominant_name}"
                    cv2.rectangle(display_image, (x, max(y - 32, 0)), (x + len(label) * 13, y), (56, 189, 248), -1)
                    cv2.putText(display_image, label, (x + 6, max(y - 10, 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 23, 42), 2)

                with col_img:
                    st.subheader("Visual Detection")
                    st.image(display_image, use_container_width=True)

                with col_metrics:
                    st.subheader("Classification Confidence")
                    for idx, res in enumerate(valid_results):
                        dominant_raw = res['dominant_emotion'].lower()
                        dominant_emoji = EMOJI_MAP.get(dominant_raw, "🙂")
                        dominant_name = dominant_raw.capitalize()
                        scores = res['emotion']

                        st.markdown(f"""
                        <div class="result-badge">
                            <span style="font-size: 0.85rem; color: #38BDF8; font-weight: 700;">DETECTED SUBJECT #{idx + 1}</span>
                            <div class="dominant-text">{dominant_emoji} {dominant_name} ({scores[dominant_raw]:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)

                        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                        for emo_name, score in sorted_scores:
                            emo_key = emo_name.lower()
                            icon = EMOJI_MAP.get(emo_key, "")
                            st.markdown(f'<div class="progress-label">{icon} <strong>{emo_name.capitalize()}</strong> — {score:.1f}%</div>', unsafe_allow_html=True)
                            st.progress(min(int(score), 100))

        except Exception as err:
            st.error(f"Inference error: {err}")
