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

# Custom Theme with CSS Animations & Enhanced Mobile Uploader
st.markdown("""
<style>
    /* Dark Slate Background */
    .stApp {
        background: linear-gradient(135deg, #070B19 0%, #0F172A 50%, #1E1B4B 100%);
        color: #F8FAFC;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes floatBounce {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-8px) rotate(8deg); }
    }

    @keyframes pulseGlow {
        0%, 100% { filter: drop-shadow(0 0 6px rgba(56, 189, 248, 0.4)); }
        50% { filter: drop-shadow(0 0 16px rgba(129, 140, 248, 0.8)); }
    }

    .animated-title-box {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }

    .animated-emoji {
        font-size: 2.8rem;
        display: inline-block;
        animation: floatBounce 2.5s ease-in-out infinite, pulseGlow 2.5s ease-in-out infinite;
    }

    .animated-gradient-text {
        font-size: 2.3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC, #F472B6, #38BDF8);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 5s ease infinite;
        letter-spacing: -0.5px;
    }

    .header-sub {
        font-size: 1rem;
        color: #94A3B8;
        margin-bottom: 1.8rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #1E293B !important;
        border: 2px dashed #38BDF8 !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(90deg, #38BDF8, #818CF8) !important;
        color: #0F172A !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploaderDropzone"] button p {
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    .result-badge {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.9);
        backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
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
    <span class="animated-gradient-text">AI Facial Expression Analytics</span>
</div>
<div class="header-sub">Deep learning real-time classification across standard affect categories • 😡 🤢 😨 😄 😢 😲 😐</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Select or Drag Photo to Analyze:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    
    # Auto-orient the image based on phone EXIF data
    raw_image = ImageOps.exif_transpose(raw_image)
    
    image_np = np.array(raw_image.convert('RGB'))
    display_image = image_np.copy()

    with st.spinner("Analyzing facial geometry and affective cues..."):
        try:
            results = DeepFace.analyze(
                img_path=image_np,
                actions=['emotion'],
                enforce_detection=False
            )

            # Filter background artifacts
            valid_results = [
                r for r in results 
                if r['region']['w'] > 60 and r['region']['h'] > 60
            ]

            if not valid_results:
                st.warning("No face detected with sufficient clarity. Try another image.")
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
                    tab_ai, tab_orig = st.tabs(["⚡ AI Output", "📷 Original Photo"])
                    with tab_ai:
                        st.image(display_image, use_container_width=True)
                    with tab_orig:
                        st.image(raw_image, use_container_width=True)

                with col_metrics:
                    st.subheader("Classification Confidence")
                    for idx, res in enumerate(valid_results):
                        dominant_raw = res['dominant_emotion'].lower()
                        dominant_emoji = EMOJI_MAP.get(dominant_raw, "🙂")
                        dominant_name = dominant_raw.capitalize()
                        scores = res['emotion']

                        st.markdown(f"""
                        <div class="result-badge">
                            <span style="font-size: 0.85rem; color: #38BDF8; font-weight: 700; letter-spacing: 0.5px;">DETECTED SUBJECT #{idx + 1}</span>
                            <div class="dominant-text">{dominant_emoji} {dominant_name} ({scores[dominant_raw]:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)

                        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                        for emo_name, score in sorted_scores:
                            emo_key = emo_name.lower()
                            icon = EMOJI_MAP.get(emo_key, "")
                            st.markdown(f'<div class="progress-label">{icon} <strong>{emo_name.capitalize()}</strong> — {score:.1f}%</div>', unsafe_allow_html=True)
                            st.progress(min(int(score), 100))

                        st.markdown("<br>", unsafe_allow_html=True)

        except Exception as err:
            st.error(f"Inference error: {err}")
