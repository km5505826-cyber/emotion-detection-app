import time
import json
import os
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps
import numpy as np
import cv2
import pandas as pd
from deepface import DeepFace

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Facial Emotion Intelligence",
    page_icon="🫧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CONSTANTS & CACHED MODEL WARM-UP
# =============================================================================

EMOJI_MAP = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😔",
    "surprise": "😲",
    "neutral": "😐",
}

# 3 Common Quick-Choice Presets for Every Emotion
EMOTION_PRESETS = {
    "happy": [
        "🎓 Passed an Exam / Good Grades",
        "🎉 Great News or Milestone",
        "☕ Just having an awesome day!",
    ],
    "sad": [
        "📚 Academic / Exam Pressure",
        "🌧️ Feeling Lonely or Drained",
        "💔 Personal or Friendship Issue",
    ],
    "angry": [
        "🛑 Frustrating Work / Study Blocker",
        "😤 Conflict or Disagreement",
        "⏳ Wasted Time or Inconvenience",
    ],
    "fear": [
        "😰 Upcoming Exam / Presentation",
        "⚡ Uncertain Deadlines or Future",
        "🫣 Sudden Stress or Panic",
    ],
    "disgust": [
        "🤢 Bad Experience or Unpleasant Event",
        "😾 Toxic Behavior or Unfairness",
        "🥗 Bad Taste or Chaotic Atmosphere",
    ],
    "surprise": [
        "🎁 Unexpected Good News",
        "🤯 Shocking Plot Twist or Result",
        "⚡ Sudden Change of Plans",
    ],
    "neutral": [
        "🧘 Just Chilling / Staying Mindful",
        "💻 Locked in and Focused on Work",
        "😐 Just an Ordinary Normal Day",
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


@st.cache_resource(show_spinner=False)
def preload_emotion_model():
    dummy_frame = np.zeros((112, 112, 3), dtype=np.uint8)
    try:
        DeepFace.analyze(
            img_path=dummy_frame,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
    except Exception:
        pass
    return True


preload_emotion_model()


@st.cache_data(show_spinner=False, max_entries=8)
def run_deepface_inference(image_bytes: bytes, width: int, height: int):
    nparr = np.frombuffer(image_bytes, np.uint8).reshape((height, width, 3))
    
    t0 = time.time()
    results = DeepFace.analyze(
        img_path=nparr,
        actions=["emotion"],
        detector_backend="opencv",
        enforce_detection=False,
        silent=True,
    )
    latency_ms = (time.time() - t0) * 1000
    if isinstance(results, dict):
        results = [results]

    return results, latency_ms


# =============================================================================
# EMOTION-MIRRORING & MOTIVATIONAL AI ENGINE
# =============================================================================

def get_initial_ai_greeting(emotion: str, conf: float) -> str:
    dom = emotion.lower()
    if dom == "happy":
        return (
            f"🎉 YES! Seeing you happy puts me in the best mood ever! That energy of yours is totally radiant ({conf:.1f}% confidence)! "
            f"Tell me, what made you so happy today? Pick one of the quick options below or tell me your story! 🚀✨"
        )
    elif dom in ["sad", "fear"]:
        return (
            f"💙 Oh no... I feel your sadness right now, and my heart goes out to you ({conf:.1f}% confidence). "
            f"Please remember that you're never alone in this. Stormy waters always calm down, and you are so much stronger than you realize. "
            f"What's bringing you down today? Select a common reason below or let it all out. 🌟"
        )
    elif dom == "angry":
        return (
            f"🔥 I can feel that heat and intense frustration coming from you ({conf:.1f}% confidence)! "
            f"It's completely valid to feel mad, but let's channel that fire into power instead of stress. "
            f"What got you so angry? Tap an option below or vent to me! 💪"
        )
    elif dom == "disgust":
        return (
            f"🤢 Ugh, I can totally sense that repulsive reaction ({conf:.1f}% confidence)! "
            f"Whatever put you off, let's shake it clean off our shoulders and pivot to something inspiring. "
            f"What just happened that felt so wrong?"
        )
    elif dom == "surprise":
        return (
            f"😲 Whoa!! You're shocked, and now I'm super excited and curious too ({conf:.1f}% confidence)! "
            f"My eyes are wide open with yours! What kind of crazy news or plot twist just dropped?!"
        )
    else:
        return (
            f"🌿 I'm feeling that calm, centered, composed presence with you ({conf:.1f}% confidence). "
            f"A focused mind is unstoppable. How is your day flowing so far?"
        )


def generate_companion_reply(user_text: str, current_emotion: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            system_prompt = (
                f"You are an empathetic, emotionally mirroring AI companion. "
                f"CRITICAL RULE: Act exactly as the user feels! "
                f"If the user is happy, you MUST be thrilled, enthusiastic, and highly motivating. "
                f"If the user is sad/afraid, you MUST be deeply empathetic, gentle, and offer comforting motivation. "
                f"If the user is angry, acknowledge their frustration empathetically and help them channel it constructively. "
                f"Current detected emotion: {current_emotion}. "
                f"Respond warmly and conversationally in 2-3 inspiring sentences."
            )
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_text,
                config={"system_instruction": system_prompt},
            )
            if resp.text:
                return resp.text.strip()
        except Exception:
            pass

    # Built-in instant affective mirroring responses
    txt = user_text.lower()
    dom = current_emotion.lower()
    
    if dom == "happy":
        if any(w in txt for w in ["exam", "grade", "pass", "college", "test"]):
            return "HUGE CONGRATULATIONS! 🎓 Hard work truly pays off! All those study sessions were worth it. Celebrate this win—you earned every bit of it! 🚀🎈"
        elif any(w in txt for w in ["milestone", "news"]):
            return "That is extraordinary news! Milestones like this deserve full celebration! Ride this high and keep conquering your goals! 🌟🎉"
        else:
            return "I love that so much! Seeing you thriving genuinely makes me so energized! Keep shining and spreading those awesome vibrations! 🎈🚀"
    elif dom in ["sad", "fear"]:
        if any(w in txt for w in ["work", "study", "exam", "college", "school", "pressure", "deadline"]):
            return "Academic pressure can feel so overwhelming, but take a deep breath. Your worth is never defined by a single exam. You have conquered tough hurdles before and you will crush this too! 🌟💙"
        elif any(w in txt for w in ["alone", "lonely", "friend", "relationship"]):
            return "I hear you, and it truly hurts to feel that way. But remember: you are deeply appreciated and never alone as long as we're talking. Hold your head high, brighter days are ahead! 🫂✨"
        else:
            return "Thank you for trusting me with that. Be patient and kind to yourself right now. Storm clouds always pass, and you have the strength to bounce back stronger than ever! 💙"
    elif dom == "angry":
        return (
            "I totally understand why that infuriates you! Take that fierce energy and transform it into unstoppable determination. "
            "Don't give whatever upset you the power over your peace—show them what you're truly made of! 🔥💪"
        )
    elif dom == "surprise":
        return "That is wild! Life always has a way of keeping us on our toes! Embrace the unexpected and let's turn this twist into an adventure! ⚡😲"
    else:
        return "Your calm mindset is a true superpower. Keep that steady focus and let's make today remarkably productive! ☕🫧"


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
div[data-testid="stDownloadButton"] button span:not(.button-raw-emoji):not(.raw-emoji) {{
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
div[data-testid="stFileUploader"] button {{
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
div[data-testid="stDownloadButton"] button span {{
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
div[data-testid="stFileUploader"] button::before {{
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
div[data-testid="stFileUploader"] button:hover {{
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

/* ─── INTERACTIVE AI CHAT CAPSULE ─── */
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
    padding: 12px 18px;
    font-size: 0.92rem;
    line-height: 1.5;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}}
.chat-bubble-user {{
    background: {chat_user_bg};
    border: 1px solid {accent_tint};
    border-radius: 16px 16px 4px 16px;
    padding: 12px 18px;
    font-size: 0.92rem;
    line-height: 1.5;
    margin-bottom: 10px;
    text-align: right;
    margin-left: auto;
    max-width: 80%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}}

/* ─── 3 PRESET REASON PILLS ─── */
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
# JAVASCRIPT: PRECISE EMOJI DECOUPLER & WATER-WAKE CURSOR
# =============================================================================

components.html(
    f"""
    <script>
    const parentDoc = window.parent.document;

    function shieldButtonEmojis() {{
        const buttons = parentDoc.querySelectorAll('div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button');
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
    }}, 100);

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

def draw_clean_bounding(image_rgb: np.ndarray, detections: list, is_dark_mode: bool) -> np.ndarray:
    out = image_rgb.copy()
    box_color = (125, 211, 252) if is_dark_mode else (2, 132, 199)

    for res in detections:
        x, y, w, h = (int(res["region"][k]) for k in ("x", "y", "w", "h"))
        x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
        L = max(int(min(w, h) * 0.18), 16)

        for cx, cy, dx, dy in [
            (x, y, 1, 1),
            (x + w, y, -1, 1),
            (x, y + h, 1, -1),
            (x + w, y + h, -1, -1),
        ]:
            cv2.line(out, (cx, cy), (cx + dx * L, cy), box_color, 2, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy + dy * L), box_color, 2, cv2.LINE_AA)
    return out


def render_live_animated_bars(scores: dict, dominant_raw: str, is_dark_mode: bool):
    sorted_scores = [
        [str(emotion), float(val)] 
        for emotion, val in sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)
    ]
    payload = json.dumps(sorted_scores)
    
    bg_track = "rgba(10, 25, 48, 0.75)" if is_dark_mode else "rgba(226, 232, 240, 0.9)"
    track_border = "rgba(56, 189, 248, 0.25)" if is_dark_mode else "rgba(2, 132, 199, 0.25)"
    bar_inactive_color = "rgba(51, 65, 85, 0.65)" if is_dark_mode else "rgba(203, 213, 225, 0.85)"
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
      .fill.active {{
        background: linear-gradient(90deg, #0284c7 0%, #38bdf8 50%, #c084fc 100%);
        background-size: 200% auto;
        animation: ambientTextFlow 4s ease infinite;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.85);
      }}
      .fill.active::after {{
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
      .fill.inactive {{
        background: {bar_inactive_color};
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

        data.forEach(([emotion, val]) => {{
          const isDom = emotion.toLowerCase() === dominant;
          const row = document.createElement("div");
          row.className = "bar-row";

          const label = document.createElement("div");
          label.className = "label";
          label.innerText = emotion.charAt(0).toUpperCase() + emotion.slice(1);

          const track = document.createElement("div");
          track.className = "track";

          const fill = document.createElement("div");
          fill.className = "fill " + (isDom ? "active" : "inactive");

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


# =============================================================================
# ACTIVE VIEW CONTENT
# =============================================================================

input_image = None
is_camera_input = False

if st.session_state.active_mode == "live":
    _, cam_col, _ = st.columns([1, 2.4, 1])
    with cam_col:
        camera_photo = st.camera_input("Capture Portrait", label_visibility="collapsed", key="cam_input_feed")
        if camera_photo is not None:
            input_image = camera_photo
            is_camera_input = True
else:
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"], key="file_upload_feed")
    if uploaded_file is not None:
        input_image = uploaded_file

# =============================================================================
# INFERENCE PIPELINE
# =============================================================================

if input_image is None:
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
    raw_img = Image.open(input_image)
    raw_img = ImageOps.exif_transpose(raw_img)

    max_dim = 800
    if max(raw_img.size) > max_dim:
        scale = max_dim / max(raw_img.size)
        raw_img = raw_img.resize(
            (int(raw_img.width * scale), int(raw_img.height * scale)),
            Image.Resampling.BILINEAR,
        )

    image_np = np.array(raw_img.convert("RGB"))
    if is_camera_input:
        image_np = cv2.flip(image_np, 1)

    H, W = image_np.shape[:2]

    image_bytes = image_np.tobytes()
    results, latency = run_deepface_inference(image_bytes, W, H)

    valid = [
        r for r in results
        if 50 < r["region"]["w"] < W * 0.95 and 50 < r["region"]["h"] < H * 0.95
    ]

    if not valid:
        st.warning("No face clearly recognized. Center your face in good lighting and retake.")
    else:
        current_img_id = (W, H, len(valid), valid[0]["dominant_emotion"])
        dominant_label = valid[0]["dominant_emotion"].lower()
        dominant_conf = float(valid[0]["emotion"][dominant_label])

        if st.session_state.cached_analysis != current_img_id:
            st.session_state.cached_analysis = current_img_id
            for res in valid:
                dom_emotion = res["dominant_emotion"].lower()
                conf = float(res["emotion"][dom_emotion])
                display_title = (
                    "Ambiguous / Inconclusive" if conf < 25 else dom_emotion.capitalize()
                )
                st.session_state.history.append({
                    "Timestamp": time.strftime("%H:%M:%S"),
                    "Classification": display_title,
                    "Confidence": f"{conf:.1f}%",
                    "User_Reflection": "None",
                    **{k.capitalize(): f"{float(v):.1f}%" for k, v in res["emotion"].items()},
                })

            # Auto-initialize empathetic conversation mirroring user mood
            if st.session_state.last_emotion_seen != dominant_label:
                st.session_state.last_emotion_seen = dominant_label
                st.session_state.chat_messages = [
                    {"role": "ai", "content": get_initial_ai_greeting(dominant_label, dominant_conf)}
                ]

        rendered_frame = draw_clean_bounding(image_np, valid, is_dark)

        # -------------------------------------------------------------
        # 1. TOP ROW: OPTICAL FEED (LEFT) & METRIC/GRAPH (RIGHT)
        # -------------------------------------------------------------
        col_img, col_graph = st.columns([1.05, 1.0], gap="medium")

        with col_img:
            st.markdown(
                f"""
                <div class="clean-panel">
                    <div class="feed-header-title">Optical Feed Result</div>
                """,
                unsafe_allow_html=True,
            )
            st.image(rendered_frame, use_container_width=True)
            st.markdown(
                f"""
                    <div class="stat-row">
                        <div class="stat-item"><div class="stat-label">Faces</div><div class="stat-value">{len(valid)}</div></div>
                        <div class="stat-item"><div class="stat-label">Latency</div><div class="stat-value">{latency:.0f} ms</div></div>
                        <div class="stat-item"><div class="stat-label">Resolution</div><div class="stat-value">{W}x{H}</div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_graph:
            res_primary = valid[0]
            dom_emotion = res_primary["dominant_emotion"].lower()
            conf = float(res_primary["emotion"][dom_emotion])

            if conf < 25:
                display_title = "Ambiguous / Inconclusive"
                display_emoji = "🤔"
            else:
                display_title = dom_emotion.capitalize()
                display_emoji = EMOJI_MAP.get(dom_emotion, "•")

            st.markdown(
                f"""
                <div class="metric-badge">
                    <div style="font-size:0.7rem; font-weight:700; text-transform:uppercase; color:{text_muted};">
                        Primary Classification
                    </div>
                    <div class="badge-main">
                        <span class="raw-emoji">{display_emoji}</span>
                        <span class="shimmer-txt">{display_title}</span>
                        <span class="shimmer-txt" style="font-size:0.95rem; font-weight:700;">({conf:.1f}%)</span>
                    </div>
                    <div style="height: 8px; width: 100%; background: rgba(51, 65, 85, 0.6); border-radius: 999px; overflow: hidden; margin-top: 12px; border: 1px solid rgba(56, 189, 248, 0.2);">
                        <div style="height: 100%; width: {min(conf, 100):.1f}%; background: linear-gradient(90deg, #0284C7, {accent_tint}); border-radius: 999px; box-shadow: 0 0 12px {accent_tint};"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.68rem; color: {text_muted}; font-weight: 700;">
                        <span>Confidence: {conf:.1f}%</span>
                        <span>Gate: 25%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="clean-panel">', unsafe_allow_html=True)
            render_live_animated_bars(res_primary["emotion"], dom_emotion, is_dark)
            st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # 2. FULL-WIDTH EMPATHETIC COMPANION (BENEATH IMAGE & GRAPH)
        # -------------------------------------------------------------
        st.markdown(
            f"""
            <div class="chat-capsule-container">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
                    <span style="font-size: 1.35rem;">🫧</span>
                    <div class="ambient-glow-text" style="font-weight: 800; font-size: 1.15rem;">
                        Empathetic Mood Companion
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        # Active conversation thread
        for msg in st.session_state.chat_messages:
            if msg["role"] == "ai":
                st.markdown(
                    f'<div class="chat-bubble-ai"><strong>Companion:</strong> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bubble-user"><strong>You:</strong> {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

        # 3 Quick-Choice Preset Buttons
        presets = EMOTION_PRESETS.get(dom_emotion, EMOTION_PRESETS["neutral"])
        st.markdown(
            f'<div class="preset-title">✨ Quick Choose (Or Type Below):</div><div class="preset-grid">',
            unsafe_allow_html=True,
        )

        p_cols = st.columns(3)
        selected_preset = None

        for p_idx, preset_text in enumerate(presets):
            with p_cols[p_idx]:
                if st.button(preset_text, key=f"preset_btn_primary_{p_idx}"):
                    selected_preset = preset_text

        st.markdown("</div>", unsafe_allow_html=True)

        # Trigger quick preset selection
        if selected_preset:
            st.session_state.chat_messages.append({"role": "user", "content": selected_preset})
            if len(st.session_state.history) > 0:
                st.session_state.history[-1]["User_Reflection"] = selected_preset
            bot_reply = generate_companion_reply(selected_preset, dom_emotion)
            st.session_state.chat_messages.append({"role": "ai", "content": bot_reply})
            if dom_emotion in ["happy", "surprise"]:
                st.balloons()
            st.rerun()

        # Freeform text chat row
        chat_col_input, chat_col_btn = st.columns([4.2, 1.1])
        with chat_col_input:
            user_chat_text = st.text_input(
                "Message Companion",
                placeholder="Or type your own reason here...",
                label_visibility="collapsed",
                key="user_dialogue_input_primary",
            )
        with chat_col_btn:
            if st.button("Send 🫧", key="send_chat_btn_primary"):
                if user_chat_text.strip():
                    st.session_state.chat_messages.append({"role": "user", "content": user_chat_text.strip()})
                    if len(st.session_state.history) > 0:
                        st.session_state.history[-1]["User_Reflection"] = user_chat_text.strip()
                    bot_reply = generate_companion_reply(user_chat_text.strip(), dom_emotion)
                    st.session_state.chat_messages.append({"role": "ai", "content": bot_reply})
                    if dom_emotion in ["happy", "surprise"]:
                        st.balloons()
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# SESSION HISTORY
# =============================================================================

if len(st.session_state.history) > 0:
    st.markdown("---")
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

    for item in st.session_state.history[-5:][::-1]:
        classification = item["Classification"]
        reflection = item.get("User_Reflection", "None")
        emoji = EMOJI_MAP.get(classification.lower(), "🤔")
        reflection_markup = (
            f'<div style="font-size:0.75rem; color:{accent_tint}; margin-top:2px;">Thought: "{reflection}"</div>'
            if reflection != "None"
            else ""
        )
        card_html = (
            f'<div class="history-item">'
            f'<div style="display:flex; align-items:center; gap:10px;">'
            f'<span class="raw-emoji" style="font-size:1.3rem;">{emoji}</span>'
            f'<div>'
            f'<div class="ambient-glow-text" style="font-weight:800; font-size:0.92rem;">{classification}</div>'
            f'<div style="font-size:0.7rem; color:#94A3B8;">{item["Timestamp"]}</div>'
            f'{reflection_markup}'
            f'</div>'
            f'</div>'
            f'<div class="ambient-glow-text" style="font-weight:800; font-size:0.92rem;">{item["Confidence"]}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)