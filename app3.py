import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import threading
import tempfile
import os
import random # Used for random song, age, and emotion selection
from collections import deque
from gtts import gTTS
import plotly.graph_objects as go
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="Random Detector Demo", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Placeholder Model Loading (Removed actual model loading)
# -----------------------------
# These functions are now placeholders for code clarity, but models are not loaded.
emotion_model = True 
age_model = True

# Load Haarcascade for face detection (This is still needed for OpenCV face detection)
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    # Fallback path if haarcascades are installed locally
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        st.error("❌ Haarcascade XML not found. Cannot detect faces.")
        st.stop()


# ---------------------------
# Globals & Constants
# ---------------------------
EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise']
EMO_EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
    "neutral": "😐", "sad": "😢", "surprise": "😲"
}
EMO_COLOR = {
    "angry":"#E74C3C", "disgust":"#7D3C98", "fear":"#34495E", "happy":"#F1C40F",
    "neutral":"#95A5A6", "sad":"#3498DB", "surprise":"#E67E22"
}

# Age detection constants (Used for random selection)
AGE_BINS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# MODEL_MEAN_VALUES is no longer needed

# TTS Messages
SPEECH_MESSAGES = {
    "happy": "Aap khush lag rahe hain! Here's a happy Bollywood song for you!",
    "sad": "Aap udaas lag rahe hain. Ye gaana suniye, better feel hoga.",
    "angry": "Aap gussa lag rahe hain. Ye powerful song suniye!",
    "fear": "Aap dar rahe hain. Don't worry, ye soothing song suniye.",
    "surprise": "Wow! Aap surprised hain! Ye exciting song aapke liye.",
    "disgust": "Here's an intense song matching your mood.",
    "neutral": "Aap calm hain. Ye peaceful song enjoy kijiye."
}

# Music Recommendations (simplified)
MUSIC_RECOMMENDATIONS = {
    "happy": [
        {"song": "Kala Chashma", "artist": "Amar Arshi, Badshah", "language": "Hindi"},
    ],
    "sad": [
        {"song": "Tum Hi Ho", "artist": "Arijit Singh", "language": "Hindi"},
    ],
    "angry": [
        {"song": "Apna Time Aayega", "artist": "Ranveer Singh, Divine", "language": "Hindi"},
    ],
    "fear": [
        {"song": "Tujhe Kitna Chahne Lage", "artist": "Arijit Singh", "language": "Hindi"},
    ],
    "surprise": [
        {"song": "Naatu Naatu", "artist": "Rahul Sipligunj, Kaala Bhairava", "language": "Telugu"},
    ],
    "disgust": [
        {"song": "Psycho Saiyaan", "artist": "Sachet Tandon, Dhvani Bhanushali", "language": "Hindi"},
    ],
    "neutral": [
        {"song": "Kun Faya Kun", "artist": "A.R. Rahman, Javed Ali", "language": "Hindi"},
    ]
}

# ---------------------------
# Session State Initialization
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=100)
if "running" not in st.session_state:
    st.session_state.running = False
if "last_speech_time" not in st.session_state:
    st.session_state.last_speech_time = 0
if "session_start" not in st.session_state:
    st.session_state.session_start = time.time()
if "last_recommended_song" not in st.session_state:
    st.session_state.last_recommended_song = None
if "last_emotion_for_song" not in st.session_state:
    st.session_state.last_emotion_for_song = None
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = {emotion: deque(maxlen=50) for emotion in EMOTIONS}
    st.session_state.time_stamps = deque(maxlen=50)


# ---------------------------
# Core Helpers
# ---------------------------
def get_music_recommendation(emotion):
    """Retrieves a random song for the detected emotion."""
    if emotion in MUSIC_RECOMMENDATIONS:
        # Use random.choice for variety
        return random.choice(MUSIC_RECOMMENDATIONS[emotion]) 
    return None

def update_emotion_graph(predictions):
    """Appends new predictions and timestamps to history."""
    current_time = time.time() - st.session_state.session_start
    st.session_state.time_stamps.append(current_time)
    
    for i, emotion in enumerate(EMOTIONS):
        st.session_state.emotion_history[emotion].append(predictions[i] * 100)

def create_live_graph():
    """Creates a Plotly timeline graph."""
    fig = go.Figure()
    
    for emotion in EMOTIONS:
        fig.add_trace(go.Scatter(
            x=list(st.session_state.time_stamps),
            y=list(st.session_state.emotion_history[emotion]),
            mode='lines',
            name=emotion.capitalize(),
            line=dict(color=EMO_COLOR[emotion], width=2),
        ))
    
    fig.update_layout(
        xaxis_title="Time (seconds)", yaxis_title="Confidence (%)", height=250,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    st.header("Detection Settings")
    
    # Threshold doesn't matter much since confidence is fixed, but keep it for UI.
    confidence_threshold = st.slider("Min probability to record", 0.0, 1.0, 0.5, 0.01) 
    
    st.header("Music & Audio")
    tts_enabled = st.checkbox("Enable speech output", value=True)
    speech_interval = st.slider("Speech interval (seconds)", 3, 15, 5)
    
    st.markdown("---")
    if st.button("🗑️ Clear History & Graphs"):
        st.session_state.history.clear()
        st.session_state.session_start = time.time()
        st.session_state.emotion_history = {emotion: deque(maxlen=50) for emotion in EMOTIONS}
        st.session_state.time_stamps = deque(maxlen=50)
        st.success("History and graphs reset.")

# ---------------------------
# UI Header & Controls
# ---------------------------
st.markdown("<h1>🔀 Random Emotion & Age Detector (Demo Mode)</h1>", unsafe_allow_html=True)
st.caption("Live webcam detection showing random age and emotion values.")

col_start, col_meta = st.columns([1,1])
with col_start:
    if not st.session_state.running:
        if st.button("▶️ Start Webcam", type="primary"):
            st.session_state.running = True
            st.session_state.session_start = time.time()
    else:
        if st.button("⏹️ Stop Webcam", type="secondary"):
            st.session_state.running = False
            
# ---------------------------
# Layout
# ---------------------------
left, right = st.columns([2,1])

with left:
    st.markdown("### 📹 Live Camera & Detection")
    camera_placeholder = st.empty()
    
    meta_cols = st.columns([1,1,1])
    fps_placeholder = meta_cols[0].empty()
    facecount_placeholder = meta_cols[1].empty()
    top_label_placeholder = meta_cols[2].empty() # Shows emotion and age
    
    st.markdown("---")
    st.markdown("#### 📈 Emotion Timeline")
    timeline_chart = st.empty()

with right:
    st.markdown("### 🎶 Music Recommendation")
    MUSIC_WINDOW = st.empty()
    YOUTUBE_WINDOW = st.empty() # Placeholder for YouTube Tool message
    AUDIO_WINDOW = st.empty() # Placeholder for TTS audio
    st.markdown("---")
    st.markdown("#### 📜 Recent History (Emotion & Age)")
    history_placeholder = st.empty()

# If not running
if not st.session_state.running:
    st.info("👆 Click **Start Webcam** to begin the random detection demonstration.")
    st.stop()
    
# ---------------------------
# Camera Setup & Main Loop
# ---------------------------

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("❌ Unable to open webcam. Check device index (0) and permissions.")
    st.session_state.running = False
    st.stop()

prev_time = time.time()
frame_count = 0
display_width = 640
small_w = 320 
last_chart_update_frame = 0 
chart_update_interval = 10 

try:
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret or frame is None:
            st.error("❌ Camera not returning frames.")
            st.session_state.running = False
            break

        frame_count += 1
        now = time.time()
        
        # --- FPS Calculation ---
        if now - prev_time >= 1.0:
            fps = frame_count / (now - prev_time)
            prev_time = now
            frame_count = 0
        else:
            fps = st.session_state.get("fps", 0.0)
        st.session_state["fps"] = fps

        # Resize and mirror the frame
        h0, w0 = frame.shape[:2]
        scale = display_width / float(w0)
        disp_frame = cv2.resize(frame, (display_width, int(h0*scale)))
        disp_frame = cv2.flip(disp_frame, 1)
        frame = cv2.flip(frame, 1) 
        
        # Detection on smaller frame
        small_h = int(small_w * h0 / w0)
        small_gray = cv2.cvtColor(cv2.resize(frame, (small_w, small_h)), cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.3, minNeighbors=5)
        
        # --- Random Results Init ---
        best_label = None
        detected_age = "N/A"
        best_prob = 0.0 # Will be 0.95 for random emotion

        # --- Process Faces ---
        if len(faces) > 0:
            # Generate random results once per frame if a face is detected
            best_label = random.choice(EMOTIONS)
            detected_age = random.choice(AGE_BINS)
            best_prob = random.uniform(0.7, 0.99)
            
            # Simulate prediction array for graph (one random emotion high, others low)
            preds = np.array([random.uniform(0.01, 0.05) for _ in EMOTIONS])
            preds[EMOTIONS.index(best_label)] = best_prob
            
            probs_map = {EMOTIONS[i]: float(preds[i]) for i in range(len(EMOTIONS))}

            # Process the first detected face for drawing
            (xs, ys, ws, hs) = faces[0]
            
            # Rescale coordinates for drawing on the display frame
            x = int(xs * (w0 / small_w))
            y = int(ys * (h0 / small_h))
            w = int(ws * (w0 / small_w))
            h = int(hs * (h0 / small_h))

            sdx = int(x * scale)
            sdy = int(y * scale)
            
            color = tuple(int(EMO_COLOR.get(best_label, "#FFFFFF").lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            # Draw bounding box and labels
            cv2.rectangle(disp_frame, (sdx, sdy), (sdx + int(w*scale), sdy + int(h*scale)), color, 3)
            
            text_emotion = f"{best_label.title()} ({best_prob:.2f})"
            text_age = f"Age: {detected_age}"
            
            cv2.putText(disp_frame, text_emotion, (sdx, max(sdy - 35, 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(disp_frame, text_age, (sdx, max(sdy - 10, 35)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # If no face is detected, use zeroed predictions for graph
            preds = np.zeros(len(EMOTIONS))
            
        # --- Update Real-Time UI ---
        
        # Metadata
        fps_placeholder.markdown(f"<div class='small'>⚡ FPS: {fps:.1f}</div>", unsafe_allow_html=True)
        facecount_placeholder.markdown(f"<div class='small'>👤 Faces: {len(faces)}</div>", unsafe_allow_html=True)
        top_label_placeholder.markdown(f"<div class='small'>🎯 Top: <strong>{best_label or 'N/A'}</strong> ({best_prob:.2f}) | Age: <strong>{detected_age}</strong></div>", unsafe_allow_html=True)

        # Video Frame
        camera_placeholder.image(cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB), use_column_width=True, channels="RGB")

        # --- History and Music Logic ---
        displayed_label = best_label or "N/A"
        
        if best_label and best_prob >= confidence_threshold:
            # 1. Add to history
            timestamp = time.strftime("%H:%M:%S")
            st.session_state.history.append((best_label, detected_age, round(best_prob,2), timestamp))

            # 2. Music/Speech Logic
            time_since_speech = now - st.session_state.last_speech_time
            is_new_emotion = (displayed_label != st.session_state.last_emotion_for_song)
            
            if is_new_emotion or time_since_speech >= speech_interval:
                song = get_music_recommendation(displayed_label)
                
                if song:
                    st.session_state.last_recommended_song = song
                    st.session_state.last_emotion_for_song = displayed_label
                    st.session_state.last_speech_time = now # Reset timer on new action

                    # Display music recommendation card
                    MUSIC_WINDOW.markdown(f"""
                    <div style='background-color: {EMO_COLOR[displayed_label]}15; padding: 20px; border-radius: 10px; border-left: 5px solid {EMO_COLOR[displayed_label]};'>
                        <h3 style='color: {EMO_COLOR[displayed_label]}; margin: 0;'>{EMO_EMOJI.get(displayed_label, '🎵')} Recommended: {song['song']}</h3>
                        <p>Artist: {song['artist']} | Language: {song['language']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # --- YouTube Tool Implementation Message ---
                    song_query = f"{song['song']} {song['artist']}"
                    
                    YOUTUBE_WINDOW.markdown(f"""
                    <div style='margin-top: 10px; padding: 10px; border: 1px dashed #FFD700; background-color: #333;'>
                        <p style='color: #FFD700; margin: 0;'>**ACTION:** Playing song via YouTube Tool...</p>
                        <p style='font-size: 12px; margin: 0;'>Query: <code>{song_query}</code></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # --- Text-to-Speech Feedback ---
                    if tts_enabled:
                        try:
                            text_to_speak = SPEECH_MESSAGES.get(displayed_label, f"I detect {displayed_label} emotion.")
                            tts = gTTS(text_to_speak, lang='hi', slow=False)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                                tts.save(fp.name)
                                with open(fp.name, "rb") as audio_file:
                                    AUDIO_WINDOW.audio(audio_file.read(), format="audio/mp3", start_time=0)
                        except Exception:
                            pass 
            
            # Update current recommendation card if it wasn't refreshed this cycle
            elif st.session_state.last_recommended_song:
                song = st.session_state.last_recommended_song
                MUSIC_WINDOW.markdown(f"""
                <div style='background-color: {EMO_COLOR[displayed_label]}15; padding: 20px; border-radius: 10px; border-left: 5px solid {EMO_COLOR[displayed_label]};'>
                    <h3 style='color: {EMO_COLOR[displayed_label]}; margin: 0;'>{EMO_EMOJI.get(displayed_label, '🎵')} Recommended: {song['song']}</h3>
                    <p>Artist: {song['artist']} | Language: {song['language']}</p>
                </div>
                """, unsafe_allow_html=True)


        # --- Chart & History Updates (Less Frequent) ---
        if (frame_count - last_chart_update_frame) >= chart_update_interval and len(st.session_state.history) > 0:
            last_chart_update_frame = frame_count

            # Update graph history using the simulated predictions
            update_emotion_graph(preds)
            timeline_chart.plotly_chart(create_live_graph(), use_container_width=True) 

            # Recent history table (Emotion, Age, Confidence, Time)
            hist = list(st.session_state.history)[-15:][::-1]
            if hist:
                df_hist = pd.DataFrame(hist, columns=["Emotion","Age","Confidence","Time"])
                df_hist['Emoji'] = df_hist['Emotion'].map(EMO_EMOJI)
                df_hist['Confidence'] = df_hist['Confidence'].apply(lambda x: f"{x:.2f}")
                df_hist = df_hist[['Time', 'Emoji', 'Emotion', 'Age', 'Confidence']]
                history_placeholder.dataframe(df_hist, use_container_width=True, hide_index=True)

        time.sleep(0.01) # Small delay
        
except Exception as e:
    st.error(f"Error in detection loop: {e}")
    st.session_state.running = False
finally:
    try:
        cap.release()
    except Exception:
        pass