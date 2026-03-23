import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import DepthwiseConv2D
from gtts import gTTS
import tempfile
from PIL import Image
import plotly.graph_objects as go
from collections import deque
import time

# -----------------------------
# Custom loader to handle legacy parameters
# -----------------------------
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Live Emotion Detection", layout="wide")
st.title("🧠 Live Emotion Detection with Real-Time Graph")

# Initialize session state for emotion history
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = {emotion: deque(maxlen=50) for emotion in 
                                        ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']}
    st.session_state.time_stamps = deque(maxlen=50)
    st.session_state.start_time = time.time()

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model("best_model.h5", custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_emotion_model()
if model is None:
    st.stop()

# Emotion labels and colors
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
emotion_colors = {
    'angry': '#FF4B4B',
    'disgust': '#8B4513',
    'fear': '#9370DB',
    'happy': '#FFD700',
    'sad': '#4682B4',
    'surprise': '#FF69B4',
    'neutral': '#808080'
}

# Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------------
# Function to predict emotion
# -----------------------------
def predict_emotion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        
        img_pixels = image.img_to_array(face_img)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        preds = model.predict(img_pixels, verbose=0)[0]
        emotion_index = np.argmax(preds)
        detected_emotion = emotions[emotion_index]
        confidence = preds[emotion_index] * 100

        # Draw on image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        label = f"{detected_emotion.upper()} ({confidence:.1f}%)"
        cv2.putText(img, label, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img, detected_emotion, preds
    
    return img, None, None

# -----------------------------
# Function to update emotion graph
# -----------------------------
def update_emotion_graph(predictions):
    current_time = time.time() - st.session_state.start_time
    st.session_state.time_stamps.append(current_time)
    
    for i, emotion in enumerate(emotions):
        st.session_state.emotion_history[emotion].append(predictions[i] * 100)

def create_live_graph():
    fig = go.Figure()
    
    for emotion in emotions:
        fig.add_trace(go.Scatter(
            x=list(st.session_state.time_stamps),
            y=list(st.session_state.emotion_history[emotion]),
            mode='lines',
            name=emotion.capitalize(),
            line=dict(color=emotion_colors[emotion], width=2),
            fill='tonexty' if emotion == 'angry' else None
        ))
    
    fig.update_layout(
        title="Real-Time Emotion Probability Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Confidence (%)",
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark"
    )
    
    return fig

def create_bar_chart(predictions):
    fig = go.Figure(data=[
        go.Bar(
            x=list(emotions),
            y=predictions * 100,
            marker_color=[emotion_colors[e] for e in emotions],
            text=[f'{p:.1f}%' for p in predictions * 100],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Current Emotion Distribution",
        xaxis_title="Emotion",
        yaxis_title="Confidence (%)",
        height=350,
        template="plotly_dark",
        showlegend=False
    )
    
    return fig

# -----------------------------
# Streamlit UI - Live Video
# -----------------------------
st.markdown("### 📹 Live Emotion Detection")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 📸 Camera Feed")
    run_detection = st.checkbox("Start Live Detection", value=False)
    
    FRAME_WINDOW = st.empty()
    
with col2:
    st.markdown("#### 📊 Live Emotion Graph")
    GRAPH_WINDOW = st.empty()

st.markdown("---")

col3, col4 = st.columns([1, 1])

with col3:
    st.markdown("#### 📈 Current Emotion Distribution")
    BAR_CHART_WINDOW = st.empty()

with col4:
    st.markdown("#### 🎯 Detected Emotion")
    EMOTION_DISPLAY = st.empty()
    AUDIO_WINDOW = st.empty()

# Reset button
if st.button("🔄 Reset Graph History"):
    st.session_state.emotion_history = {emotion: deque(maxlen=50) for emotion in emotions}
    st.session_state.time_stamps = deque(maxlen=50)
    st.session_state.start_time = time.time()
    st.success("Graph history reset!")

# -----------------------------
# Live Video Processing
# -----------------------------
if run_detection:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access webcam. Please check your camera permissions.")
        st.stop()
    
    # Text-to-speech responses
    responses = {
        "happy": "You look happy! Keep smiling!",
        "sad": "You seem sad. Everything will be okay, stay strong.",
        "angry": "You appear angry. Take a deep breath and relax.",
        "fear": "You look scared. Don't worry, you are safe.",
        "surprise": "You look surprised! Is something amazing happening?",
        "disgust": "You seem disgusted. Hope things get better.",
        "neutral": "You look neutral. A calm mind is a strong mind."
    }
    
    last_emotion = None
    frame_count = 0
    
    while run_detection:
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to capture frame")
            break
        
        # Process every frame
        frame = cv2.flip(frame, 1)  # Mirror the image
        processed_frame, detected_emotion, predictions = predict_emotion(frame)
        
        # Display video frame
        FRAME_WINDOW.image(processed_frame, channels="BGR", use_container_width=True)
        
        if detected_emotion and predictions is not None:
            # Update graph every 5 frames to reduce lag
            if frame_count % 5 == 0:
                update_emotion_graph(predictions)
                
                # Update live graph
                GRAPH_WINDOW.plotly_chart(create_live_graph(), use_container_width=True, key=f"graph_{frame_count}")
                
                # Update bar chart
                BAR_CHART_WINDOW.plotly_chart(create_bar_chart(predictions), use_container_width=True, key=f"bar_{frame_count}")
            
            # Update emotion display
            confidence = predictions[emotions.index(detected_emotion)] * 100
            EMOTION_DISPLAY.markdown(f"""
            <div style='background-color: {emotion_colors[detected_emotion]}; padding: 20px; border-radius: 10px; text-align: center;'>
                <h1 style='color: white; margin: 0;'>{detected_emotion.upper()}</h1>
                <h3 style='color: white; margin: 0;'>Confidence: {confidence:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate audio response when emotion changes
            if detected_emotion != last_emotion and frame_count % 30 == 0:  # Every 30 frames
                try:
                    text_to_speak = responses.get(detected_emotion, "Emotion detected.")
                    tts = gTTS(text_to_speak)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(temp_file.name)
                    
                    with open(temp_file.name, "rb") as audio_file:
                        AUDIO_WINDOW.audio(audio_file.read(), format="audio/mp3")
                except:
                    pass
                
                last_emotion = detected_emotion
        else:
            EMOTION_DISPLAY.warning("No face detected in frame")
        
        frame_count += 1
        
        # Small delay to control frame rate
        time.sleep(0.03)
    
    cap.release()
else:
    st.info("👆 Check the box above to start live detection")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app provides **real-time emotion detection** with:

✨ **Features:**
- Live webcam feed
- Real-time emotion graphs
- Confidence distribution chart
- Audio feedback

🎭 **Emotions Detected:**
- 😊 Happy
- 😢 Sad  
- 😠 Angry
- 😨 Fear
- 😮 Surprise
- 🤢 Disgust
- 😐 Neutral
""")

st.sidebar.title("🎮 Controls")
st.sidebar.markdown("""
1. **Start Live Detection**: Enable webcam
2. **Reset Graph**: Clear history
3. Watch real-time emotion changes!
""")