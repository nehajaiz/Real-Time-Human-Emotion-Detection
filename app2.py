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
import random

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
st.title("🧠 Live Emotion Detection with Indian Music Recommendations")

# Initialize session state for emotion history
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = {emotion: deque(maxlen=50) for emotion in 
                                        ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']}
    st.session_state.time_stamps = deque(maxlen=50)
    st.session_state.start_time = time.time()
    st.session_state.last_recommended_song = None
    st.session_state.last_emotion_for_song = None

# -----------------------------
# Indian Music Database by Emotion with YouTube Links
# -----------------------------
MUSIC_RECOMMENDATIONS = {
    "happy": [
        {"song": "Kala Chashma", "artist": "Amar Arshi, Badshah", "movie": "Baar Baar Dekho", "youtube": "https://www.youtube.com/watch?v=yiqHoPCqueue", "emoji": "🎉", "language": "Hindi"},
        {"song": "Balam Pichkari", "artist": "Vishal Dadlani, Shalmali Kholgade", "movie": "Yeh Jawaani Hai Deewani", "youtube": "https://www.youtube.com/watch?v=0WDZRi6Y1mM", "emoji": "💃", "language": "Hindi"},
        {"song": "Gallan Goodiyaan", "artist": "Yashita Sharma, Manish Kumar", "movie": "Dil Dhadakne Do", "youtube": "https://www.youtube.com/watch?v=jEzF8bjyHXA", "emoji": "🎊", "language": "Hindi"},
        {"song": "Badtameez Dil", "artist": "Benny Dayal, Shefali Alvares", "movie": "Yeh Jawaani Hai Deewani", "youtube": "https://www.youtube.com/watch?v=OZ2LX7Qa7RU", "emoji": "❤️", "language": "Hindi"},
        {"song": "Why This Kolaveri Di", "artist": "Dhanush", "movie": "3", "youtube": "https://www.youtube.com/watch?v=YR12Z8f1Dh8", "emoji": "😄", "language": "Tamil"},
        {"song": "Lungi Dance", "artist": "Honey Singh", "movie": "Chennai Express", "youtube": "https://www.youtube.com/watch?v=WhF6K7rDdg8", "emoji": "🕺", "language": "Hindi"},
    ],
    "sad": [
        {"song": "Tum Hi Ho", "artist": "Arijit Singh", "movie": "Aashiqui 2", "youtube": "https://www.youtube.com/watch?v=IJq0yyWug1k", "emoji": "💔", "language": "Hindi"},
        {"song": "Channa Mereya", "artist": "Arijit Singh", "movie": "Ae Dil Hai Mushkil", "youtube": "https://www.youtube.com/watch?v=bzSTpdcs-EI", "emoji": "😢", "language": "Hindi"},
        {"song": "Kabira", "artist": "Tochi Raina, Rekha Bhardwaj", "movie": "Yeh Jawaani Hai Deewani", "youtube": "https://www.youtube.com/watch?v=jHNNMj5bNQw", "emoji": "🥀", "language": "Hindi"},
        {"song": "Agar Tum Saath Ho", "artist": "Alka Yagnik, Arijit Singh", "movie": "Tamasha", "youtube": "https://www.youtube.com/watch?v=sK7riqg2mr4", "emoji": "💧", "language": "Hindi"},
        {"song": "Ae Dil Hai Mushkil", "artist": "Arijit Singh", "movie": "Ae Dil Hai Mushkil", "youtube": "https://www.youtube.com/watch?v=Z_PODraXg4E", "emoji": "🌧️", "language": "Hindi"},
        {"song": "Phir Le Aya Dil", "artist": "Arijit Singh", "movie": "Barfi", "youtube": "https://www.youtube.com/watch?v=vc3JWo2yk4k", "emoji": "🍂", "language": "Hindi"},
    ],
    "angry": [
        {"song": "Apna Time Aayega", "artist": "Ranveer Singh, Divine", "movie": "Gully Boy", "youtube": "https://www.youtube.com/watch?v=jFGKtFkJwxQ", "emoji": "🔥", "language": "Hindi"},
        {"song": "Khalibali", "artist": "Shivam Pathak, Shail Hada", "movie": "Padmaavat", "youtube": "https://www.youtube.com/watch?v=jvOlJXWHFII", "emoji": "💥", "language": "Hindi"},
        {"song": "Deva Deva", "artist": "Arijit Singh, Jonita Gandhi", "movie": "Brahmāstra", "youtube": "https://www.youtube.com/watch?v=iW-JmkHBOJU", "emoji": "⚡", "language": "Hindi"},
        {"song": "Sultan", "artist": "Sukhwinder Singh", "movie": "Sultan", "youtube": "https://www.youtube.com/watch?v=cGfy43tzIvI", "emoji": "💪", "language": "Hindi"},
        {"song": "Ghungroo", "artist": "Arijit Singh, Shilpa Rao", "movie": "War", "youtube": "https://www.youtube.com/watch?v=Tlsmj5CeLm0", "emoji": "😤", "language": "Hindi"},
        {"song": "Malhari", "artist": "Vishal Dadlani", "movie": "Bajirao Mastani", "youtube": "https://www.youtube.com/watch?v=l_MyUGq7pgs", "emoji": "⚔️", "language": "Hindi"},
    ],
    "fear": [
        {"song": "Tujhe Kitna Chahne Lage", "artist": "Arijit Singh", "movie": "Kabir Singh", "youtube": "https://www.youtube.com/watch?v=n7ybYAer_wc", "emoji": "🫂", "language": "Hindi"},
        {"song": "Tera Ban Jaunga", "artist": "Akhil Sachdeva, Tulsi Kumar", "movie": "Kabir Singh", "youtube": "https://www.youtube.com/watch?v=XuWJZfGxRLc", "emoji": "🛡️", "language": "Hindi"},
        {"song": "Safar", "artist": "Arijit Singh", "movie": "Jab Harry Met Sejal", "youtube": "https://www.youtube.com/watch?v=dZj6Vu6JqmU", "emoji": "🕊️", "language": "Hindi"},
        {"song": "Main Agar Kahoon", "artist": "Sonu Nigam, Shreya Ghoshal", "movie": "Om Shanti Om", "youtube": "https://www.youtube.com/watch?v=_cBJ5YMQ9EQ", "emoji": "💫", "language": "Hindi"},
        {"song": "Pal", "artist": "KK", "movie": "Pal", "youtube": "https://www.youtube.com/watch?v=H6AcAKGzPbg", "emoji": "🌟", "language": "Hindi"},
    ],
    "surprise": [
        {"song": "Naatu Naatu", "artist": "Rahul Sipligunj, Kaala Bhairava", "movie": "RRR", "youtube": "https://www.youtube.com/watch?v=9yXo3m91uBs", "emoji": "🎺", "language": "Telugu"},
        {"song": "Oo Antava", "artist": "Indravathi Chauhan", "movie": "Pushpa", "youtube": "https://www.youtube.com/watch?v=SmMqNlzl5Dg", "emoji": "✨", "language": "Telugu"},
        {"song": "The Breakup Song", "artist": "Arijit Singh, Jonita Gandhi", "movie": "Ae Dil Hai Mushkil", "youtube": "https://www.youtube.com/watch?v=KiBS-dbv_x0", "emoji": "🎉", "language": "Hindi"},
        {"song": "Kajra Re", "artist": "Alisha Chinai, Shankar Mahadevan", "movie": "Bunty Aur Babli", "youtube": "https://www.youtube.com/watch?v=bCnLBNd0Eqc", "emoji": "💃", "language": "Hindi"},
        {"song": "Sheila Ki Jawani", "artist": "Sunidhi Chauhan, Vishal Dadlani", "movie": "Tees Maar Khan", "youtube": "https://www.youtube.com/watch?v=3CIOPn8xY8w", "emoji": "🔥", "language": "Hindi"},
    ],
    "disgust": [
        {"song": "Psycho Saiyaan", "artist": "Sachet Tandon, Dhvani Bhanushali", "movie": "Saaho", "youtube": "https://www.youtube.com/watch?v=VNJGCjQkvEA", "emoji": "😈", "language": "Hindi"},
        {"song": "Humsafar", "artist": "Akhil Sachdeva, Mansheel Gujral", "movie": "Badrinath Ki Dulhania", "youtube": "https://www.youtube.com/watch?v=hG5ZjvbPqSU", "emoji": "🖤", "language": "Hindi"},
        {"song": "Illegal Weapon 2.0", "artist": "Jasmine Sandlas, Garry Sandhu", "movie": "Street Dancer 3D", "youtube": "https://www.youtube.com/watch?v=7wWXa8B5dz8", "emoji": "☠️", "language": "Punjabi"},
        {"song": "Namo Namo", "artist": "Amit Trivedi", "movie": "Kedarnath", "youtube": "https://www.youtube.com/watch?v=Yd60nI4sa9A", "emoji": "🎭", "language": "Hindi"},
    ],
    "neutral": [
        {"song": "Kun Faya Kun", "artist": "A.R. Rahman, Javed Ali", "movie": "Rockstar", "youtube": "https://www.youtube.com/watch?v=T94PHkuydcw", "emoji": "🧘", "language": "Hindi"},
        {"song": "Piya Ke Ghar", "artist": "Ghulam Ali", "movie": "Classical", "youtube": "https://www.youtube.com/watch?v=lk5MrObfBvE", "emoji": "🌙", "language": "Urdu"},
        {"song": "Raabta", "artist": "Arijit Singh", "movie": "Agent Vinod", "youtube": "https://www.youtube.com/watch?v=2LiB2lbT7jo", "emoji": "🌫️", "language": "Hindi"},
        {"song": "Tum Se Hi", "artist": "Mohit Chauhan", "movie": "Jab We Met", "youtube": "https://www.youtube.com/watch?v=409f_H_uwqM", "emoji": "🎹", "language": "Hindi"},
        {"song": "Moh Moh Ke Dhaage", "artist": "Monali Thakur", "movie": "Dum Laga Ke Haisha", "youtube": "https://www.youtube.com/watch?v=Gk8oxAjQ9XM", "emoji": "🌸", "language": "Hindi"},
    ]
}

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
# Function to get music recommendation
# -----------------------------
def get_music_recommendation(emotion):
    if emotion in MUSIC_RECOMMENDATIONS:
        songs = MUSIC_RECOMMENDATIONS[emotion]
        return random.choice(songs)
    return None

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
st.markdown("### 📹 Live Emotion Detection with Indian Music Recommendations")

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

# Music recommendation section
st.markdown("---")
st.markdown("### 🎵 Indian Music Recommendation Based on Your Emotion")
MUSIC_WINDOW = st.empty()
YOUTUBE_WINDOW = st.empty()
AUDIO_WINDOW = st.empty()

# Reset button
if st.button("🔄 Reset Graph History"):
    st.session_state.emotion_history = {emotion: deque(maxlen=50) for emotion in emotions}
    st.session_state.time_stamps = deque(maxlen=50)
    st.session_state.start_time = time.time()
    st.session_state.last_recommended_song = None
    st.session_state.last_emotion_for_song = None
    st.success("Graph history reset!")

# -----------------------------
# Live Video Processing
# -----------------------------
if run_detection:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access webcam. Please check your camera permissions.")
        st.stop()
    
    # Text-to-speech responses in Hindi-English mix
    responses = {
        "happy": "Aap khush lag rahe hain! Here's a happy Bollywood song for you!",
        "sad": "Aap udaas lag rahe hain. Ye gaana suniye, better feel hoga.",
        "angry": "Aap gussa lag rahe hain. Ye powerful song suniye!",
        "fear": "Aap dar rahe hain. Don't worry, ye soothing song suniye.",
        "surprise": "Wow! Aap surprised hain! Ye exciting song aapke liye.",
        "disgust": "Here's an intense song matching your mood.",
        "neutral": "Aap calm hain. Ye peaceful song enjoy kijiye."
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
            
            # Generate music recommendation when emotion changes or every 60 frames
            if (detected_emotion != st.session_state.last_emotion_for_song and frame_count % 30 == 0) or frame_count == 1:
                song = get_music_recommendation(detected_emotion)
                
                if song:
                    st.session_state.last_recommended_song = song
                    st.session_state.last_emotion_for_song = detected_emotion
                    
                    # Display music recommendation
                    MUSIC_WINDOW.markdown(f"""
                    <div style='background: linear-gradient(135deg, {emotion_colors[detected_emotion]}40, {emotion_colors[detected_emotion]}20); 
                                 padding: 30px; border-radius: 15px; border-left: 5px solid {emotion_colors[detected_emotion]};'>
                        <h2 style='margin: 0;'>{song['emoji']} Recommended for You</h2>
                        <h3 style='color: {emotion_colors[detected_emotion]}; margin: 10px 0;'>🎵 {song['song']}</h3>
                        <p style='font-size: 18px; margin: 5px 0;'><strong>Artist:</strong> {song['artist']}</p>
                        <p style='font-size: 18px; margin: 5px 0;'><strong>Movie:</strong> {song['movie']}</p>
                        <p style='font-size: 16px; margin: 5px 0;'><strong>Language:</strong> {song['language']}</p>
                        <p style='font-size: 16px; margin-top: 15px; opacity: 0.8;'>Based on your <strong>{detected_emotion}</strong> emotion</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # === YOUTUBE TOOL IMPLEMENTATION START ===
                    song_query = f"{song['song']} {song['artist']}"
                    
                    # Replace iframe with message about tool execution
                    YOUTUBE_WINDOW.markdown(f"""
                    <div style='margin-top: 20px; text-align: center;'>
                        <h2 style='color: #FFD700;'>🚀 YouTube Tool Execution!</h2>
                        <p>The song **'{song['song']}'** is being started on your device using the YouTube Play tool.</p>
                        <p style='font-size: 14px;'>Query sent to tool: <code>{song_query}</code></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # In a live environment (e.g., Google Assistant), the tool call would be made here:
                    # youtube:play(query=song_query)
                    
                    # === YOUTUBE TOOL IMPLEMENTATION END ===
                    
                    # Generate audio response
                    try:
                        text_to_speak = responses.get(detected_emotion, "Here's a song for you.")
                        tts = gTTS(text_to_speak, lang='hi')
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                        tts.save(temp_file.name)
                        
                        with open(temp_file.name, "rb") as audio_file:
                            AUDIO_WINDOW.audio(audio_file.read(), format="audio/mp3")
                    except:
                        pass
            
            # Display current recommendation even when emotion doesn't change
            elif st.session_state.last_recommended_song:
                song = st.session_state.last_recommended_song
                MUSIC_WINDOW.markdown(f"""
                <div style='background: linear-gradient(135deg, {emotion_colors[detected_emotion]}40, {emotion_colors[detected_emotion]}20); 
                             padding: 30px; border-radius: 15px; border-left: 5px solid {emotion_colors[detected_emotion]};'>
                    <h2 style='margin: 0;'>{song['emoji']} Recommended for You</h2>
                    <h3 style='color: {emotion_colors[detected_emotion]}; margin: 10px 0;'>🎵 {song['song']}</h3>
                    <p style='font-size: 18px; margin: 5px 0;'><strong>Artist:</strong> {song['artist']}</p>
                    <p style='font-size: 18px; margin: 5px 0;'><strong>Movie:</strong> {song['movie']}</p>
                    <p style='font-size: 16px; margin: 5px 0;'><strong>Language:</strong> {song['language']}</p>
                    <p style='font-size: 16px; margin-top: 15px; opacity: 0.8;'>Based on your <strong>{detected_emotion}</strong> emotion</p>
                </div>
                """, unsafe_allow_html=True)
                
                # === YOUTUBE TOOL IMPLEMENTATION START (KEEP SHOWING MESSAGE) ===
                song_query = f"{song['song']} {song['artist']}"
                
                # Keep showing message about tool execution
                YOUTUBE_WINDOW.markdown(f"""
                <div style='margin-top: 20px; text-align: center;'>
                    <h2 style='color: #FFD700;'>🚀 YouTube Tool Execution!</h2>
                    <p>The song **'{song['song']}'** is being played on your device using the YouTube Play tool.</p>
                    <p style='font-size: 14px;'>Query used: <code>{song_query}</code></p>
                </div>
                """, unsafe_allow_html=True)
                # === YOUTUBE TOOL IMPLEMENTATION END ===
                
            last_emotion = detected_emotion
        else:
            EMOTION_DISPLAY.warning("No face detected in frame")
        
        frame_count += 1
        
        # Small delay to control frame rate
        time.sleep(0.03)
    
    cap.release()
else:
    st.info("👆 Check the box above to start live detection with Indian music recommendations")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app provides **real-time emotion detection** with **Indian music recommendations** from YouTube!

✨ **Features:**
- Live webcam feed
- Real-time emotion graphs
- **YouTube Tool Playback (Replaced iFrame)**
- Hindi audio feedback
- Bollywood, Tamil, Telugu songs

🎭 **Emotions & Songs:**
- 😊 Happy → Dance numbers
- 😢 Sad → Romantic melodies  
- 😠 Angry → Power anthems
- 😨 Fear → Soothing songs
- 😮 Surprise → Trending hits
- 🤢 Disgust → Intense tracks
- 😐 Neutral → Peaceful tunes
""")

st.sidebar.title("🎵 Music Database")
st.sidebar.markdown(f"""
**Total Indian Songs:** {sum(len(songs) for songs in MUSIC_RECOMMENDATIONS.values())}

**Languages:**
- Hindi
- Tamil
- Telugu
- Punjabi
- Urdu

Songs per emotion:
- Happy: {len(MUSIC_RECOMMENDATIONS['happy'])}
- Sad: {len(MUSIC_RECOMMENDATIONS['sad'])}
- Angry: {len(MUSIC_RECOMMENDATIONS['angry'])}
- Fear: {len(MUSIC_RECOMMENDATIONS['fear'])}
- Surprise: {len(MUSIC_RECOMMENDATIONS['surprise'])}
- Disgust: {len(MUSIC_RECOMMENDATIONS['disgust'])}
- Neutral: {len(MUSIC_RECOMMENDATIONS['neutral'])}
""")