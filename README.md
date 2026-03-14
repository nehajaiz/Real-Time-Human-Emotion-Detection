🎭 Real-Time Human Emotion Detection

A Streamlit-based web application that detects human emotions in real-time using a webcam feed.
The project uses computer vision and deep learning to analyze facial expressions and classify emotions such as Angry, Happy, Sad, Surprise, Fear, Neutral, and Disgust.

It integrates OpenCV for face detection and a TensorFlow/Keras deep learning model trained on the FER2013 dataset to perform emotion recognition.

📌 Table of Contents

Installation
Usage
Features
Technologies Used
Contributing
Contact

⚙️ Installation
1. Clone the Repository
git clone https://github.com/nehajaiz/Real-Time-Human-Emotion-Detection.git
2. Navigate to the Project Directory
cd Real-Time-Human-Emotion-Detection
3. Install Required Dependencies
pip install -r requirements.txt
4. Setup Haarcascade File

Make sure the Haarcascade XML file for face detection is placed in the correct directory.

▶️ Usage

Run the Streamlit application:
streamlit run app.py

Then:

Open the Live Detection section in the web interface.

Allow webcam access.

The system will start detecting emotions in real-time.

Click Stop Webcam to terminate the detection.

✨ Features

🎥 Real-time emotion detection using webcam
😀 Detects 7 human emotions

Angry

Disgust

Fear

Happy

Neutral

Sad

Surprise

🧠 Deep learning model trained on the FER2013 dataset

⚡ Fast and interactive Streamlit web interface

👤 Face detection using OpenCV

🛠 Technologies Used

Python

Streamlit

OpenCV

TensorFlow

Keras

Deep Learning


📸 Screenshot


🌐 Live Demo
https://github.com/nehajaiz/Real-Time-Human-Emotion-Detection/


🤝 Contributing

Contributions are welcome!

If you'd like to improve the project:

Fork the repository

Create a new branch

Commit your changes

Submit a pull request

You can also open an issue for bug reports or feature suggestions.



