 Real-Time ASL Translation System
A real-time American Sign Language (ASL) translation system that converts sign language gestures into text and speech, with support for multiple languages. Built with deep learning, computer vision, and natural language processing.
Show Image
Show Image
Show Image
Show Image
 Features

Real-Time Detection: Live ASL gesture recognition using webcam
Advanced Neural Architecture: BiLSTM-CTC model for sentence-level recognition
Multi-Language Support: Translate ASL to multiple languages (Spanish, French, German, etc.)
Text-to-Speech: Audio output for recognized and translated text
MediaPipe Integration: Robust hand and pose landmark detection
Responsive Web Interface: Clean, mobile-friendly UI
Frame Skipping & Buffering: Optimized for real-time performance

Architecture
Model Architecture
Input (Keypoints) → Masking → BiLSTM Layers → Dense → Softmax → CTC Decoding

Input: 258-dimensional feature vector (hand + pose landmarks)
BiLSTM: 2 layers with 128 units each for temporal modeling
CTC Loss: Handles variable-length sequences without alignment
Greedy Decoding: Real-time character prediction

System Components
┌─────────────────────────────────────────────────────────────┐
│                        Web Interface                         │
│                     (HTML + JavaScript)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                      Flask Server                            │
│                       (app.py)                               │
└──────┬───────────────────────────────────┬──────────────────┘
       │                                   │
┌──────▼──────────────────┐    ┌──────────▼─────────────────┐
│  Realtime ASL Engine    │    │ Translation & TTS Engine   │
│  (realtime_engine.py)   │    │  (translation_tts.py)      │
│                         │    │                            │
│  • MediaPipe Detection  │    │  • Google Translate API    │
│  • Keypoint Extraction  │    │  • gTTS (Text-to-Speech)   │
│  • Frame Buffering      │    │  • Audio Caching           │
│  • Model Inference      │    │                            │
└──────┬──────────────────┘    └────────────────────────────┘
       │
┌──────▼──────────────────┐
│   BiLSTM-CTC Model      │
│     (model.py)          │
│                         │
│  • Variable-length seq  │
│  • CTC decoding         │
│  • Character prediction │
└─────────────────────────┘
🚀 Getting Started
Prerequisites
bashPython 3.8+
Webcam
Installation

Clone the repository

bashgit clone https://github.com/yourusername/asl-translation-system.git
cd asl-translation-system

Create virtual environment

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt

Set up model files

bash# Create models directory
mkdir -p models

# Place your trained model files:
# - models/asl_model.h5
# - models/char_map.json
Usage

Start the server

bashpython app.py

Open your browser

http://localhost:8080

Use the interface

Click "Start Detection" to begin ASL recognition
Select target language for translation
View real-time text and hear audio output



 Requirements
txttensorflow>=2.8.0
keras>=2.8.0
opencv-python>=4.5.0
mediapipe>=0.9.0
flask>=2.0.0
numpy>=1.21.0
googletrans==4.0.0rc1
gtts>=2.3.0
🎯 Model Training
The BiLSTM-CTC model is designed for sentence-level ASL recognition:
Data Format

Input: Sequences of 258-dimensional keypoint vectors
Output: Character sequences (a-z + space)
CTC Blank: Index 27

Training Steps

Prepare dataset with labeled ASL sentences
Extract MediaPipe keypoints (hand + pose landmarks)
Train using CTC loss
Export model to .h5 format

See model.py for architecture details.
🔧 Configuration
Edit the config dictionary in app.py:
pythonconfig = {
    'model': {
        'model_path': 'models/asl_model.h5',
        'char_map_path': 'models/char_map.json'
    },
    'webcam': {
        'camera_id': 0,
        'width': 640,
        'height': 480
    },
    'processing': {
        'buffer_size': 30,
        'min_frames_for_prediction': 5,
        'frame_skip': 1,
        'confidence_threshold': 0.2,
        'gesture_pause_frames': 5
    },
    'mediapipe': {
        'hand_detection_confidence': 0.5,
        'hand_tracking_confidence': 0.5,
        'pose_detection_confidence': 0.5,
        'pose_tracking_confidence': 0.5
    },
    'translation': {'enabled': True},
    'tts': {'enabled': True, 'slow': False}
}
 Performance Optimization

Frame Skipping: Process every Nth frame for better performance
Buffer Management: Rolling window of recent keypoints
Audio Caching: Avoid regenerating identical audio
Confidence Filtering: Only output high-confidence predictions

 Supported Languages

English (en)
Spanish (es)
French (fr)
German (de)
Italian (it)
Portuguese (pt)
And more via Google Translate API

📁 Project Structure
asl-translation-system/
│
├── app.py                  # Flask application & server
├── model.py                # BiLSTM-CTC model architecture
├── realtime_engine.py      # Real-time detection engine
├── translation_tts.py      # Translation & TTS module
│
├── models/
│   ├── asl_model.h5        # Trained model weights
│   └── char_map.json       # Character mapping
│
├── static/
│   └── index.html          # Web interface
│
├── requirements.txt        # Python dependencies
└── README.md              # This file
 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

 Future Improvements

 Beam search decoding for better accuracy
 Support for more ASL signs and gestures
 Offline translation models
 Mobile app version
 Multi-user support
 Cloud deployment guide
 Pre-trained model checkpoints
 Dataset collection tools

 Limitations

Requires good lighting conditions
Currently supports fingerspelling and basic gestures
Translation quality depends on Google Translate API
Real-time performance varies with hardware

 Acknowledgments

MediaPipe by Google for landmark detection
TensorFlow/Keras for deep learning framework
gTTS for text-to-speech conversion
ASL Community for inspiration and resources

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
