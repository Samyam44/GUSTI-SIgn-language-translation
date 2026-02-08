from flask import Flask, render_template, Response, request, jsonify
import threading
import cv2
import time
import base64

from realtime_engine import RealtimeASLEngine, WebcamCapture
from translation_tts import TranslationTTSEngine

# ================= Flask App =================
app = Flask(__name__)

# ================= Global Instances =================
config = {
    'model': {
        'model_path': 'models/asl_model.h5',
        'char_map_path': 'models/char_map.json'
    },
    'webcam': {'camera_id': 0, 'width': 640, 'height': 480},
    'processing': {'buffer_size': 30, 'min_frames_for_prediction': 5, 'frame_skip': 1, 'confidence_threshold': 0.2, 'gesture_pause_frames': 5},
    'mediapipe': {
        'hand_detection_confidence': 0.5,
        'hand_tracking_confidence': 0.5,
        'pose_detection_confidence': 0.5,
        'pose_tracking_confidence': 0.5
    },
    'translation': {'enabled': True},
    'tts': {'enabled': True, 'slow': False}
}

engine = RealtimeASLEngine(config, config['model']['model_path'], config['model']['char_map_path'])
tts_engine = TranslationTTSEngine(config)
# FIX: Pass frame_skip parameter
webcam = WebcamCapture(
    camera_id=config['webcam']['camera_id'], 
    width=config['webcam']['width'], 
    height=config['webcam']['height'],
    frame_skip=config['processing']['frame_skip']
)
detection_active = False
current_language = 'en'
latest_sentence = ""
latest_translation = ""
latest_audio_base64 = ""

lock = threading.Lock()

# Debug counter
frame_counter = 0
processed_counter = 0

# ================= Video Generator =================
def generate_frames():
    global detection_active, latest_sentence, latest_translation, latest_audio_base64
    global frame_counter, processed_counter
    
    while True:
        should_process, frame = webcam.read()
        if frame is None:
            continue

        frame_counter += 1

        if detection_active and should_process:
            processed_counter += 1
            print(f"[DEBUG] Processing frame {processed_counter} (detection active, buffer: {len(engine.keypoint_buffer)})")
            
            pred, sentence, conf, stats = engine.process_frame(frame)
            
            print(f"[DEBUG] Prediction: '{pred}' | Sentence: '{sentence}' | Confidence: {conf:.3f}")

            with lock:
                latest_sentence = sentence
                if sentence:
                    latest_translation, latest_audio_base64, _ = tts_engine.process_sentence(sentence, target_lang=current_language)
                    print(f"[DEBUG] Translation: '{latest_translation}'")
                else:
                    latest_translation = ""
                    latest_audio_base64 = ""

        # Log every 100 frames to show it's running
        if frame_counter % 100 == 0:
            print(f"[INFO] Frames: {frame_counter} | Processed: {processed_counter} | Detection: {detection_active}")

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ================= Routes =================
@app.route('/')
def index():
    return app.send_static_file('index.html')  # Serve the responsive HTML

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_active
    data = request.json
    detection_active = data.get('active', False)
    print(f"[INFO] Detection toggled: {detection_active}")
    if not detection_active:
        engine.reset_sentence()
    return jsonify({'status':'ok', 'active': detection_active})

@app.route('/get_translation')
def get_translation():
    with lock:
        return jsonify({
            'sentence': latest_sentence,
            'translation': latest_translation,
            'audio_base64': latest_audio_base64
        })

@app.route('/set_language', methods=['POST'])
def set_language():
    global current_language
    data = request.json
    lang = data.get('language')
    if lang:
        current_language = lang
    print(f"[INFO] Language set to: {current_language}")
    return jsonify({'status':'ok', 'current_language': current_language})

# ================= Cleanup =================
def cleanup():
    webcam.release()
    engine.cleanup()

# ================= Main =================
if __name__ == '__main__':
    try:
        print("[INFO] Starting ASL Translation Server on port 8080...")
        print("[INFO] Open http://localhost:8080 in your browser")
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    finally:
        cleanup()
