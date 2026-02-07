"""
Real-time ASL Recognition Flask Application
Supports dynamic switching between sentence-level and alphabet-level ASL recognition engines.
"""

import os
import sys
import threading
import logging
from typing import Optional, Tuple
from flask import Flask, render_template, Response, jsonify, request
import cv2
import base64
import numpy as np

# Import custom modules (assumed to be in the same directory)
from realtime_engine import RealtimeASLEngine, RealtimeAlphabetEngine
from translation_tts import TranslationTTS
from webcam_capture import WebcamCapture

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'asl-recognition-secret-key')

# ============================================================================
# Global Variables and Thread Locks
# ============================================================================

# Thread locks for safe access to shared variables
frame_lock = threading.Lock()
translation_lock = threading.Lock()
engine_lock = threading.Lock()

# Shared output variables (used by both engines)
latest_sentence = ""
latest_translation = ""
latest_audio_base64 = ""

# Detection state
detection_enabled = True

# Current language for translation
current_language = "es"  # Default: Spanish

# Engine selector: 'sentence' or 'alphabet'
current_engine_type = 'sentence'

# Engine instances
sentence_engine: Optional[RealtimeASLEngine] = None
alphabet_engine: Optional[RealtimeAlphabetEngine] = None

# Webcam capture instance
webcam: Optional[WebcamCapture] = None

# Translation/TTS instance
translator: Optional[TranslationTTS] = None

# Separate smoothing buffers for each engine
sentence_smoothing_buffer = []
alphabet_smoothing_buffer = []

# Smoothing configuration
SMOOTHING_WINDOW_SIZE = 5  # Number of frames to consider for smoothing
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for prediction acceptance

# ============================================================================
# Helper Functions
# ============================================================================

def get_active_engine():
    """
    Returns the currently active engine based on current_engine_type.
    Thread-safe with engine_lock.
    """
    with engine_lock:
        if current_engine_type == 'alphabet':
            return alphabet_engine
        else:
            return sentence_engine


def get_active_smoothing_buffer():
    """
    Returns the smoothing buffer for the currently active engine.
    """
    with engine_lock:
        if current_engine_type == 'alphabet':
            return alphabet_smoothing_buffer
        else:
            return sentence_smoothing_buffer


def set_active_smoothing_buffer(buffer):
    """
    Sets the smoothing buffer for the currently active engine.
    """
    global sentence_smoothing_buffer, alphabet_smoothing_buffer
    
    with engine_lock:
        if current_engine_type == 'alphabet':
            alphabet_smoothing_buffer = buffer
        else:
            sentence_smoothing_buffer = buffer


def smooth_predictions(predictions, buffer, window_size=SMOOTHING_WINDOW_SIZE):
    """
    Smooths predictions using a sliding window approach.
    
    Args:
        predictions: List of (label, confidence) tuples from current frame
        buffer: Smoothing buffer for the active engine
        window_size: Number of frames to consider
        
    Returns:
        Smoothed prediction string or empty string
    """
    # Add current predictions to buffer
    if predictions and len(predictions) > 0:
        buffer.append(predictions[0])  # Top prediction
    
    # Maintain buffer size
    if len(buffer) > window_size:
        buffer.pop(0)
    
    # If buffer not full enough, return empty
    if len(buffer) < window_size // 2:
        return ""
    
    # Count occurrences of each label
    label_counts = {}
    total_confidence = {}
    
    for label, confidence in buffer:
        if confidence >= CONFIDENCE_THRESHOLD:
            if label not in label_counts:
                label_counts[label] = 0
                total_confidence[label] = 0.0
            label_counts[label] += 1
            total_confidence[label] += confidence
    
    # Find most frequent label with highest average confidence
    if not label_counts:
        return ""
    
    best_label = max(
        label_counts.keys(),
        key=lambda x: (label_counts[x], total_confidence[x] / label_counts[x])
    )
    
    # Require majority vote
    if label_counts[best_label] >= window_size // 2:
        return best_label
    
    return ""


def cleanup_resources():
    """
    Cleanup all resources (engines, webcam, translator) on shutdown.
    """
    global sentence_engine, alphabet_engine, webcam, translator
    
    logger.info("Cleaning up resources...")
    
    if sentence_engine:
        try:
            sentence_engine.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up sentence engine: {e}")
    
    if alphabet_engine:
        try:
            alphabet_engine.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up alphabet engine: {e}")
    
    if webcam:
        try:
            webcam.release()
        except Exception as e:
            logger.error(f"Error releasing webcam: {e}")
    
    if translator:
        try:
            translator.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up translator: {e}")
    
    logger.info("Resource cleanup complete")


# ============================================================================
# Initialization
# ============================================================================

def initialize_engines():
    """
    Initialize both ASL recognition engines at startup.
    """
    global sentence_engine, alphabet_engine
    
    try:
        logger.info("Initializing sentence-level ASL engine...")
        sentence_engine = RealtimeASLEngine(
            model_path='sentence_model.h5',
            labels_path='sentence_labels.txt',
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        logger.info("Sentence engine initialized successfully")
        
        logger.info("Initializing alphabet-level ASL engine...")
        alphabet_engine = RealtimeAlphabetEngine(
            model_path='alphabet_model.h5',
            labels_path='alphabet_labels.txt',
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        logger.info("Alphabet engine initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing engines: {e}")
        return False


def initialize_webcam():
    """
    Initialize webcam capture.
    """
    global webcam
    
    try:
        logger.info("Initializing webcam...")
        webcam = WebcamCapture(camera_index=0, width=640, height=480)
        logger.info("Webcam initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing webcam: {e}")
        return False


def initialize_translator():
    """
    Initialize translation and TTS service.
    """
    global translator
    
    try:
        logger.info("Initializing translation/TTS service...")
        translator = TranslationTTS()
        logger.info("Translation/TTS service initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing translator: {e}")
        return False


# ============================================================================
# Frame Processing
# ============================================================================

def generate_frames():
    """
    Generator function that yields video frames with ASL predictions.
    Processes frames using the currently active engine.
    """
    global latest_sentence, latest_translation, latest_audio_base64
    global detection_enabled
    
    if not webcam:
        logger.error("Webcam not initialized")
        return
    
    logger.info("Starting frame generation...")
    
    while True:
        try:
            # Capture frame from webcam
            frame = webcam.read_frame()
            
            if frame is None:
                logger.warning("Failed to capture frame")
                continue
            
            # Process frame only if detection is enabled
            if detection_enabled:
                # Get the active engine
                active_engine = get_active_engine()
                
                if active_engine is None:
                    logger.error("No active engine available")
                    continue
                
                # Process frame through active engine
                try:
                    predictions, annotated_frame = active_engine.process_frame(frame)
                    
                    # Get active smoothing buffer
                    buffer = get_active_smoothing_buffer()
                    
                    # Apply smoothing
                    smoothed_prediction = smooth_predictions(predictions, buffer)
                    
                    # Update smoothing buffer
                    set_active_smoothing_buffer(buffer)
                    
                    # Update latest sentence if we have a valid prediction
                    if smoothed_prediction:
                        with translation_lock:
                            # For alphabet mode, append letters
                            if current_engine_type == 'alphabet':
                                # Avoid duplicate consecutive letters
                                if not latest_sentence.endswith(smoothed_prediction):
                                    latest_sentence += smoothed_prediction
                            else:
                                # For sentence mode, update the whole sentence
                                latest_sentence = smoothed_prediction
                    
                    # Use annotated frame if available, otherwise use original
                    output_frame = annotated_frame if annotated_frame is not None else frame
                    
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    output_frame = frame
            else:
                output_frame = frame
            
            # Add status overlay to frame
            output_frame = add_status_overlay(output_frame)
            
            # Encode frame as JPEG
            try:
                ret, buffer = cv2.imencode('.jpg', output_frame)
                if not ret:
                    logger.warning("Failed to encode frame")
                    continue
                
                frame_bytes = buffer.tobytes()
                
                # Yield frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except Exception as e:
                logger.error(f"Error encoding frame: {e}")
                continue
                
        except Exception as e:
            logger.error(f"Error in frame generation loop: {e}")
            continue


def add_status_overlay(frame):
    """
    Add status information overlay to the frame.
    """
    overlay = frame.copy()
    
    # Prepare status text
    engine_status = f"Engine: {current_engine_type.upper()}"
    detection_status = f"Detection: {'ON' if detection_enabled else 'OFF'}"
    
    # Add semi-transparent background
    cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Add text
    cv2.putText(frame, engine_status, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, detection_status, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return frame


# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """
    Render the main application page.
    """
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
    Video streaming route. Returns a multipart response with MJPEG frames.
    """
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/get_translation', methods=['GET'])
def get_translation():
    """
    Get the latest ASL sentence and its translation.
    Returns JSON with sentence, translation, and audio data.
    """
    global latest_sentence, latest_translation, latest_audio_base64
    
    with translation_lock:
        # Generate translation if sentence has changed and translator is available
        if latest_sentence and translator:
            try:
                # Translate the sentence
                translation = translator.translate(latest_sentence, target_language=current_language)
                
                # Generate TTS audio
                audio_data = translator.text_to_speech(translation, language=current_language)
                
                # Encode audio as base64
                if audio_data:
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    latest_audio_base64 = audio_base64
                    latest_translation = translation
                else:
                    latest_audio_base64 = ""
                    latest_translation = translation
                    
            except Exception as e:
                logger.error(f"Error generating translation: {e}")
                latest_translation = ""
                latest_audio_base64 = ""
        
        # Return current state
        return jsonify({
            'sentence': latest_sentence,
            'translation': latest_translation,
            'audio': latest_audio_base64,
            'engine': current_engine_type
        })


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """
    Toggle ASL detection on/off.
    """
    global detection_enabled
    
    detection_enabled = not detection_enabled
    logger.info(f"Detection {'enabled' if detection_enabled else 'disabled'}")
    
    return jsonify({
        'success': True,
        'detection_enabled': detection_enabled
    })


@app.route('/set_language', methods=['POST'])
def set_language():
    """
    Set the target language for translation.
    Expects JSON: {"language": "es"} (language code)
    """
    global current_language
    
    try:
        data = request.get_json()
        language = data.get('language', 'es')
        
        # Validate language code (basic check)
        if len(language) != 2:
            return jsonify({
                'success': False,
                'error': 'Invalid language code'
            }), 400
        
        current_language = language
        logger.info(f"Language set to: {language}")
        
        return jsonify({
            'success': True,
            'language': current_language
        })
        
    except Exception as e:
        logger.error(f"Error setting language: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/set_engine', methods=['POST'])
def set_engine():
    """
    Switch between sentence and alphabet recognition engines.
    Expects JSON: {"engine": "sentence"} or {"engine": "alphabet"}
    """
    global current_engine_type, latest_sentence
    
    try:
        data = request.get_json()
        engine_type = data.get('engine', 'sentence')
        
        # Validate engine type
        if engine_type not in ['sentence', 'alphabet']:
            return jsonify({
                'success': False,
                'error': 'Invalid engine type. Must be "sentence" or "alphabet"'
            }), 400
        
        # Check if requested engine is available
        with engine_lock:
            if engine_type == 'alphabet' and alphabet_engine is None:
                return jsonify({
                    'success': False,
                    'error': 'Alphabet engine not initialized'
                }), 500
            
            if engine_type == 'sentence' and sentence_engine is None:
                return jsonify({
                    'success': False,
                    'error': 'Sentence engine not initialized'
                }), 500
            
            # Switch engine
            current_engine_type = engine_type
        
        # Clear latest sentence when switching engines
        with translation_lock:
            latest_sentence = ""
        
        logger.info(f"Switched to {engine_type} engine")
        
        return jsonify({
            'success': True,
            'engine': current_engine_type
        })
        
    except Exception as e:
        logger.error(f"Error switching engine: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    """
    Clear the current sentence/translation.
    Useful for alphabet mode to start fresh.
    """
    global latest_sentence, latest_translation, latest_audio_base64
    
    with translation_lock:
        latest_sentence = ""
        latest_translation = ""
        latest_audio_base64 = ""
    
    logger.info("Sentence cleared")
    
    return jsonify({
        'success': True
    })


@app.route('/get_status', methods=['GET'])
def get_status():
    """
    Get current application status.
    """
    return jsonify({
        'detection_enabled': detection_enabled,
        'current_engine': current_engine_type,
        'current_language': current_language,
        'sentence_engine_available': sentence_engine is not None,
        'alphabet_engine_available': alphabet_engine is not None,
        'webcam_available': webcam is not None,
        'translator_available': translator is not None
    })


# ============================================================================
# Application Startup and Shutdown
# ============================================================================

@app.before_first_request
def startup():
    """
    Initialize all components before first request.
    """
    logger.info("Starting application initialization...")
    
    # Initialize engines
    if not initialize_engines():
        logger.error("Failed to initialize engines")
        sys.exit(1)
    
    # Initialize webcam
    if not initialize_webcam():
        logger.error("Failed to initialize webcam")
        sys.exit(1)
    
    # Initialize translator
    if not initialize_translator():
        logger.warning("Failed to initialize translator - translation features disabled")
    
    logger.info("Application initialization complete")


def shutdown():
    """
    Cleanup handler for application shutdown.
    """
    cleanup_resources()


# Register shutdown handler
import atexit
atexit.register(shutdown)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    try:
        logger.info("="*60)
        logger.info("Real-time ASL Recognition Application")
        logger.info("Dual Engine Mode: Sentence + Alphabet Recognition")
        logger.info("="*60)
        
        # Run Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production
            threaded=True,
            use_reloader=False  # Disable reloader to prevent double initialization
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        cleanup_resources()
    except Exception as e:
        logger.error(f"Application error: {e}")
        cleanup_resources()
        sys.exit(1)
