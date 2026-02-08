"""
Real-time ASL recognition engine with webcam processing,
buffering, and sentence building logic.
"""

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time
from typing import Optional, Tuple, List
import tensorflow as tf
from tensorflow import keras

from utils import load_char_map
from model import greedy_decode


def load_model_with_custom_objects(model_path: str) -> keras.Model:
    """
    Load ASL model with proper handling of custom TensorFlow operations.
    
    This function tries multiple strategies to load the model:
    1. Load with custom objects
    2. Load weights only if model has complex custom layers
    
    Args:
        model_path: Path to the .h5 model file
        
    Returns:
        Loaded Keras model
    """
    print(f"Attempting to load model from: {model_path}")
    
    # Strategy 1: Try loading weights only (most reliable for models with custom ops)
    try:
        print("Strategy: Building model architecture and loading weights...")
        from model import build_ctc_model
        
        # Build fresh model with standard architecture
        model = build_ctc_model(
            input_dim=258,      # Hand + pose keypoints
            lstm_units=128,
            num_layers=2,
            dropout=0.3,
            num_classes=28,
            dense_units=128
        )
        
        # Load only the weights (bypasses custom layer issues)
        model.load_weights(model_path)
        print(f"✓ Model architecture built and weights loaded successfully")
        return model
        
    except Exception as e1:
        print(f"⚠ Could not load weights: {e1}")
        print("Trying alternative strategy...")
    
    # Strategy 2: Try loading full model with custom objects
    try:
        print("Strategy: Loading full model with custom objects...")
        
        # Create wrapper classes for TensorFlow operations
        class NotEqualLayer(keras.layers.Layer):
            def call(self, inputs):
                return tf.math.not_equal(inputs[0], inputs[1])
        
        class EqualLayer(keras.layers.Layer):
            def call(self, inputs):
                return tf.math.equal(inputs[0], inputs[1])
        
        custom_objects = {
            'NotEqual': NotEqualLayer,
            'Equal': EqualLayer,
            'ReduceAny': lambda: tf.reduce_any,
            'ReduceAll': lambda: tf.reduce_all,
        }
        
        model = keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        print(f"✓ Model loaded with custom objects")
        return model
        
    except Exception as e2:
        print(f"✗ Failed to load model: {e2}")
        raise RuntimeError(f"Could not load model from {model_path}. Error: {e2}")


class RealtimeASLEngine:
    """
    Real-time ASL sentence recognition engine.
    Handles webcam input, keypoint extraction, buffering, and prediction.
    """
    
    def __init__(self, config: dict, model_path: str, char_map_path: str):
        """
        Initialize real-time engine.
        
        Args:
            config: Configuration dictionary
            model_path: Path to trained model
            char_map_path: Path to character mapping
        """
        self.config = config
        
        # Load model with custom objects handling
        print("Loading ASL recognition model...")
        self.model = load_model_with_custom_objects(model_path)
        
        # Load character mapping
        self.char_to_idx, self.idx_to_char = load_char_map(char_map_path)
        self.blank_index = len(self.char_to_idx) - 1
        
        # Initialize MediaPipe
        self._init_mediapipe()
        
        # Keypoint buffer for temporal context
        self.keypoint_buffer = deque(
            maxlen=config['processing']['buffer_size']
        )
        
        # Sentence building
        self.current_sentence = []
        self.last_prediction = ""
        self.stable_prediction_count = 0
        self.pause_counter = 0
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Statistics
        self.total_predictions = 0
        self.processing_times = deque(maxlen=30)
        
    def _init_mediapipe(self):
        """Initialize MediaPipe landmarkers."""
        print("Initializing MediaPipe...")
        
        # Hand landmarker
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path='hand_landmarker.task'
            ),
            num_hands=2,
            min_hand_detection_confidence=self.config['mediapipe']['hand_detection_confidence'],
            min_hand_presence_confidence=self.config['mediapipe']['hand_detection_confidence'],
            min_tracking_confidence=self.config['mediapipe']['hand_tracking_confidence']
        )
        
        # Pose landmarker
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path='pose_landmarker_full.task'
            ),
            min_pose_detection_confidence=self.config['mediapipe']['pose_detection_confidence'],
            min_pose_presence_confidence=self.config['mediapipe']['pose_detection_confidence'],
            min_tracking_confidence=self.config['mediapipe']['pose_tracking_confidence']
        )
        
        try:
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
            print("✓ MediaPipe initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe: {e}")
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose and hand keypoints from frame.
        
        Args:
            frame: RGB frame (H, W, 3)
        
        Returns:
            Keypoint vector (258,) or None if extraction fails
        """
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Extract landmarks
        hand_result = self.hand_landmarker.detect(mp_image)
        pose_result = self.pose_landmarker.detect(mp_image)
        
        # Initialize keypoint vector (33*4 + 21*3*2 = 258)
        keypoints = np.zeros(258, dtype=np.float32)
        
        # Fill pose landmarks (33 landmarks × 4 values)
        if pose_result.pose_landmarks:
            pose_landmarks = pose_result.pose_landmarks[0]
            for i, landmark in enumerate(pose_landmarks):
                keypoints[i*4:(i+1)*4] = [
                    landmark.x, landmark.y, landmark.z, landmark.visibility
                ]
        
        # Fill hand landmarks (2 hands × 21 landmarks × 3 values)
        offset = 132
        if hand_result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_result.hand_landmarks[:2]):
                hand_offset = offset + hand_idx * 63
                for i, landmark in enumerate(hand_landmarks):
                    keypoints[hand_offset + i*3:hand_offset + (i+1)*3] = [
                        landmark.x, landmark.y, landmark.z
                    ]
        
        return keypoints
    
    def predict_from_buffer(self) -> Tuple[str, float]:
        """
        Predict sign/sentence from current keypoint buffer.
        
        Returns:
            (predicted_text, confidence)
        """
        if len(self.keypoint_buffer) < self.config['processing']['min_frames_for_prediction']:
            return "", 0.0
        
        # Convert buffer to numpy array
        sequence = np.array(list(self.keypoint_buffer), dtype=np.float32)
        
        # Add batch dimension
        sequence_batch = np.expand_dims(sequence, axis=0)
        
        # Predict
        start_time = time.time()
        predictions = self.model.predict(sequence_batch, verbose=0)
        inference_time = time.time() - start_time
        
        self.processing_times.append(inference_time)
        
        # Decode
        predicted_text = greedy_decode(
            predictions[0],
            self.idx_to_char,
            blank_index=self.blank_index
        )
        
        # Calculate confidence (average max probability)
        confidence = np.mean(np.max(predictions[0], axis=1))
        
        return predicted_text, confidence
    
    def update_sentence(self, prediction: str, confidence: float) -> str:
        """
        Update sentence based on prediction with stability checking.
        
        Args:
            prediction: Current prediction
            confidence: Prediction confidence
        
        Returns:
            Current complete sentence
        """
        min_confidence = self.config['processing']['confidence_threshold']
        pause_threshold = self.config['processing']['gesture_pause_frames']
        
        # Filter low confidence predictions
        if confidence < min_confidence:
            self.pause_counter += 1
            if self.pause_counter > pause_threshold:
                self.stable_prediction_count = 0
                self.last_prediction = ""
            return ' '.join(self.current_sentence)
        
        # Reset pause counter
        self.pause_counter = 0
        
        # Check prediction stability
        if prediction == self.last_prediction and prediction:
            self.stable_prediction_count += 1
        else:
            self.stable_prediction_count = 1
            self.last_prediction = prediction
        
        # Add to sentence if stable and not already present
        if self.stable_prediction_count >= 5:  # Stable for 5 frames
            if prediction and (not self.current_sentence or 
                              self.current_sentence[-1] != prediction):
                self.current_sentence.append(prediction)
                self.stable_prediction_count = 0
                self.total_predictions += 1
        
        return ' '.join(self.current_sentence)
    
    def reset_sentence(self):
        """Reset current sentence."""
        self.current_sentence = []
        self.last_prediction = ""
        self.stable_prediction_count = 0
        self.pause_counter = 0
        self.keypoint_buffer.clear()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[str, str, float, dict]:
        """
        Process single frame: extract keypoints, predict, update sentence.
        
        Args:
            frame: BGR frame from webcam
        
        Returns:
            (current_prediction, sentence, confidence, stats)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract keypoints
        keypoints = self.extract_keypoints(frame_rgb)
        
        # Add to buffer
        if keypoints is not None:
            self.keypoint_buffer.append(keypoints)
        
        # Predict if buffer is sufficient
        prediction, confidence = self.predict_from_buffer()
        
        # Update sentence
        sentence = self.update_sentence(prediction, confidence)
        
        # Update FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        
        # Statistics
        stats = {
            'fps': round(self.fps, 1),
            'buffer_size': len(self.keypoint_buffer),
            'confidence': round(confidence, 3),
            'total_predictions': self.total_predictions,
            'avg_inference_time': round(np.mean(self.processing_times) * 1000, 1) if self.processing_times else 0
        }
        
        return prediction, sentence, confidence, stats
    
    def draw_info(self, frame: np.ndarray, prediction: str, sentence: str, 
                  confidence: float, stats: dict) -> np.ndarray:
        """
        Draw information overlay on frame.
        
        Args:
            frame: Input frame
            prediction: Current prediction
            sentence: Complete sentence
            confidence: Prediction confidence
            stats: Statistics dictionary
        
        Returns:
            Frame with overlay
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Semi-transparent background for text
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Current prediction
        cv2.putText(frame, f"Current: {prediction}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Sentence
        cv2.putText(frame, f"Sentence: {sentence}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Confidence bar
        bar_width = int(confidence * 200)
        cv2.rectangle(frame, (10, 90), (210, 110), (50, 50, 50), -1)
        color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
        cv2.rectangle(frame, (10, 90), (10 + bar_width, 110), color, -1)
        cv2.putText(frame, f"{confidence:.2f}", (220, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Stats
        stats_text = (
            f"FPS: {stats['fps']} | "
            f"Buffer: {stats['buffer_size']} | "
            f"Inference: {stats['avg_inference_time']}ms"
        )
        cv2.putText(frame, stats_text, (10, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()


class WebcamCapture:
    """
    Webcam capture utility with frame skipping.
    """
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480,
                 frame_skip: int = 2):
        """
        Initialize webcam capture.
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            frame_skip: Process every nth frame
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame_skip = frame_skip
        self.frame_count = 0
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame with skipping.
        
        Returns:
            (should_process, frame)
        """
        ret, frame = self.cap.read()
        
        if not ret:
            return False, None
        
        self.frame_count += 1
        
        # Skip frames for performance
        should_process = (self.frame_count % self.frame_skip == 0)
        
        return should_process, frame
    
    def release(self):
        """Release camera."""
        self.cap.release()


if __name__ == "__main__":
    """
    Standalone test of real-time engine.
    """
    import json
    
    # Load configuration
    with open('realtime_config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize engine
    engine = RealtimeASLEngine(
        config=config,
        model_path=config['model']['model_path'],
        char_map_path=config['model']['char_map_path']
    )
    
    # Initialize webcam
    webcam = WebcamCapture(
        camera_id=config['webcam']['camera_id'],
        width=config['webcam']['width'],
        height=config['webcam']['height'],
        frame_skip=config['processing']['frame_skip']
    )
    
    print("\n" + "=" * 60)
    print("Real-time ASL Recognition - Standalone Test")
    print("=" * 60)
    print("Controls:")
    print("  Press 'r' to reset sentence")
    print("  Press 'q' to quit")
    print("=" * 60 + "\n")
    
    try:
        while True:
            should_process, frame = webcam.read()
            
            if frame is None:
                break
            
            if should_process:
                # Process frame
                prediction, sentence, confidence, stats = engine.process_frame(frame)
                
                # Draw overlay
                frame = engine.draw_info(frame, prediction, sentence, confidence, stats)
            
            # Display
            cv2.imshow('ASL Recognition', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                engine.reset_sentence()
                print("Sentence reset")
    
    finally:
        webcam.release()
        cv2.destroyAllWindows()
        engine.cleanup()
        print("\nEngine stopped")
