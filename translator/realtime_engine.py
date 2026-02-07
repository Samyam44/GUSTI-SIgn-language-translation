"""
Realtime ASL Recognition Engines
Provides both sentence-level and alphabet-level ASL recognition.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional
import tensorflow as tf


class RealtimeASLEngine:
    """
    Sentence-level ASL recognition engine.
    Recognizes complete ASL phrases and sentences.
    """
    
    def __init__(self, model_path: str, labels_path: str, confidence_threshold: float = 0.6):
        """
        Initialize the sentence-level ASL recognition engine.
        
        Args:
            model_path: Path to the trained TensorFlow model (.h5 file)
            labels_path: Path to the labels file (.txt file)
            confidence_threshold: Minimum confidence threshold for predictions
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.labels = []
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load model and labels
        self._load_model(model_path)
        self._load_labels(labels_path)
    
    def _load_model(self, model_path: str):
        """Load the TensorFlow model."""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Sentence model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence model: {e}")
    
    def _load_labels(self, labels_path: str):
        """Load class labels from file."""
        try:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.labels)} sentence labels")
        except Exception as e:
            raise RuntimeError(f"Failed to load labels: {e}")
    
    def _extract_features(self, hand_landmarks) -> np.ndarray:
        """
        Extract features from hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Tuple[str, float]], Optional[np.ndarray]]:
        """
        Process a video frame and return ASL predictions.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Tuple of (predictions, annotated_frame)
            predictions: List of (label, confidence) tuples sorted by confidence
            annotated_frame: Frame with hand landmarks drawn (or None)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        predictions = []
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extract features
                features = self._extract_features(hand_landmarks)
                
                # Reshape for model input
                features = features.reshape(1, -1)
                
                # Get predictions from model
                if self.model:
                    model_predictions = self.model.predict(features, verbose=0)
                    
                    # Get top predictions
                    top_indices = np.argsort(model_predictions[0])[::-1][:5]
                    
                    for idx in top_indices:
                        label = self.labels[idx] if idx < len(self.labels) else f"Unknown_{idx}"
                        confidence = float(model_predictions[0][idx])
                        
                        if confidence >= self.confidence_threshold:
                            predictions.append((label, confidence))
        
        return predictions, annotated_frame
    
    def cleanup(self):
        """Cleanup resources."""
        if self.hands:
            self.hands.close()


class RealtimeAlphabetEngine:
    """
    Alphabet-level ASL recognition engine.
    Recognizes individual ASL letters (fingerspelling).
    """
    
    def __init__(self, model_path: str, labels_path: str, confidence_threshold: float = 0.6):
        """
        Initialize the alphabet-level ASL recognition engine.
        
        Args:
            model_path: Path to the trained TensorFlow model (.h5 file)
            labels_path: Path to the labels file (.txt file)
            confidence_threshold: Minimum confidence threshold for predictions
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.labels = []
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Alphabet typically uses one hand
            min_detection_confidence=0.7,  # Higher threshold for letters
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load model and labels
        self._load_model(model_path)
        self._load_labels(labels_path)
    
    def _load_model(self, model_path: str):
        """Load the TensorFlow model."""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Alphabet model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load alphabet model: {e}")
    
    def _load_labels(self, labels_path: str):
        """Load class labels from file."""
        try:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.labels)} alphabet labels")
        except Exception as e:
            raise RuntimeError(f"Failed to load labels: {e}")
    
    def _extract_features(self, hand_landmarks) -> np.ndarray:
        """
        Extract features from hand landmarks for alphabet recognition.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Extract raw coordinates
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        # Add relative angles and distances (helps with alphabet recognition)
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        # Calculate distances between fingertips and palm
        palm_center = landmarks_array[0]  # Wrist as reference
        for tip_idx in [4, 8, 12, 16, 20]:  # Thumb, Index, Middle, Ring, Pinky tips
            distance = np.linalg.norm(landmarks_array[tip_idx] - palm_center)
            features.append(distance)
        
        return np.array(features)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Tuple[str, float]], Optional[np.ndarray]]:
        """
        Process a video frame and return alphabet predictions.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Tuple of (predictions, annotated_frame)
            predictions: List of (label, confidence) tuples sorted by confidence
            annotated_frame: Frame with hand landmarks drawn (or None)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        predictions = []
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Extract features
                features = self._extract_features(hand_landmarks)
                
                # Reshape for model input
                features = features.reshape(1, -1)
                
                # Get predictions from model
                if self.model:
                    model_predictions = self.model.predict(features, verbose=0)
                    
                    # Get top predictions
                    top_indices = np.argsort(model_predictions[0])[::-1][:3]
                    
                    for idx in top_indices:
                        label = self.labels[idx] if idx < len(self.labels) else f"Unknown_{idx}"
                        confidence = float(model_predictions[0][idx])
                        
                        if confidence >= self.confidence_threshold:
                            predictions.append((label, confidence))
                
                # Add prediction overlay
                if predictions:
                    top_label, top_conf = predictions[0]
                    cv2.putText(
                        annotated_frame,
                        f"{top_label}: {top_conf:.2f}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3
                    )
        
        return predictions, annotated_frame
    
    def cleanup(self):
        """Cleanup resources."""
        if self.hands:
            self.hands.close()
