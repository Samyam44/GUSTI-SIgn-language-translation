"""
ASL Sentence Recognition Model with BiLSTM + CTC architecture.
Handles variable-length sequences with Masking and CTC loss.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Dict
import json


def build_ctc_model(
    input_dim: int,
    lstm_units: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    num_classes: int = 28,  # 26 letters + space + blank
    dense_units: int = 128
) -> keras.Model:
    """
    Build BiLSTM-CTC model for ASL sentence recognition.
    
    Architecture:
        Input (batch, time, features)
        -> Masking
        -> BiLSTM layers with dropout
        -> Dense layer
        -> Softmax output (character probabilities per timestep)
    
    Args:
        input_dim: Number of input features per timestep
        lstm_units: Number of units in each LSTM layer
        num_layers: Number of BiLSTM layers
        dropout: Dropout rate
        num_classes: Number of output classes (including CTC blank)
        dense_units: Number of units in dense layer before output
    
    Returns:
        Keras Model with CTC architecture
    """
    # Input layer
    input_seq = layers.Input(shape=(None, input_dim), name='input_sequence')
    
    # Masking layer to handle variable-length sequences
    # Assumes padding value is 0.0
    x = layers.Masking(mask_value=0.0, name='masking')(input_seq)
    
    # BiLSTM layers
    for i in range(num_layers):
        x = layers.Bidirectional(
            layers.LSTM(
                lstm_units,
                return_sequences=True,
                dropout=0.0,  # Use separate dropout layers for better control
                name=f'lstm_{i+1}'
            ),
            name=f'bidirectional_{i+1}'
        )(x)
        
        # Dropout after each BiLSTM
        x = layers.Dropout(dropout, name=f'dropout_{i+1}')(x)
    
    # Dense layer before output
    x = layers.Dense(dense_units, activation='relu', name='dense_pre_output')(x)
    x = layers.Dropout(dropout, name='dropout_final')(x)
    
    # Output layer: character probabilities at each timestep
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = keras.Model(inputs=input_seq, outputs=output, name='ASL_CTC_Model')
    
    return model


def ctc_lambda_loss(y_true, y_pred):
    """
    CTC loss function for Keras.
    This is a wrapper that will be used with model.compile().
    
    Note: The actual CTC loss computation requires input_length and label_length,
    which will be provided during training through a custom training loop.
    
    This function is a placeholder for the model architecture.
    """
    return y_pred  # Placeholder; actual loss computed in training loop


class CTCModel:
    """
    Wrapper class for CTC model with training utilities.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize CTC model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.model = None
        self.char_to_idx = None
        self.idx_to_char = None
        
    def build(self, input_dim: int, num_classes: int):
        """
        Build the model architecture.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
        """
        self.model = build_ctc_model(
            input_dim=input_dim,
            lstm_units=self.config['model']['lstm_units'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            num_classes=num_classes,
            dense_units=self.config['model']['dense_units']
        )
        
        print("\nModel architecture:")
        self.model.summary()
        
    def compile_model(self, learning_rate: float = 0.001):
        """
        Compile model with optimizer.
        Note: Loss will be computed manually in training loop for CTC.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer)
        
    def save_model(self, filepath: str):
        """Save model weights."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights."""
        self.model = keras.models.load_model(filepath, compile=False)
        print(f"Model loaded from {filepath}")


def compute_ctc_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    input_lengths: np.ndarray,
    label_lengths: np.ndarray
) -> tf.Tensor:
    """
    Compute CTC loss for a batch.
    
    Args:
        y_true: Ground truth labels (batch, max_label_length)
        y_pred: Model predictions (batch, max_time, num_classes)
        input_lengths: Actual sequence lengths (batch,)
        label_lengths: Actual label lengths (batch,)
    
    Returns:
        CTC loss tensor
    """
    # Convert to tensors if needed
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    input_lengths = tf.cast(input_lengths, tf.int32)
    label_lengths = tf.cast(label_lengths, tf.int32)
    
    # Compute CTC loss
    # Note: y_pred should be (batch, time, num_classes)
    # y_true should be (batch, max_label_length)
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=label_lengths,
        logit_length=input_lengths,
        logits_time_major=False,
        blank_index=-1  # Last index is blank
    )
    
    return tf.reduce_mean(loss)


def greedy_decode(
    predictions: np.ndarray,
    idx_to_char: Dict[int, str],
    blank_index: int = -1
) -> str:
    """
    Greedy CTC decoding: take argmax at each timestep and collapse.
    
    Algorithm:
    1. Take argmax at each timestep
    2. Remove consecutive duplicates
    3. Remove blank tokens
    4. Convert indices to characters
    
    Args:
        predictions: Model output (time, num_classes) - probabilities
        idx_to_char: Index to character mapping
        blank_index: Index of CTC blank token (default: -1 for last index)
    
    Returns:
        Decoded string
    """
    # Take argmax at each timestep
    if blank_index == -1:
        blank_index = predictions.shape[1] - 1
    
    indices = np.argmax(predictions, axis=1)
    
    # Remove consecutive duplicates
    collapsed = []
    prev_idx = -1
    for idx in indices:
        if idx != prev_idx:
            collapsed.append(idx)
            prev_idx = idx
    
    # Remove blanks and convert to characters
    chars = []
    for idx in collapsed:
        if idx != blank_index and idx in idx_to_char:
            chars.append(idx_to_char[idx])
    
    return ''.join(chars)


def batch_decode(
    batch_predictions: np.ndarray,
    idx_to_char: Dict[int, str],
    blank_index: int = -1
) -> list:
    """
    Decode a batch of predictions.
    
    Args:
        batch_predictions: Model output (batch, time, num_classes)
        idx_to_char: Index to character mapping
        blank_index: Index of CTC blank token
    
    Returns:
        List of decoded strings
    """
    decoded_texts = []
    for i in range(batch_predictions.shape[0]):
        text = greedy_decode(batch_predictions[i], idx_to_char, blank_index)
        decoded_texts.append(text)
    
    return decoded_texts


if __name__ == "__main__":
    # Test model building
    print("Testing model architecture...")
    
    # Sample configuration
    config = {
        'model': {
            'lstm_units': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'dense_units': 128
        }
    }
    
    # Build model
    ctc_model = CTCModel(config)
    ctc_model.build(input_dim=258, num_classes=28)
    ctc_model.compile_model(learning_rate=0.001)
    
    print("\nModel built successfully!")
    print(f"Total parameters: {ctc_model.model.count_params():,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    time_steps = 50
    features = 258
    
    dummy_input = np.random.randn(batch_size, time_steps, features).astype(np.float32)
    output = ctc_model.model.predict(dummy_input, verbose=0)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum per timestep (should be ~1.0): {output[0].sum(axis=1)[:5]}")
    
    # Test greedy decoding
    print("\nTesting greedy decoding...")
    idx_to_char = {i: chr(ord('a') + i - 1) if i > 0 else ' ' for i in range(27)}
    idx_to_char[27] = '<blank>'
    
    sample_pred = output[0]  # (time, num_classes)
    decoded = greedy_decode(sample_pred, idx_to_char, blank_index=27)
    print(f"Decoded text: '{decoded}'")
