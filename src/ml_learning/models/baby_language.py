"""
Baby Language Model - Character-level LSTM model for generating baby-like text.

ARCHITECTURE:
    Input (text) → Embedding → LSTM (RNN) → LSTM (RNN) → Dense → Output (next character)

    Uses LSTM (Long Short-Term Memory), which is a type of RNN that's better at
    remembering long-term patterns in sequences.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

from src.ml_learning.data.loaders import load_baby_talk_dataset
from src.ml_learning.config.settings import BABY_LANGUAGE_CONFIG
from src.ml_learning.utils.logging import setup_logger

logger = setup_logger(__name__)


class BabyLanguageModel:
    """Character-level LSTM model for baby talk generation."""

    def __init__(self, texts: list[str] | None = None, seq_length: int | None = None):
        """
        Initialize the baby language model.

        Args:
            texts: List of baby-talk sentences (loads from file if None)
            seq_length: Number of characters to use as input sequence
        """
        if texts is None:
            texts = load_baby_talk_dataset()
            logger.info(f"Loaded {len(texts)} phrases from dataset")

        if seq_length is None:
            seq_length = BABY_LANGUAGE_CONFIG["seq_length"]

        self.seq_length = seq_length

        # Combine all texts and get unique characters
        self.text = '\n'.join(texts)
        self.chars = sorted(set(self.text))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

        logger.info(f"Total characters in dataset: {len(self.text)}")
        logger.info(f"Unique characters: {len(self.chars)}")
        logger.info(f"Characters: {''.join(self.chars)}")

        # Prepare training sequences
        self.X, self.y = self._prepare_sequences()

    def _prepare_sequences(self) -> tuple[np.ndarray, np.ndarray]:
        """Create input-output sequence pairs."""
        X, y = [], []

        for i in range(len(self.text) - self.seq_length):
            sequence = self.text[i:i + self.seq_length]
            target = self.text[i + self.seq_length]

            # Convert to indices
            X.append([self.char_to_idx[c] for c in sequence])
            y.append(self.char_to_idx[target])

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        logger.info(f"Training sequences: {len(X)}")
        return X, y

    def build_model(
        self,
        lstm_units: int | None = None,
        dropout: float | None = None,
        embedding_dim: int | None = None,
    ) -> keras.Model:
        """
        Build a simple LSTM model.

        Args:
            lstm_units: Number of LSTM units (neurons)
            dropout: Dropout rate for regularization
            embedding_dim: Dimension of embedding layer

        Returns:
            Compiled Keras model
        """
        # Use config defaults if not provided
        if lstm_units is None:
            lstm_units = BABY_LANGUAGE_CONFIG["lstm_units"]
        if dropout is None:
            dropout = BABY_LANGUAGE_CONFIG["dropout"]
        if embedding_dim is None:
            embedding_dim = BABY_LANGUAGE_CONFIG["embedding_dim"]

        model = keras.Sequential([
            layers.Embedding(
                input_dim=len(self.chars),
                output_dim=embedding_dim,
                input_length=self.seq_length
            ),
            layers.LSTM(lstm_units, return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(lstm_units),
            layers.Dropout(dropout),
            layers.Dense(len(self.chars), activation='softmax')
        ])

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        self.model = model
        logger.info("Model built successfully!")
        logger.info(f"\n{model.summary()}")
        return model

    def train(
        self,
        epochs: int | None = None,
        batch_size: int | None = None,
        patience: int | None = None,
    ):
        """
        Train the model.

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            patience: Early stopping patience

        Returns:
            Keras History object
        """
        if not hasattr(self, 'model'):
            self.build_model()

        # Use config defaults if not provided
        if epochs is None:
            epochs = BABY_LANGUAGE_CONFIG["epochs"]
        if batch_size is None:
            batch_size = BABY_LANGUAGE_CONFIG["batch_size"]
        if patience is None:
            patience = BABY_LANGUAGE_CONFIG["patience"]

        logger.info(f"Training for {epochs} epochs with batch size {batch_size}...")

        # Add early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=patience,
            restore_best_weights=True
        )

        history = self.model.fit(
            self.X, self.y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=1
        )

        return history

    def generate_text(
        self,
        seed_text: str = "me want",
        length: int | None = None,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate baby-like text.

        Args:
            seed_text: Starting text (should be at least seq_length chars)
            length: Number of characters to generate
            temperature: Higher = more random, lower = more predictable
                        (0.5 = conservative, 1.0 = balanced, 1.5 = creative)

        Returns:
            Generated text
        """
        if length is None:
            length = BABY_LANGUAGE_CONFIG["generation_length"]

        # Pad seed text if needed
        if len(seed_text) < self.seq_length:
            seed_text = seed_text.ljust(self.seq_length)

        generated = seed_text

        for _ in range(length):
            # Take the last seq_length characters
            sequence = generated[-self.seq_length:]

            # Convert to indices
            x = np.array([[self.char_to_idx[c] for c in sequence]])

            # Predict next character
            predictions = self.model.predict(x, verbose=0)[0]

            # Apply temperature
            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))

            # Sample from the predictions
            next_idx = np.random.choice(len(predictions), p=predictions)
            next_char = self.idx_to_char[next_idx]

            generated += next_char

        return generated

    def save_model(self, filepath: Path | str | None = None):
        """Save the trained model."""
        if filepath is None:
            filepath = BABY_LANGUAGE_CONFIG["model_save_path"]
        else:
            filepath = Path(filepath)

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(filepath)

        # Save character mappings
        chars_path = filepath.parent / f"{filepath.stem}_chars.npy"
        np.save(chars_path, {
            'chars': self.chars,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'seq_length': self.seq_length
        })

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Path | str | None = None):
        """Load a trained model."""
        if filepath is None:
            filepath = BABY_LANGUAGE_CONFIG["model_save_path"]
        else:
            filepath = Path(filepath)

        # Load model
        self.model = keras.models.load_model(filepath)

        # Load character mappings
        chars_path = filepath.parent / f"{filepath.stem}_chars.npy"
        data = np.load(chars_path, allow_pickle=True).item()
        self.chars = data['chars']
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = data['idx_to_char']
        self.seq_length = data['seq_length']

        logger.info(f"Model loaded from {filepath}")


def train_baby_model() -> BabyLanguageModel:
    """Train a new baby language model with default settings."""
    logger.info("=" * 60)
    logger.info("Baby Language Model Trainer")
    logger.info("=" * 60)

    # Initialize model with dataset
    baby_model = BabyLanguageModel()

    # Build and train
    baby_model.build_model()
    baby_model.train()

    # Generate some baby talk!
    logger.info("\n" + "=" * 60)
    logger.info("Generating Baby Talk!")
    logger.info("=" * 60)

    seed_phrases = BABY_LANGUAGE_CONFIG["default_seed_phrases"]
    temperatures = BABY_LANGUAGE_CONFIG["temperatures"]

    for seed in seed_phrases:
        logger.info(f"\nSeed: '{seed}'")
        logger.info("-" * 40)

        # Generate with different temperatures
        for temp in temperatures:
            generated = baby_model.generate_text(
                seed_text=seed,
                temperature=temp
            )
            logger.info(f"[temp={temp}] {generated}")

    # Save the model
    baby_model.save_model()

    return baby_model


def generate_baby_talk(prompt: str = "me want", num_sentences: int = 5):
    """
    Quick function to generate baby talk using pre-trained model.

    Args:
        prompt: Starting prompt
        num_sentences: Number of sentences to generate
    """
    # Load pre-trained model
    baby_model = BabyLanguageModel()
    baby_model.load_model()

    logger.info(f"Baby says (starting with '{prompt}'):\n")
    for i in range(num_sentences):
        text = baby_model.generate_text(prompt, length=30, temperature=1.0)
        # Extract sentence
        sentences = text.split('\n')
        logger.info(f"{i+1}. {sentences[0]}")


if __name__ == "__main__":
    # Train the model
    baby_model = train_baby_model()

    # After training, generate more examples
    logger.info("\n\n" + "=" * 60)
    logger.info("Extra Baby Talk Examples")
    logger.info("=" * 60)
    generate_baby_talk("me want", num_sentences=10)
