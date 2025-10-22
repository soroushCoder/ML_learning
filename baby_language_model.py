"""
Baby Language Model - A tiny character-level LSTM model that generates baby-like text
Simple enough to train on Google Colab with a small custom dataset

ARCHITECTURE:
    Input (text) → Embedding → LSTM (RNN) → LSTM (RNN) → Dense → Output (next character)

    Yes, this uses RNN! Specifically LSTM (Long Short-Term Memory), which is a type of RNN
    that's better at remembering long-term patterns in sequences.

HOW TO RUN:
    Method 1 (Simplest):
        python baby_language_model.py

    Method 2 (In your own script):
        from baby_language_model import BabyLanguageModel, BABY_DATASET
        baby = BabyLanguageModel(BABY_DATASET)
        baby.build_model()
        baby.train(epochs=150)
        baby.generate_text("me want", length=50)

    Method 3 (Use the main function):
        from baby_language_model import main
        baby_model = main()
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

# ============================================================================
# STEP 1: Create your baby-talk dataset
# ============================================================================

BABY_DATASET = [
    # Original phrases
    "me want cookie",
    "me hungry now",
    "mommy me want milk",
    "daddy play with me",
    "me like toy",
    "me no like broccoli",
    "me want go park",
    "me sleepy now",
    "me love mommy",
    "me love daddy",
    "me want more cookie",
    "play ball with me",
    "me see doggy",
    "me pet kitty",
    "me want juice",
    "me no want nap",
    "me big boy now",
    "me big girl now",
    "me do it myself",
    "me help mommy",
    "me help daddy",
    "more milk please",
    "me want up",
    "me go potty",
    "me wash hands",
    "me brush teeth",
    "me wear shoes",
    "me pick this one",
    "me share toy",
    "me say sorry",
    "me good kid",
    "me eat apple",
    "me drink water",
    "me draw picture",
    "me sing song",
    "me read book",
    "me count one two three",
    "me know colors",
    "me see birdie",
    "me hear music",
    # New phrases - Food & Eating
    "me want banana",
    "me like pizza",
    "me eat sandwich",
    "me want snack",
    "me drink juice box",
    "me like yogurt",
    "me want cereal",
    "me eat strawberry",
    "me like cheese",
    "me want crackers",
    "me eat pasta",
    "me like chicken",
    "me want ice cream",
    "me full now",
    "me want more food",
    # New phrases - Play & Activities
    "me play outside",
    "me ride bike",
    "me jump high",
    "me run fast",
    "me swing on swing",
    "me slide down slide",
    "me build blocks",
    "me play with car",
    "me play with doll",
    "me color picture",
    "me paint picture",
    "me make tower",
    "me play hide seek",
    "me catch ball",
    "me throw ball",
    # New phrases - Animals
    "me see bunny",
    "me like puppy",
    "me see fishy",
    "me hear cow moo",
    "me see horsey",
    "me like turtle",
    "me see duck",
    "me hear rooster",
    "me pet hamster",
    "me see butterfly",
    # New phrases - Family & Friends
    "me hug mommy",
    "me kiss daddy",
    "me love grandma",
    "me love grandpa",
    "me play with sister",
    "me play with brother",
    "me miss mommy",
    "me see baby",
    "me be gentle",
    "me love family",
    # New phrases - Feelings & Emotions
    "me happy now",
    "me sad now",
    "me scared",
    "me brave",
    "me excited",
    "me tired",
    "me not feel good",
    "me feel better",
    "me proud",
    "me shy",
    # New phrases - Daily Routines
    "me wake up",
    "me get dressed",
    "me put on shirt",
    "me put on pants",
    "me tie shoes",
    "me comb hair",
    "me take bath",
    "me dry off",
    "me go bed",
    "me need blanket",
    # New phrases - Learning & Discovery
    "me learn abc",
    "me know numbers",
    "me count to ten",
    "me spell name",
    "me know shapes",
    "me find circle",
    "me see rainbow",
    "me know red",
    "me know blue",
    "me smart",
    # New phrases - Communication
    "me tell you secret",
    "me ask question",
    "me say thank you",
    "me say excuse me",
    "me say please",
    "me tell story",
    "me whisper",
    "me talk quiet",
    "me be nice",
    "me use words",
]

# ============================================================================
# STEP 2: Prepare the data
# ============================================================================

class BabyLanguageModel:
    def __init__(self, texts, seq_length=20):
        """
        Initialize the baby language model

        Args:
            texts: List of baby-talk sentences
            seq_length: Number of characters to use as input sequence
        """
        self.seq_length = seq_length

        # Combine all texts and get unique characters
        self.text = '\n'.join(texts)
        self.chars = sorted(set(self.text))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

        print(f"Total characters in dataset: {len(self.text)}")
        print(f"Unique characters: {len(self.chars)}")
        print(f"Characters: {''.join(self.chars)}")

        # Prepare training sequences
        self.X, self.y = self._prepare_sequences()

    def _prepare_sequences(self):
        """Create input-output sequence pairs"""
        X, y = [], []

        for i in range(len(self.text) - self.seq_length):
            sequence = self.text[i:i + self.seq_length]
            target = self.text[i + self.seq_length]

            # Convert to indices
            X.append([self.char_to_idx[c] for c in sequence])
            y.append(self.char_to_idx[target])

        # Convert to numpy arrays and normalize
        X = np.array(X)
        y = np.array(y)

        print(f"Training sequences: {len(X)}")
        return X, y

    def build_model(self, lstm_units=128, dropout=0.2):
        """
        Build a simple LSTM model

        Args:
            lstm_units: Number of LSTM units (neurons)
            dropout: Dropout rate for regularization
        """
        model = keras.Sequential([
            layers.Embedding(
                input_dim=len(self.chars),
                output_dim=32,
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
        print("\nModel built successfully!")
        print(model.summary())
        return model

    def train(self, epochs=100, batch_size=32):
        """
        Train the model

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        if not hasattr(self, 'model'):
            self.build_model()

        print(f"\nTraining for {epochs} epochs...")

        # Add early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
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

    def generate_text(self, seed_text="me want", length=50, temperature=1.0):
        """
        Generate baby-like text

        Args:
            seed_text: Starting text (should be at least seq_length chars)
            length: Number of characters to generate
            temperature: Higher = more random, lower = more predictable
                        (0.5 = conservative, 1.0 = balanced, 1.5 = creative)
        """
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

    def save_model(self, filepath='baby_language_model.keras'):
        """Save the trained model"""
        self.model.save(filepath)
        # Save character mappings
        np.save(filepath.replace('.keras', '_chars.npy'),
                {'chars': self.chars,
                 'char_to_idx': self.char_to_idx,
                 'idx_to_char': self.idx_to_char,
                 'seq_length': self.seq_length})
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='baby_language_model.keras'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        # Load character mappings
        data = np.load(filepath.replace('.keras', '_chars.npy'),
                      allow_pickle=True).item()
        self.chars = data['chars']
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = data['idx_to_char']
        self.seq_length = data['seq_length']
        print(f"Model loaded from {filepath}")


# ============================================================================
# STEP 3: Main training and generation script
# ============================================================================

def main():
    print("=" * 60)
    print("Baby Language Model Trainer")
    print("=" * 60)

    # Initialize model with dataset
    baby_model = BabyLanguageModel(BABY_DATASET, seq_length=15)

    # Build and train
    baby_model.build_model(lstm_units=64, dropout=0.2)
    baby_model.train(epochs=200, batch_size=32)

    # Generate some baby talk!
    print("\n" + "=" * 60)
    print("Generating Baby Talk!")
    print("=" * 60)

    seed_phrases = ["me want", "me love", "me see", "me play", "me eat"]

    for seed in seed_phrases:
        print(f"\nSeed: '{seed}'")
        print("-" * 40)

        # Generate with different temperatures
        for temp in [0.5, 1.0, 1.5]:
            generated = baby_model.generate_text(
                seed_text=seed,
                length=40,
                temperature=temp
            )
            print(f"[temp={temp}] {generated}")

    # Save the model
    baby_model.save_model('baby_language_model.keras')

    return baby_model


# ============================================================================
# STEP 4: Quick generation function for later use
# ============================================================================

def generate_baby_talk(prompt="me want", num_sentences=5):
    """
    Quick function to generate baby talk (after training)
    """
    # Load pre-trained model
    baby_model = BabyLanguageModel(BABY_DATASET)
    baby_model.load_model('baby_language_model.keras')

    print(f"Baby says (starting with '{prompt}'):\n")
    for i in range(num_sentences):
        text = baby_model.generate_text(prompt, length=30, temperature=1.0)
        # Extract sentence
        sentences = text.split('\n')
        print(f"{i+1}. {sentences[0]}")


if __name__ == "__main__":
    # Train the model
    baby_model = main()

    # After training, generate more examples
    print("\n\n" + "=" * 60)
    print("Extra Baby Talk Examples")
    print("=" * 60)
    generate_baby_talk("me want", num_sentences=10)
