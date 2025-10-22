"""
🍼 Baby Language Model - Google Colab Ready Version 👶

Copy this entire file into a Google Colab cell and run!
It will train a tiny model that talks like a baby.

Usage in Google Colab:
1. Copy all this code
2. Paste into a new Colab cell
3. Run the cell (Shift+Enter)
4. Wait 2-3 minutes for training
5. See baby talk generated!
"""

# ============================================================================
# STEP 1: Install dependencies (run this first in Colab)
# ============================================================================

# Uncomment the line below if running in Google Colab for the first time:
# !pip install tensorflow numpy -q

# ============================================================================
# STEP 2: Import libraries
# ============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ============================================================================
# STEP 3: Your baby talk dataset (ADD YOUR OWN SENTENCES HERE!)
# ============================================================================

BABY_TALK = [
    "me want cookie", "me hungry now", "mommy me want milk",
    "daddy play with me", "me like toy", "me no like broccoli",
    "me want go park", "me sleepy now", "me love mommy",
    "me love daddy", "me want more cookie", "play ball with me",
    "me see doggy", "me pet kitty", "me want juice",
    "me no want nap", "me big boy now", "me big girl now",
    "me do it myself", "me help mommy", "me help daddy",
    "more milk please", "me want up", "me go potty",
    "me wash hands", "me brush teeth", "me wear shoes",
    "me pick this one", "me share toy", "me say sorry",
    "me good kid", "me eat apple", "me drink water",
    "me draw picture", "me sing song", "me read book",
    "me count one two three", "me know colors", "me see birdie",
    "me hear music", "me want hug", "me love you",
    "me happy today", "me sad now", "me tired",
    "me play outside", "me ride bike", "me swim pool",
    "me build blocks", "me paint", "me dance",
]

print(f"✅ Dataset loaded: {len(BABY_TALK)} baby sentences")

# ============================================================================
# STEP 4: Prepare the data
# ============================================================================

# Combine all text
text = '\n'.join(BABY_TALK)
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

print(f"✅ Unique characters: {len(chars)}")
print(f"✅ Total text length: {len(text)} characters")

# Create training sequences
SEQ_LENGTH = 15  # How many characters to look at
X, y = [], []

for i in range(len(text) - SEQ_LENGTH):
    sequence = text[i:i + SEQ_LENGTH]
    target = text[i + SEQ_LENGTH]
    X.append([char_to_idx[c] for c in sequence])
    y.append(char_to_idx[target])

X = np.array(X)
y = np.array(y)

print(f"✅ Training sequences created: {len(X)}")

# ============================================================================
# STEP 5: Build the neural network
# ============================================================================

model = keras.Sequential([
    layers.Embedding(
        input_dim=len(chars),
        output_dim=32,
        input_length=SEQ_LENGTH
    ),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(len(chars), activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\n✅ Model built!")
print(f"Total parameters: {model.count_params():,}")

# ============================================================================
# STEP 6: Train the model
# ============================================================================

print("\n🚀 Starting training...\n")

history = model.fit(
    X, y,
    batch_size=32,
    epochs=150,  # Adjust this (more = better, but slower)
    verbose=1
)

print("\n✅ Training complete!")
print(f"Final loss: {history.history['loss'][-1]:.4f}")
print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")

# ============================================================================
# STEP 7: Generate baby talk function
# ============================================================================

def generate_baby_talk(seed_text="me want", length=50, temperature=1.0):
    """
    Generate baby-like text!

    Args:
        seed_text: Starting phrase
        length: How many characters to generate
        temperature: 0.5=safe, 1.0=normal, 1.5=creative
    """
    # Pad seed text if needed
    if len(seed_text) < SEQ_LENGTH:
        seed_text = seed_text.ljust(SEQ_LENGTH)

    generated = seed_text

    for _ in range(length):
        # Get last SEQ_LENGTH characters
        sequence = generated[-SEQ_LENGTH:]

        # Convert to numbers
        x = np.array([[char_to_idx[c] for c in sequence]])

        # Predict next character
        predictions = model.predict(x, verbose=0)[0]

        # Apply temperature (controls randomness)
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        # Pick a character
        next_idx = np.random.choice(len(predictions), p=predictions)
        next_char = idx_to_char[next_idx]

        generated += next_char

    return generated

# ============================================================================
# STEP 8: Generate and display baby talk!
# ============================================================================

print("\n" + "=" * 60)
print("👶 BABY TALK GENERATOR 👶")
print("=" * 60)

# Try different starting phrases
seed_phrases = ["me want", "me love", "me see", "me play", "mommy", "daddy"]

for seed in seed_phrases:
    print(f"\n🌱 Seed: '{seed}'")
    print("-" * 60)

    # Generate with different creativity levels
    for temp in [0.5, 1.0, 1.5]:
        text = generate_baby_talk(seed, length=40, temperature=temp)
        # Clean up and show first line
        first_line = text.split('\n')[0]
        print(f"[creativity={temp:.1f}] {first_line}")

print("\n" + "=" * 60)

# ============================================================================
# STEP 9: Interactive generation (try your own prompts!)
# ============================================================================

print("\n🎮 Now try your own prompts!")
print("=" * 60)

# Change these to whatever you want!
custom_prompts = [
    "me hungry",
    "me tired",
    "me want play",
]

for prompt in custom_prompts:
    text = generate_baby_talk(prompt, length=35, temperature=1.0)
    print(f"\n👶 Input: '{prompt}'")
    print(f"💬 Says: {text.split(chr(10))[0]}")  # First sentence only

# ============================================================================
# STEP 10: Temperature comparison
# ============================================================================

print("\n\n🌡️ Temperature Comparison (same prompt, different creativity)")
print("=" * 60)

prompt = "me want"

print(f"\nPrompt: '{prompt}'\n")

print("🎯 Conservative (temp=0.5) - Plays it safe:")
print(generate_baby_talk(prompt, length=50, temperature=0.5))

print("\n⚖️ Balanced (temp=1.0) - Normal creativity:")
print(generate_baby_talk(prompt, length=50, temperature=1.0))

print("\n🎨 Creative (temp=1.5) - Wild and random:")
print(generate_baby_talk(prompt, length=50, temperature=1.5))

# ============================================================================
# STEP 11: Save the model (optional)
# ============================================================================

print("\n\n💾 Saving model...")
model.save('baby_language_model.keras')
np.save('baby_chars.npy', {
    'chars': chars,
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char
})
print("✅ Model saved! You can load it later.")

# ============================================================================
# BONUS: Fun generation patterns
# ============================================================================

print("\n\n🎉 BONUS: Baby Story Time!")
print("=" * 60)

story_prompts = [
    "me wake up",
    "me hungry",
    "me eat",
    "me play",
    "me tired",
]

print("\n📖 A Day in Baby's Life:\n")
for i, prompt in enumerate(story_prompts, 1):
    text = generate_baby_talk(prompt, length=30, temperature=0.8)
    sentence = text.split('\n')[0]  # First line only
    print(f"{i}. {sentence}")

print("\n" + "=" * 60)
print("🎊 All done! Your baby AI is ready to talk! 🎊")
print("=" * 60)

print("""
📝 Next steps:
1. Add more sentences to BABY_TALK list above
2. Train for more epochs (change epochs=150 to higher)
3. Try different temperatures (0.3 to 2.0)
4. Experiment with different seed phrases
5. Share your funniest generated sentences!
""")
