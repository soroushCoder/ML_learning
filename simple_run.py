"""
SIMPLEST way to run the baby model
Just run: python simple_run.py
"""

from baby_language_model import BabyLanguageModel, BABY_DATASET

# Create model
baby = BabyLanguageModel(BABY_DATASET, seq_length=15)

# Build it
baby.build_model(lstm_units=64)

# Train it
baby.train(epochs=150)

# Use it!
print("\n🎉 Baby says:")
print(baby.generate_text("me want", length=50))
print(baby.generate_text("me love", length=50))
print(baby.generate_text("mommy", length=50))
