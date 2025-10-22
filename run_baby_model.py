"""
Simple script to run the baby language model
This shows you exactly how to use baby_language_model.py
"""

from baby_language_model import BabyLanguageModel, BABY_DATASET

# ============================================================================
# METHOD 1: Quick Run (uses the main() function)
# ============================================================================

def quick_run():
    """Simplest way - just import and run main()"""
    from baby_language_model import main

    print("Running the complete training and generation...\n")
    baby_model = main()
    return baby_model


# ============================================================================
# METHOD 2: Step-by-Step (more control)
# ============================================================================

def step_by_step_run():
    """Run with full control over each step"""

    print("=" * 60)
    print("Step-by-Step Baby Model Training")
    print("=" * 60)

    # Step 1: Create the model
    print("\n[1/4] Creating model...")
    baby_model = BabyLanguageModel(BABY_DATASET, seq_length=15)

    # Step 2: Build the neural network
    print("\n[2/4] Building neural network...")
    baby_model.build_model(lstm_units=64, dropout=0.2)

    # Step 3: Train it
    print("\n[3/4] Training (this takes 2-3 minutes)...")
    baby_model.train(epochs=150, batch_size=32)

    # Step 4: Generate baby talk
    print("\n[4/4] Generating baby talk...")
    print("\n" + "=" * 60)
    print("Baby Says:")
    print("=" * 60)

    prompts = ["me want", "me love", "me see", "mommy"]

    for prompt in prompts:
        print(f"\n🌱 Starting with: '{prompt}'")
        text = baby_model.generate_text(prompt, length=40, temperature=1.0)
        print(f"👶 Generated: {text}")

    # Step 5: Save the model
    print("\n[5/4] Saving model...")
    baby_model.save_model('my_baby_model.keras')
    print("✅ Model saved!")

    return baby_model


# ============================================================================
# METHOD 3: Custom Training
# ============================================================================

def custom_training():
    """Customize everything - your own dataset, parameters, etc."""

    # Your own custom dataset
    MY_CUSTOM_DATASET = [
        "me want cookie",
        "me want milk",
        "me love mommy",
        "me love daddy",
        "me play toy",
        "me see doggy",
        "me pet kitty",
        # Add more here!
    ]

    print("Training with custom dataset...")

    # Create model with custom settings
    baby_model = BabyLanguageModel(
        texts=MY_CUSTOM_DATASET,
        seq_length=12  # Shorter sequence for small dataset
    )

    # Build smaller model for faster training
    baby_model.build_model(
        lstm_units=32,   # Smaller = faster
        dropout=0.1
    )

    # Quick training
    baby_model.train(
        epochs=100,      # Fewer epochs = faster
        batch_size=16
    )

    # Generate
    print("\nGenerating with custom model:")
    text = baby_model.generate_text("me want", length=30)
    print(text)

    return baby_model


# ============================================================================
# METHOD 4: Load Pre-trained Model (after training once)
# ============================================================================

def use_pretrained_model():
    """Load a previously trained model and generate text"""

    print("Loading pre-trained model...")

    # Create model instance (need this for the class methods)
    baby_model = BabyLanguageModel(BABY_DATASET)

    # Load the trained weights
    try:
        baby_model.load_model('my_baby_model.keras')

        # Now generate!
        print("\nGenerating from pre-trained model:")
        for i in range(5):
            text = baby_model.generate_text("me want", length=35, temperature=1.0)
            print(f"{i+1}. {text}")

        return baby_model

    except FileNotFoundError:
        print("❌ No pre-trained model found!")
        print("Run step_by_step_run() first to create one.")
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║        Baby Language Model - How to Run Guide         ║
    ╚════════════════════════════════════════════════════════╝

    Choose a method:

    1. quick_run()           - Fastest, uses built-in main()
    2. step_by_step_run()    - Full control, see each step
    3. custom_training()     - Use your own dataset
    4. use_pretrained_model() - Load and use saved model

    """)

    # Change this to run different methods:

    # Uncomment ONE of these lines:

    # baby_model = quick_run()              # Method 1
    baby_model = step_by_step_run()         # Method 2 (RECOMMENDED)
    # baby_model = custom_training()        # Method 3
    # baby_model = use_pretrained_model()   # Method 4

    print("\n\n✅ All done! Try generating more:")
    print(">>> baby_model.generate_text('me want', length=50)")
