def main():
    print("=" * 50)
    print("ML Learning Project")
    print("=" * 50)
    print("\nAvailable modules:")
    print("1. Prediction (RNN - Gold Price Prediction)")
    print("2. ML Perceptron (Curved Decision Boundary)")
    print("3. Baby Language Model (RNN/LSTM - Text Generation)")
    print("4. Exit")

    while True:
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            print("\nRunning Prediction module...")
            try:
                import predictaion
                # The predictaion.py file runs automatically when imported
            except ImportError as e:
                print(f"Error importing predictaion module: {e}")
            except Exception as e:
                print(f"Error running predictaion module: {e}")
            break

        elif choice == "2":
            print("\nRunning ML Perceptron module...")
            try:
                import ML_Percepetron
                # The ML_Percepetron.py file runs automatically when imported
            except ImportError as e:
                print(f"Error importing ML_Percepetron module: {e}")
            except Exception as e:
                print(f"Error running ML_Percepetron module: {e}")
            break

        elif choice == "3":
            print("\nRunning Baby Language Model...")
            try:
                from baby_language_model import BabyLanguageModel, BABY_DATASET

                print("\n[Step 1/4] Creating baby language model...")
                baby = BabyLanguageModel(BABY_DATASET, seq_length=20)

                print("\n[Step 2/4] Building RNN/LSTM neural network...")
                baby.build_model(lstm_units=128, dropout=0.2)

                print("\n[Step 3/4] Training (this takes ~2-3 minutes)...")
                baby.train(epochs=400, batch_size=32)

                print("\n[Step 4/4] Generating baby talk!\n")
                print("=" * 50)
                print("Baby Says:")
                print("=" * 50)

                # Generate with different prompts
                prompts = ["me want", "me love", "me see", "mommy"]
                for prompt in prompts:
                    text = baby.generate_text(prompt, length=40, temperature=0.5)
                    print(f"\n🌱 '{prompt}' → {text}")

                # Save the model
                print("\n\n💾 Saving model...")
                baby.save_model('baby_language_model.keras')
                print("✅ Model saved successfully!")

            except ImportError as e:
                print(f"Error importing baby_language_model module: {e}")
                print("Make sure baby_language_model.py exists in the same directory.")
            except Exception as e:
                print(f"Error running baby_language_model module: {e}")
            break

        elif choice == "4":
            print("\nExiting... Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
