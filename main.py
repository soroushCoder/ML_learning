#!/usr/bin/env python3
"""
ML Learning Project - Main Entry Point

Interactive CLI for running different machine learning models:
1. Time Series Prediction (RNN for gold price forecasting)
2. Perceptron Classifier (with polynomial features)
3. Baby Language Model (LSTM for text generation)
"""

from src.ml_learning.models import baby_language, prediction, perceptron
from src.ml_learning.utils.logging import setup_logger

logger = setup_logger(__name__)


def run_prediction():
    """Run the time series prediction model."""
    logger.info("\n" + "=" * 60)
    logger.info("Time Series Prediction - Gold Price Forecasting")
    logger.info("=" * 60)

    try:
        predicted_price = prediction.train_and_predict()
        logger.info(f"\nPrediction complete! Next month's price: ${predicted_price:.2f}")
    except Exception as e:
        logger.error(f"Error running prediction module: {e}", exc_info=True)


def run_perceptron():
    """Run the perceptron classifier."""
    logger.info("\n" + "=" * 60)
    logger.info("Perceptron Classifier - Binary Classification")
    logger.info("=" * 60)

    try:
        model = perceptron.train_and_visualize()
        logger.info("\nPerceptron training and visualization complete!")
    except Exception as e:
        logger.error(f"Error running perceptron module: {e}", exc_info=True)


def run_baby_language():
    """Run the baby language model."""
    logger.info("\n" + "=" * 60)
    logger.info("Baby Language Model - Text Generation")
    logger.info("=" * 60)

    try:
        logger.info("\n[Step 1/4] Loading dataset and creating model...")
        baby = baby_language.BabyLanguageModel()

        logger.info("\n[Step 2/4] Building LSTM neural network...")
        baby.build_model()

        logger.info("\n[Step 3/4] Training (this may take a few minutes)...")
        baby.train()

        logger.info("\n[Step 4/4] Generating baby talk!")
        logger.info("=" * 60)
        logger.info("Baby Says:")
        logger.info("=" * 60)

        # Generate with different prompts
        prompts = ["me want", "me love", "me see", "mommy"]
        for prompt in prompts:
            text = baby.generate_text(prompt, length=40, temperature=0.5)
            logger.info(f"\n'{prompt}' → {text}")

        # Save the model
        logger.info("\n\nSaving model...")
        baby.save_model()
        logger.info("Model saved successfully!")

    except Exception as e:
        logger.error(f"Error running baby language model: {e}", exc_info=True)


def print_menu():
    """Print the main menu."""
    print("\n" + "=" * 60)
    print("ML Learning Project - Interactive Menu")
    print("=" * 60)
    print("\nAvailable modules:")
    print("1. Time Series Prediction (RNN - Gold Price Forecasting)")
    print("2. Perceptron Classifier (Curved Decision Boundary)")
    print("3. Baby Language Model (LSTM - Text Generation)")
    print("4. Exit")
    print("=" * 60)


def main():
    """Main entry point for the ML Learning Project."""
    logger.info("Starting ML Learning Project...")

    while True:
        print_menu()
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            run_prediction()
            break

        elif choice == "2":
            run_perceptron()
            break

        elif choice == "3":
            run_baby_language()
            break

        elif choice == "4":
            logger.info("\nExiting... Goodbye!")
            break

        else:
            logger.warning("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
