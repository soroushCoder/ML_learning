"""Centralized configuration for all ML models."""

from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PATHS = {
    "project_root": PROJECT_ROOT,
    "data_raw": DATA_DIR / "raw",
    "data_processed": DATA_DIR / "processed",
    "models_saved": MODELS_DIR / "saved",
    "models_checkpoints": MODELS_DIR / "checkpoints",
    "outputs_viz": OUTPUTS_DIR / "visualizations",
}


# Baby Language Model Configuration
BABY_LANGUAGE_CONFIG = {
    "seq_length": 15,
    "lstm_units": 64,
    "dropout": 0.2,
    "embedding_dim": 32,
    "epochs": 200,
    "batch_size": 32,
    "patience": 10,  # for early stopping
    "model_save_path": PATHS["models_saved"] / "baby_language" / "model.keras",
    "chars_save_path": PATHS["models_saved"] / "baby_language" / "chars.npy",
    # Text generation settings
    "default_seed_phrases": ["me want", "me love", "me see", "me play", "me eat"],
    "generation_length": 40,
    "temperatures": [0.5, 1.0, 1.5],
}


# Time Series Prediction (RNN) Configuration
PREDICTION_CONFIG = {
    "window_size": 12,  # Use last 12 months to predict next
    "rnn_units": 8,
    "input_dim": 1,
    "output_dim": 1,
    "learning_rate": 1e-2,
    "epochs": 300,
    "print_every": 100,
    "model_save_path": PATHS["models_saved"] / "prediction" / "rnn_model",
    # Data normalization will be done to [0, 1]
}


# Perceptron Configuration
PERCEPTRON_CONFIG = {
    "learning_rate": 0.01,
    "epochs": 100,
    "print_every": 10,
    "decision_threshold": 0.5,
    # Polynomial features: adds x1^2, x2^2, x1*x2
    "use_polynomial_features": True,
    # Visualization settings
    "plot_figsize": (10, 8),
    "contour_resolution": 200,
    "save_plot_path": PATHS["outputs_viz"] / "perceptron_boundary.png",
    # Test points
    "test_points": [
        [10, 10],
        [2, 1],
        [1, 4],
        [7, 9],
        [5, 5],
        [4, 2],
        [0, 0],
    ],
}


# Global settings
RANDOM_SEED = 42
VERBOSE = True
