"""Data loading utilities for ML Learning package."""

import numpy as np
from pathlib import Path


def get_data_path(filename: str) -> Path:
    """Get absolute path to data file."""
    # Get the project root (3 levels up from this file)
    project_root = Path(__file__).parent.parent.parent.parent
    return project_root / "data" / "raw" / filename


def load_baby_talk_dataset() -> list[str]:
    """
    Load baby talk dataset from file.

    Returns:
        List of baby talk phrases
    """
    filepath = get_data_path("baby_talk_dataset.txt")
    with open(filepath, 'r', encoding='utf-8') as f:
        # Read all lines and strip whitespace
        phrases = [line.strip() for line in f if line.strip()]
    return phrases


def load_gold_prices() -> np.ndarray:
    """
    Load gold price time series data from CSV.

    Returns:
        NumPy array of gold prices
    """
    filepath = get_data_path("gold_prices.csv")
    # Skip header row, load only the price column (index 1)
    prices = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=1, dtype=np.float32)
    return prices


def load_perceptron_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load perceptron training data from CSV.

    Returns:
        Tuple of (X, y) where:
        - X: Feature matrix (n_samples, 2)
        - y: Labels (n_samples,)
    """
    filepath = get_data_path("perceptron_data.csv")
    # Load all data
    data = np.loadtxt(filepath, delimiter=',', skiprows=1, dtype=np.float32)
    # Split into features and labels
    X = data[:, :2]  # First two columns
    y = data[:, 2]   # Last column
    return X, y
