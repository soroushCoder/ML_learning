"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np


@pytest.fixture
def sample_baby_phrases():
    """Sample baby talk phrases for testing."""
    return [
        "me want cookie",
        "me love mommy",
        "me play ball",
    ]


@pytest.fixture
def sample_gold_prices():
    """Sample gold prices for testing."""
    return np.array([1800, 1850, 1900, 1950, 2000, 2050], dtype=np.float32)


@pytest.fixture
def sample_perceptron_data():
    """Sample data for perceptron testing."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.float32)
    return X, y
