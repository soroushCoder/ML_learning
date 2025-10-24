"""Tests for perceptron classifier."""

import pytest
import numpy as np
from src.ml_learning.models.perceptron import (
    add_polynomial_features,
    sigmoid,
    PolynomialPerceptron,
)


class TestPolynomialFeatures:
    """Test suite for polynomial features."""

    def test_add_polynomial_features(self):
        """Test polynomial feature transformation."""
        X = np.array([[1, 2], [3, 4]])
        X_poly = add_polynomial_features(X)

        # Should have 5 features: x1, x2, x1^2, x2^2, x1*x2
        assert X_poly.shape == (2, 5)

        # Check first row: [1, 2, 1, 4, 2]
        np.testing.assert_array_almost_equal(X_poly[0], [1, 2, 1, 4, 2])

        # Check second row: [3, 4, 9, 16, 12]
        np.testing.assert_array_almost_equal(X_poly[1], [3, 4, 9, 16, 12])


class TestSigmoid:
    """Test suite for sigmoid function."""

    def test_sigmoid_values(self):
        """Test sigmoid output values."""
        # Sigmoid of 0 should be 0.5
        assert sigmoid(np.array([0]))[0] == pytest.approx(0.5)

        # Sigmoid is bounded [0, 1]
        x = np.array([-1000, -10, 0, 10, 1000])
        y = sigmoid(x)

        assert np.all(y >= 0)
        assert np.all(y <= 1)

    def test_sigmoid_monotonic(self):
        """Test that sigmoid is monotonically increasing."""
        x = np.linspace(-10, 10, 100)
        y = sigmoid(x)

        # Check monotonicity
        assert np.all(np.diff(y) >= 0)


class TestPolynomialPerceptron:
    """Test suite for PolynomialPerceptron."""

    def test_initialization(self, sample_perceptron_data):
        """Test perceptron initialization."""
        X, y = sample_perceptron_data
        perceptron = PolynomialPerceptron(X=X, y=y, use_polynomial=True)

        assert perceptron.X.shape == (4, 5)  # Polynomial features
        assert perceptron.y.shape == (4,)
        assert perceptron.w.shape == (5,)

    def test_initialization_no_polynomial(self, sample_perceptron_data):
        """Test perceptron without polynomial features."""
        X, y = sample_perceptron_data
        perceptron = PolynomialPerceptron(X=X, y=y, use_polynomial=False)

        assert perceptron.X.shape == (4, 2)  # Original features
        assert perceptron.w.shape == (2,)

    def test_prediction_shape(self, sample_perceptron_data):
        """Test prediction output shape."""
        X, y = sample_perceptron_data
        perceptron = PolynomialPerceptron(X=X, y=y, use_polynomial=True)

        predictions = perceptron.predict(X)

        assert predictions.shape == (4,)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)

    def test_training(self, sample_perceptron_data):
        """Test perceptron training."""
        X, y = sample_perceptron_data
        perceptron = PolynomialPerceptron(X=X, y=y, use_polynomial=True)

        # Store initial weights
        initial_weights = perceptron.w.copy()

        # Train
        perceptron.train(epochs=10, learning_rate=0.01)

        # Weights should have changed
        assert not np.allclose(perceptron.w, initial_weights)

    def test_evaluation(self, sample_perceptron_data):
        """Test model evaluation."""
        X, y = sample_perceptron_data
        perceptron = PolynomialPerceptron(X=X, y=y, use_polynomial=True)

        perceptron.train(epochs=50)
        results = perceptron.evaluate(X, y)

        assert "accuracy" in results
        assert "predictions" in results
        assert "predicted_classes" in results
        assert 0 <= results["accuracy"] <= 1
