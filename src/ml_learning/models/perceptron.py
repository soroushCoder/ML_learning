"""
Perceptron Classifier - Binary classifier with polynomial features.

This module implements a single perceptron with polynomial feature transformation
to create a curved (non-linear) decision boundary.
"""

import numpy as np
from pathlib import Path

from src.ml_learning.data.loaders import load_perceptron_data
from src.ml_learning.config.settings import PERCEPTRON_CONFIG
from src.ml_learning.utils.logging import setup_logger
from src.ml_learning.utils.visualization import plot_decision_boundary

logger = setup_logger(__name__)


def add_polynomial_features(X: np.ndarray) -> np.ndarray:
    """
    Add polynomial features to create non-linear decision boundary.

    Adds: x1^2, x2^2, and x1*x2

    Args:
        X: Input features (n_samples, 2)

    Returns:
        Augmented features (n_samples, 5)
    """
    x1_squared = X[:, 0] ** 2
    x2_squared = X[:, 1] ** 2
    x1_x2 = X[:, 0] * X[:, 1]
    return np.column_stack([X, x1_squared, x2_squared, x1_x2])


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.

    Args:
        x: Input values

    Returns:
        Sigmoid output (between 0 and 1)
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow


class PolynomialPerceptron:
    """Perceptron classifier with polynomial features."""

    def __init__(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        use_polynomial: bool = True,
    ):
        """
        Initialize the perceptron.

        Args:
            X: Training features (loads from file if None)
            y: Training labels (loads from file if None)
            use_polynomial: Whether to use polynomial features
        """
        if X is None or y is None:
            X, y = load_perceptron_data()
            logger.info(f"Loaded {len(X)} training samples")

        self.X_original = X
        self.y = y
        self.use_polynomial = use_polynomial

        # Transform features if using polynomial
        if self.use_polynomial:
            self.X = add_polynomial_features(X)
            logger.info(f"Features augmented: {X.shape[1]} → {self.X.shape[1]}")
        else:
            self.X = X

        # Initialize weights and bias
        self.w = np.zeros(self.X.shape[1])
        self.b = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features (n_samples, 2)

        Returns:
            Predictions between 0 and 1
        """
        # Transform features if using polynomial
        if self.use_polynomial:
            if X.shape[1] == 2:  # Original features
                X = add_polynomial_features(X)

        z = np.dot(X, self.w) + self.b
        return sigmoid(z)

    def train(
        self,
        learning_rate: float | None = None,
        epochs: int | None = None,
        print_every: int | None = None,
    ):
        """
        Train the perceptron using gradient descent.

        Args:
            learning_rate: Learning rate for weight updates
            epochs: Number of training epochs
            print_every: Print progress every N epochs
        """
        # Use config defaults if not provided
        if learning_rate is None:
            learning_rate = PERCEPTRON_CONFIG["learning_rate"]
        if epochs is None:
            epochs = PERCEPTRON_CONFIG["epochs"]
        if print_every is None:
            print_every = PERCEPTRON_CONFIG["print_every"]

        logger.info(f"Training for {epochs} epochs with learning rate {learning_rate}...")

        for epoch in range(epochs):
            # Train on each sample
            for i in range(len(self.X)):
                z = np.dot(self.X[i], self.w) + self.b
                y_pred = sigmoid(z)
                error = self.y[i] - y_pred

                # Update weights and bias
                self.w += learning_rate * error * self.X[i]
                self.b += learning_rate * error

            # Print progress
            if (epoch + 1) % print_every == 0:
                logger.info(f"Epoch {epoch + 1}: weights={self.w}, bias={self.b:.4f}")

        logger.info("Training complete!")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the model.

        Args:
            X: Test features
            y: True labels

        Returns:
            Dictionary with accuracy metrics
        """
        predictions = self.predict(X)
        predicted_classes = (predictions >= PERCEPTRON_CONFIG["decision_threshold"]).astype(int)
        accuracy = np.mean(predicted_classes == y)

        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "predicted_classes": predicted_classes,
        }

    def visualize(self, test_points: np.ndarray | None = None, save: bool = True):
        """
        Visualize the decision boundary.

        Args:
            test_points: Optional test points to plot
            save: Whether to save the plot
        """
        if test_points is None:
            test_points = np.array(PERCEPTRON_CONFIG["test_points"])

        save_path = PERCEPTRON_CONFIG["save_plot_path"] if save else None

        plot_decision_boundary(
            X=self.X_original,
            y=self.y,
            predict_fn=self.predict,
            test_points=test_points,
            title="Perceptron with Curved Decision Boundary\n(Using Polynomial Features)",
            save_path=save_path,
            figsize=PERCEPTRON_CONFIG["plot_figsize"],
            resolution=PERCEPTRON_CONFIG["contour_resolution"],
        )


def train_and_visualize() -> PolynomialPerceptron:
    """
    Train the perceptron and visualize results.

    Returns:
        Trained perceptron model
    """
    logger.info("=" * 60)
    logger.info("Polynomial Perceptron Classifier")
    logger.info("=" * 60)

    # Initialize and train
    perceptron = PolynomialPerceptron()
    perceptron.train()

    # Test the model
    test_points = np.array(PERCEPTRON_CONFIG["test_points"])

    logger.info("\nTest results:")
    for point in test_points:
        prediction = perceptron.predict(point.reshape(1, -1)).item()
        predicted_class = 1 if prediction >= PERCEPTRON_CONFIG["decision_threshold"] else 0
        logger.info(
            f"Point {point} → sigmoid={prediction:.4f}, class {predicted_class}"
        )

    # Visualize
    logger.info("\nGenerating visualization...")
    perceptron.visualize(test_points=test_points)

    return perceptron


if __name__ == "__main__":
    train_and_visualize()
