"""
Time Series Prediction Model - Custom RNN for gold price prediction.

This module implements a minimal RNN with explicit weight matrices for educational purposes.
The model predicts the next month's gold price based on a sliding window of historical prices.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

from src.ml_learning.data.loaders import load_gold_prices
from src.ml_learning.config.settings import PREDICTION_CONFIG
from src.ml_learning.utils.logging import setup_logger

logger = setup_logger(__name__)


class CustomRNNCell(tf.keras.layers.Layer):
    """
    Custom RNN cell with explicit weight matrices.

    This implements the basic RNN equations:
        h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t)
        y_t = W_hy @ h_t
    """

    def __init__(self, rnn_units: int, input_dim: int, output_dim: int):
        """
        Initialize the RNN cell.

        Args:
            rnn_units: Number of hidden units
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super().__init__()

        # Initialize weight matrices
        # glorot_uniform: smart way to pick starting weights for smooth learning
        self.W_xh = self.add_weight(
            shape=(rnn_units, input_dim),
            initializer="glorot_uniform",
            name="W_xh"
        )
        self.W_hh = self.add_weight(
            shape=(rnn_units, rnn_units),
            initializer="orthogonal",
            name="W_hh"
        )
        self.W_hy = self.add_weight(
            shape=(output_dim, rnn_units),
            initializer="glorot_uniform",
            name="W_hy"
        )

        self.rnn_units = rnn_units

    def reset_state(self, batch_size: int):
        """Reset hidden state to zeros for new sequence."""
        self.h = tf.zeros((batch_size, self.rnn_units, 1), dtype=tf.float32)

    def step(self, x_t: tf.Tensor) -> tf.Tensor:
        """
        Perform one RNN step.

        Args:
            x_t: Input at time t, shape (batch, input_dim, 1)

        Returns:
            Output at time t, shape (batch, output_dim, 1)
        """
        h_prev = self.h  # (batch, rnn, 1)

        # Reshape for proper broadcasting
        h_prev_2d = tf.squeeze(h_prev, axis=-1)  # (batch, rnn)
        x_t_2d = tf.squeeze(x_t, axis=-1)  # (batch, input_dim)

        # Matrix multiplications: W @ x where W is (out, in) and x is (batch, in)
        whh_h = tf.matmul(h_prev_2d, self.W_hh, transpose_b=True)  # (batch, rnn)
        wxh_x = tf.matmul(x_t_2d, self.W_xh, transpose_b=True)  # (batch, rnn)

        # Update hidden state: h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t)
        h_new = tf.math.tanh(whh_h + wxh_x)  # (batch, rnn)
        self.h = tf.expand_dims(h_new, axis=-1)  # (batch, rnn, 1)

        # Output: y_t = W_hy @ h_t
        y_t = tf.matmul(h_new, self.W_hy, transpose_b=True)  # (batch, output_dim)
        y_t = tf.expand_dims(y_t, axis=-1)  # (batch, output_dim, 1)

        return y_t


class GoldPricePredictor:
    """Gold price prediction model using custom RNN."""

    def __init__(
        self,
        prices: np.ndarray | None = None,
        window_size: int | None = None,
    ):
        """
        Initialize the predictor.

        Args:
            prices: Array of historical prices (loads from file if None)
            window_size: Number of past months to use for prediction
        """
        if prices is None:
            prices = load_gold_prices()
            logger.info(f"Loaded {len(prices)} historical prices")

        if window_size is None:
            window_size = PREDICTION_CONFIG["window_size"]

        self.prices = prices.astype(np.float32)
        self.window_size = window_size

        # Normalize prices to [0, 1]
        self.p_min = self.prices.min()
        self.p_max = self.prices.max()
        self.prices_scaled = (self.prices - self.p_min) / (self.p_max - self.p_min + 1e-9)

        # Prepare training data
        self.X_train, self.y_train, self.X_last = self._prepare_data()

        # Initialize RNN cell
        self.cell = CustomRNNCell(
            rnn_units=PREDICTION_CONFIG["rnn_units"],
            input_dim=PREDICTION_CONFIG["input_dim"],
            output_dim=PREDICTION_CONFIG["output_dim"],
        )

        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(PREDICTION_CONFIG["learning_rate"])

    def _prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data using sliding window.

        Returns:
            Tuple of (X_train, y_train, X_last)
        """
        X, y = [], []

        # Build (X, y): X[i] -> window_size months, y[i] -> next month
        for i in range(len(self.prices_scaled) - self.window_size):
            X.append(self.prices_scaled[i:i + self.window_size])
            y.append(self.prices_scaled[i + self.window_size])

        X = np.array(X, dtype=np.float32)  # (N, window_size)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)  # (N, 1)

        # Last window to predict the NEXT (unseen) month
        X_train, y_train = X[:-1], y[:-1]  # All pairs except the last one
        X_last = X[-1:]  # Last row for future prediction

        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Window size: {self.window_size}")

        return X_train, y_train, X_last

    def run_sequence(self, x_seq: tf.Tensor) -> tf.Tensor:
        """
        Run RNN on a sequence.

        Args:
            x_seq: Input sequence (batch, window_size)

        Returns:
            Last output (batch, 1)
        """
        batch = tf.shape(x_seq)[0]
        self.cell.reset_state(batch)

        # Reshape to (batch, window_size, 1, 1)
        x_seq = tf.reshape(x_seq, (batch, self.window_size, 1, 1))

        # Feed one month at a time
        for t in range(self.window_size):
            y_t = self.cell.step(x_seq[:, t])  # (batch, 1, 1)

        return tf.reshape(y_t, (batch, 1))  # Return last step output

    @tf.function
    def train_step(self, xb: tf.Tensor, yb: tf.Tensor) -> tf.Tensor:
        """
        Perform one training step.

        Args:
            xb: Batch of input sequences
            yb: Batch of target values

        Returns:
            Loss value
        """
        with tf.GradientTape() as tape:
            pred = self.run_sequence(xb)
            loss = tf.reduce_mean((pred - yb) ** 2)

        grads = tape.gradient(loss, self.cell.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.cell.trainable_variables))

        return loss

    def train(self, epochs: int | None = None, print_every: int | None = None):
        """
        Train the model.

        Args:
            epochs: Number of training epochs
            print_every: Print loss every N epochs
        """
        if epochs is None:
            epochs = PREDICTION_CONFIG["epochs"]
        if print_every is None:
            print_every = PREDICTION_CONFIG["print_every"]

        logger.info(f"Training for {epochs} epochs...")

        # Convert to tensors
        Xb = tf.convert_to_tensor(self.X_train, dtype=tf.float32)
        yb = tf.convert_to_tensor(self.y_train, dtype=tf.float32)

        for epoch in range(epochs):
            loss = self.train_step(Xb, yb)

            if (epoch + 1) % print_every == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.numpy():.6f}")

        logger.info("Training complete!")

    def predict_next(self) -> float:
        """
        Predict the next month's price.

        Returns:
            Predicted price (denormalized)
        """
        X_last_tf = tf.convert_to_tensor(self.X_last, dtype=tf.float32)
        pred_scaled = self.run_sequence(X_last_tf).numpy()[0, 0]

        # Denormalize
        pred_price = pred_scaled * (self.p_max - self.p_min) + self.p_min

        return pred_price


def train_and_predict() -> float:
    """
    Train the gold price predictor and make a prediction.

    Returns:
        Predicted next month's price
    """
    logger.info("=" * 60)
    logger.info("Gold Price Prediction Model")
    logger.info("=" * 60)

    # Initialize predictor
    predictor = GoldPricePredictor()

    # Train
    predictor.train()

    # Predict next month
    predicted_price = predictor.predict_next()

    logger.info("\n" + "=" * 60)
    logger.info(f"Predicted NEXT month's gold price: ${predicted_price:.2f}")
    logger.info("=" * 60)

    return predicted_price


if __name__ == "__main__":
    train_and_predict()
