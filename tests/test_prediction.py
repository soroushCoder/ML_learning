"""Tests for time series prediction model."""

import pytest
import numpy as np
import tensorflow as tf
from src.ml_learning.models.prediction import (
    CustomRNNCell,
    GoldPricePredictor,
    train_and_predict,
)


class TestCustomRNNCell:
    """Test suite for CustomRNNCell."""

    def test_initialization(self):
        """Test RNN cell initialization."""
        cell = CustomRNNCell(rnn_units=8, input_dim=1, output_dim=1)

        assert cell.rnn_units == 8
        assert cell.W_xh.shape == (8, 1)
        assert cell.W_hh.shape == (8, 8)
        assert cell.W_hy.shape == (1, 8)

    def test_reset_state(self):
        """Test state reset."""
        cell = CustomRNNCell(rnn_units=8, input_dim=1, output_dim=1)
        cell.reset_state(batch_size=4)

        assert cell.h.shape == (4, 8, 1)
        assert tf.reduce_all(cell.h == 0)

    def test_step(self):
        """Test single RNN step."""
        cell = CustomRNNCell(rnn_units=8, input_dim=1, output_dim=1)
        cell.reset_state(batch_size=2)

        x_t = tf.ones((2, 1, 1), dtype=tf.float32)
        y_t = cell.step(x_t)

        assert y_t.shape == (2, 1, 1)


class TestGoldPricePredictor:
    """Test suite for GoldPricePredictor."""

    def test_initialization(self, sample_gold_prices):
        """Test predictor initialization."""
        predictor = GoldPricePredictor(prices=sample_gold_prices, window_size=3)

        assert predictor.window_size == 3
        assert len(predictor.prices) == 6
        assert predictor.p_min == 1800.0
        assert predictor.p_max == 2050.0

    def test_data_preparation(self, sample_gold_prices):
        """Test data preparation."""
        predictor = GoldPricePredictor(prices=sample_gold_prices, window_size=3)

        # With 6 prices and window=3, we get 3 samples (indices 0-2, 1-3, 2-4)
        # Training uses all but last: 2 samples
        assert len(predictor.X_train) == 2
        assert len(predictor.y_train) == 2
        assert predictor.X_last.shape == (1, 3)

    def test_normalization(self, sample_gold_prices):
        """Test price normalization."""
        predictor = GoldPricePredictor(prices=sample_gold_prices, window_size=3)

        # Check normalization to [0, 1]
        assert predictor.prices_scaled.min() >= 0
        assert predictor.prices_scaled.max() <= 1

    def test_prediction_shape(self, sample_gold_prices):
        """Test prediction output shape."""
        predictor = GoldPricePredictor(prices=sample_gold_prices, window_size=3)

        # Run sequence on last window
        X_last_tf = tf.convert_to_tensor(predictor.X_last, dtype=tf.float32)
        output = predictor.run_sequence(X_last_tf)

        assert output.shape == (1, 1)

    def test_predict_next(self, sample_gold_prices):
        """Test next price prediction."""
        predictor = GoldPricePredictor(prices=sample_gold_prices, window_size=3)

        # Train briefly
        predictor.train(epochs=10)

        # Predict
        pred = predictor.predict_next()

        assert isinstance(pred, (float, np.floating))
        assert pred > 0  # Price should be positive
