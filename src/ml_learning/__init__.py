"""
ML Learning Package - A collection of machine learning models for educational purposes.

This package contains implementations of:
- Baby Language Model (LSTM-based text generation)
- Time Series Prediction (RNN for gold price prediction)
- Perceptron Classifier (Binary classification with polynomial features)
"""

__version__ = "0.2.0"
__author__ = "ML Learning Project"

from src.ml_learning.models import baby_language, prediction, perceptron

__all__ = ["baby_language", "prediction", "perceptron"]
