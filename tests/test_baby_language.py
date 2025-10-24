"""Tests for baby language model."""

import pytest
import numpy as np
from src.ml_learning.models.baby_language import BabyLanguageModel, train_baby_model


class TestBabyLanguageModel:
    """Test suite for BabyLanguageModel."""

    def test_initialization(self, sample_baby_phrases):
        """Test model initialization."""
        model = BabyLanguageModel(texts=sample_baby_phrases, seq_length=5)

        assert model.seq_length == 5
        assert len(model.chars) > 0
        assert len(model.char_to_idx) == len(model.chars)
        assert len(model.X) > 0
        assert len(model.y) > 0

    def test_sequence_preparation(self, sample_baby_phrases):
        """Test sequence preparation."""
        model = BabyLanguageModel(texts=sample_baby_phrases, seq_length=5)

        # Check shapes
        assert model.X.shape[1] == 5  # seq_length
        assert len(model.y) == len(model.X)

        # Check values are valid indices
        assert model.X.max() < len(model.chars)
        assert model.y.max() < len(model.chars)

    def test_model_building(self, sample_baby_phrases):
        """Test model building."""
        model = BabyLanguageModel(texts=sample_baby_phrases, seq_length=5)
        keras_model = model.build_model(lstm_units=32, dropout=0.2)

        assert keras_model is not None
        assert len(keras_model.layers) == 5  # Embedding + 2xLSTM + 2xDropout + Dense

    def test_text_generation(self, sample_baby_phrases):
        """Test text generation (without training)."""
        model = BabyLanguageModel(texts=sample_baby_phrases, seq_length=5)
        model.build_model(lstm_units=16)

        # Generate text (will be random without training)
        generated = model.generate_text(seed_text="me", length=10)

        assert len(generated) >= 10
        assert isinstance(generated, str)

    def test_character_mappings(self, sample_baby_phrases):
        """Test character to index mappings."""
        model = BabyLanguageModel(texts=sample_baby_phrases, seq_length=5)

        # Test bidirectional mapping
        for char, idx in model.char_to_idx.items():
            assert model.idx_to_char[idx] == char
