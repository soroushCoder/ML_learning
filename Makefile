# Makefile for ML Learning Project
# This file provides convenient commands to manage and run the project

.PHONY: help install install-dev run clean test predict perceptron baby-model

# Default target - shows help
help:
	@echo "ML Learning Project - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install dev dependencies (includes pytest)"
	@echo ""
	@echo "Running Models:"
	@echo "  make run           - Run interactive menu"
	@echo "  make predict       - Run gold price prediction directly"
	@echo "  make perceptron    - Run perceptron classifier directly"
	@echo "  make baby-model    - Run baby language model directly"
	@echo ""
	@echo "Development:"
	@echo "  make test          - Run all tests with pytest"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make clean         - Remove cache files and artifacts"
	@echo "  make clean-all     - Remove cache, models, and outputs"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. Run 'make install-dev' to set up the project"
	@echo "  2. Run 'make test' to verify everything works"
	@echo "  3. Run 'make run' to start the interactive menu"
	@echo ""

# Install production dependencies
install:
	@echo "Installing production dependencies..."
	@uv sync --no-dev
	@echo "Installation complete!"

# Install dev dependencies (includes pytest)
install-dev:
	@echo "Installing development dependencies..."
	@uv sync
	@uv pip install pytest pytest-cov
	@echo "Development environment ready!"

# Run the main program
run: install
	@echo "Starting ML Learning Project..."
	@echo ""
	@uv run python main.py

# Run tests with pytest
test: install-dev
	@echo "Running tests..."
	@uv run pytest -v

# Run tests with coverage
test-cov: install-dev
	@echo "Running tests with coverage..."
	@uv run pytest --cov=src/ml_learning --cov-report=term-missing --cov-report=html

# Clean up Python cache files and temporary files
clean:
	@echo "Cleaning up cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf htmlcov/ .coverage
	@echo "Cleanup complete!"

# Clean everything including models and outputs
clean-all: clean
	@echo "Removing trained models and outputs..."
	@rm -rf models/saved/*/
	@rm -rf outputs/visualizations/*
	@echo "Deep cleanup complete!"

# Run prediction module directly
predict: install
	@echo "Running Gold Price Prediction..."
	@echo "1" | uv run python main.py

# Run perceptron module directly
perceptron: install
	@echo "Running Perceptron Classifier..."
	@echo "2" | uv run python main.py

# Run baby language model directly
baby-model: install
	@echo "Running Baby Language Model..."
	@echo "3" | uv run python main.py
