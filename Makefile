# Makefile for ML Learning Project
# This file provides convenient commands to manage and run the project

.PHONY: help install run clean test

# Default target - shows help
help:
	@echo "ML Learning Project - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "  make install    - Install all project dependencies"
	@echo "  make run        - Run the main program (installs deps if needed)"
	@echo "  make clean      - Remove Python cache files and temporary files"
	@echo "  make test       - Run tests (if available)"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. Run 'make install' to set up the project"
	@echo "  2. Run 'make run' to start the program"
	@echo ""

# Install dependencies using uv
install:
	@echo "Installing dependencies with uv..."
	@uv sync
	@echo "Installation complete!"

# Run the main program (installs dependencies first if needed)
run: install
	@echo "Starting ML Learning Project..."
	@echo ""
	@uv run python main.py

# Clean up Python cache files and temporary files
clean:
	@echo "Cleaning up temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete!"

# Run prediction module directly
predict: install
	@echo "Running Gold Price Prediction..."
	@echo "1" | uv run python main.py

# Run perceptron module directly
perceptron: install
	@echo "Running ML Perceptron..."
	@echo "2" | uv run python main.py

# Run tests (placeholder for future test implementation)
test:
	@echo "No tests configured yet."
	@echo "To add tests, create a tests/ directory and use pytest."
