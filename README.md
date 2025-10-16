# ML Learning Project

A machine learning project featuring gold price prediction using RNN and perceptron-based decision boundary visualization.

## Requirements

- Python >= 3.13
- Dependencies managed via `uv`

## Installation

1. Install dependencies:
```bash
uv sync
```

This will create a virtual environment at `.venv` and install all required packages:
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- tensorflow >= 2.13.0

## Running the Project

### Option 1: Using uv (Recommended)
```bash
uv run python main.py
```

### Option 2: Activate virtual environment first
```bash
source .venv/bin/activate
python main.py
```

## Available Modules

The project includes an interactive menu with the following options:

1. **Prediction (RNN - Gold Price Prediction)** - Runs the gold price prediction model using RNN
2. **ML Perceptron (Curved Decision Boundary)** - Visualizes decision boundaries with perceptron
3. **Exit** - Exit the program

## Usage

When you run the main script, you'll see an interactive menu. Simply enter the number corresponding to the module you want to run:

```
==================================================
ML Learning Project
==================================================

Available modules:
1. Prediction (RNN - Gold Price Prediction)
2. ML Perceptron (Curved Decision Boundary)
3. Exit

Enter your choice (1-3):
```

## Project Structure

- `main.py` - Main entry point with interactive menu
- `predictaion.py` - RNN-based gold price prediction module
- `ML_Percepetron.py` - Perceptron with curved decision boundary visualization