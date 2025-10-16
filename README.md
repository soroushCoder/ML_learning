# ML Learning Project

A beginner-friendly collection of machine learning examples featuring RNN-based gold price prediction and perceptron models.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [Available Modules](#available-modules)
- [Makefile Commands](#makefile-commands)
- [Troubleshooting](#troubleshooting)
- [Learning Resources](#learning-resources)

## Prerequisites

Before running this project, you need:

### 1. Python 3.8 or higher
- Check if installed: `python --version` or `python3 --version`
- Download from: https://www.python.org/downloads/

### 2. uv (Python package manager)
Install uv based on your operating system:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Quick Start

### Easiest Way: Using Make

If you have `make` installed (comes with macOS/Linux, install on Windows via Git Bash or WSL):

```bash
# Install dependencies and run the project in one command
make run
```

That's it! Everything will be set up automatically.

### Alternative: Manual Setup

1. **Navigate to the project directory:**
   ```bash
   cd ML_learning
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Run the project:**
   ```bash
   uv run python main.py
   ```

## Running the Project

When you run the project, you'll see an interactive menu:

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

Simply type `1`, `2`, or `3` and press Enter.

## Available Modules

### 1. Prediction (RNN - Gold Price Prediction)

**What it does:**
- Uses a Recurrent Neural Network (RNN) to predict future gold prices
- Trains on 36 months of historical data
- Uses the last 12 months to predict the next month's price

**How it works:**
1. Loads historical gold price data
2. Scales prices to [0,1] range for better training
3. Trains an RNN for 300 epochs (learning iterations)
4. Makes a prediction for the next month

**Example output:**
```
epoch 100 loss 0.0012345
epoch 200 loss 0.0008234
epoch 300 loss 0.0005123

Predicted NEXT month's gold price: 2110.50
```

**Key parameters you can modify in `predictaion.py`:**
- `WINDOW = 12`: Number of months to look back
- `HIDDEN_UNITS = 8`: RNN memory size
- `epochs = 300`: Number of training iterations

### 2. ML Perceptron (Curved Decision Boundary)

**What it does:**
- Demonstrates how a perceptron classifies data
- Shows decision boundaries (how the model separates different classes)
- Great for understanding basic neural network concepts

### 3. Exit

- Exits the application gracefully

## Project Structure

```
ML_learning/
├── main.py                 # Main entry point with interactive menu
├── predictaion.py          # RNN gold price prediction module
├── ML_Percepetron.py       # Perceptron module
├── pyproject.toml          # Project configuration and dependencies
├── uv.lock                 # Locked dependency versions
├── Makefile                # Automation scripts (makes life easier!)
└── README.md               # This file
```

## Makefile Commands

The Makefile provides convenient shortcuts:

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Install all project dependencies |
| `make run` | Install dependencies and run the main program |
| `make predict` | Run the prediction module directly |
| `make perceptron` | Run the perceptron module directly |
| `make clean` | Remove Python cache files |

**Examples:**
```bash
# See all available commands
make help

# Just install dependencies
make install

# Run the prediction module without the menu
make predict

# Clean up cache files
make clean
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "No module named 'numpy'" or similar errors

**Problem:** Dependencies are not installed.

**Solution:**
```bash
uv sync
# or
make install
```

#### 2. "uv: command not found"

**Problem:** uv is not installed or not in your PATH.

**Solution:**
- Install uv using the instructions in [Prerequisites](#prerequisites)
- Restart your terminal after installation

#### 3. "python: command not found"

**Problem:** Python is not installed or the command is different on your system.

**Solution:**
- Try `python3` instead of `python`
- Or install Python from python.org

#### 4. TensorFlow installation issues

**Problem:** TensorFlow may not support your Python version.

**Solution:**
- Use Python 3.8-3.11 (TensorFlow may not support 3.12+)
- Try: `uv sync --refresh` to reinstall dependencies
- Check compatibility: https://www.tensorflow.org/install

#### 5. Dimension mismatch errors in prediction

**Problem:** This was a bug in the RNN implementation (now fixed).

**Solution:**
- Make sure you have the latest version of `predictaion.py`
- The `step` function should properly handle batch matrix multiplication

#### 6. Make command not found (Windows)

**Problem:** `make` is not available on Windows by default.

**Solution:**
- Use Git Bash (comes with Git for Windows)
- Or use Windows Subsystem for Linux (WSL)
- Or run commands manually (see [Alternative: Manual Setup](#alternative-manual-setup))

## Dependencies

This project uses:

- **TensorFlow** (>= 2.13.0): Deep learning framework
- **NumPy** (>= 1.24.0): Numerical computing
- **Matplotlib** (>= 3.7.0): Plotting and visualization

All dependencies are automatically managed by `uv` and specified in `pyproject.toml`.

## Learning Resources

### Understanding RNNs (Recurrent Neural Networks)

RNNs are special neural networks designed for sequential data:
- They process data step-by-step (like reading a sentence word by word)
- They maintain a "hidden state" - a memory of what they've seen
- Perfect for: time series, text, speech, video

**Key concept:** Each step uses the current input AND the previous hidden state.

### RNN Architecture in This Project

```
Input (x_t) ───┐
               ├──> [RNN Cell] ───> Output (y_t)
Hidden (h_t-1)─┘         │
                         └──> Hidden (h_t)
```

**The math:**
- `h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t)` (update memory)
- `y_t = W_hy @ h_t` (make prediction)

### Key Concepts Explained

1. **Window Size (WINDOW = 12)**
   - How much history to look at
   - 12 = use 12 months to predict month 13

2. **Hidden Units (HIDDEN_UNITS = 8)**
   - Size of the RNN's memory
   - More units = more capacity, but slower training

3. **Epochs (300)**
   - One epoch = one pass through all training data
   - More epochs = better learning, but risk of overfitting

4. **Scaling**
   - Normalizes prices to [0,1]
   - Helps neural networks train faster and better

5. **Loss Function (MSE)**
   - Mean Squared Error
   - Measures how far predictions are from actual values
   - Lower = better

## Next Steps

### For Beginners

1. **Run the examples** to see them work
2. **Read the code** in `predictaion.py` with comments
3. **Modify parameters**:
   - Change `WINDOW` to use more/fewer months
   - Increase epochs to see if accuracy improves
4. **Try your own data** - replace the price array with real data

### For Advanced Users

1. **Add validation set** to prevent overfitting
2. **Implement LSTM** (Long Short-Term Memory) for better performance
3. **Add more features** (trading volume, market indicators)
4. **Save/load models** using TensorFlow's save API
5. **Add visualization** of predictions vs actual prices

## Contributing

This is a learning project. Feel free to:
- Experiment with the code
- Add new features
- Share improvements
- Use it for educational purposes

## License

This is an educational project. Free to use and modify!