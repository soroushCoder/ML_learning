"""Visualization utilities for ML models."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    predict_fn,
    test_points: np.ndarray | None = None,
    title: str = "Decision Boundary",
    save_path: Path | None = None,
    figsize: tuple = (10, 8),
    resolution: int = 200,
):
    """
    Plot decision boundary for a binary classifier.

    Args:
        X: Training features (n_samples, 2)
        y: Training labels (n_samples,)
        predict_fn: Function that takes X and returns predictions
        test_points: Optional test points to visualize
        title: Plot title
        save_path: Path to save the plot (optional)
        figsize: Figure size
        resolution: Grid resolution for decision boundary
    """
    # Create mesh grid
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, resolution),
        np.linspace(x2_min, x2_max, resolution)
    )

    # Predict for all grid points
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    Z = predict_fn(grid_points)
    Z = Z.reshape(xx1.shape)

    # Create the plot
    plt.figure(figsize=figsize)

    # Plot decision boundary as contour
    plt.contourf(xx1, xx2, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['red', 'blue'])
    plt.contour(xx1, xx2, Z, levels=[0.5], colors='black', linewidths=2)

    # Plot training points
    passed = X[y == 1]
    failed = X[y == 0]
    plt.scatter(passed[:, 0], passed[:, 1], c='blue', marker='o',
                s=100, edgecolors='k', label='Class 1')
    plt.scatter(failed[:, 0], failed[:, 1], c='red', marker='x',
                s=100, linewidths=2, label='Class 0')

    # Plot test points if provided
    if test_points is not None:
        test_preds = predict_fn(test_points)
        for i, (point, pred) in enumerate(zip(test_points, test_preds)):
            color = 'blue' if pred >= 0.5 else 'red'
            label = 'Test Points' if i == 0 else ''
            plt.scatter(point[0], point[1], c=color, marker='s',
                       s=150, edgecolors='green', linewidths=3, label=label)

    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_training_history(
    history,
    metrics: list[str] = ['loss', 'accuracy'],
    title: str = "Training History",
    save_path: Path | None = None,
    figsize: tuple = (12, 4),
):
    """
    Plot training history from Keras model.

    Args:
        history: Keras History object or dict with metrics
        metrics: List of metrics to plot
        title: Plot title
        save_path: Path to save the plot (optional)
        figsize: Figure size
    """
    # Convert History object to dict if needed
    if hasattr(history, 'history'):
        history = history.history

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in history:
            ax.plot(history[metric], label=f'Training {metric}')
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'], label=f'Validation {metric}')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_time_series_prediction(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Time Series Prediction",
    save_path: Path | None = None,
    figsize: tuple = (12, 6),
):
    """
    Plot actual vs predicted values for time series.

    Args:
        actual: Actual values
        predicted: Predicted values
        title: Plot title
        save_path: Path to save the plot (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    time_steps = np.arange(len(actual))
    plt.plot(time_steps, actual, 'b-', label='Actual', linewidth=2)
    plt.plot(time_steps, predicted, 'r--', label='Predicted', linewidth=2)

    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()
