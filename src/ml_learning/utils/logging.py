"""Logging utilities for ML Learning package."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "ml_learning",
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_dir: Path | None = None,
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_to_file: Whether to also log to a file
        log_dir: Directory for log files (if log_to_file=True)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger
