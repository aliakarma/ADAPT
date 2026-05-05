"""
AgHealth+ — Logging Configuration
Centralised loguru setup for the entire system.
"""
import os
import sys
from pathlib import Path
from loguru import logger


def setup_logging(log_dir: str = "results/logs", level: str = "INFO") -> None:
    """Configure loguru sinks: stderr + rotating file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger.remove()  # Remove default sink

    # Console sink — clean format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level:<8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> — "
               "<level>{message}</level>",
        level=level,
        colorize=True,
    )

    # File sink — structured for audit trail
    logger.add(
        os.path.join(log_dir, "aghealth_{time:YYYY-MM-DD}.log"),
        rotation="1 day",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} — {message}",
        level="DEBUG",
        enqueue=True,
    )

    logger.info("AgHealth+ logging initialised (level={})", level)


def get_logger(name: str):
    """Return a named logger bound to a module/component."""
    return logger.bind(component=name)
