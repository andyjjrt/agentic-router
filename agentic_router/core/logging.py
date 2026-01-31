"""Logging configuration for the agentic_router package.

This module provides logging setup that only affects the agentic_router loggers,
without modifying external library loggers.
"""

import logging
import sys


def setup_logger(
    level: int = logging.INFO,
    debug: bool = False,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Set up logging for the agentic_router package only.

    This function configures logging specifically for the agentic_router namespace,
    without affecting external loggers (e.g., httpx, asyncio, etc.).

    Args:
        level: The logging level for the root agentic_router logger.
        debug: If True, sets the level to DEBUG regardless of the level argument.
        format_string: The format string for log messages.

    Returns:
        The configured agentic_router logger.
    """
    # Get the root logger for the agentic_router package
    logger = logging.getLogger("agentic_router")

    # Set the level
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create a handler that outputs to stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)  # Handler passes all messages, logger filters

    # Create and set the formatter
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Prevent propagation to the root logger
    logger.propagate = False

    return logger
