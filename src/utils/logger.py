"""Logging configuration for the application."""

import os
import sys
import logging
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        format_string: Optional custom format string
    """
    # Remove default logger
    logger.remove()
    
    # Default format
    if not format_string:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=format_string,
        level=log_level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format_string,
            level=log_level,
            rotation="100 MB",
            retention="10 days",
            compression="zip"
        )
    
    # Set up standard logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Configure standard logging to use our handler
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Log startup message
    logger.info(f"Logging initialized - Level: {log_level}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")


def get_logger(name: str) -> "logger":
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Create convenience loggers for each module
class Loggers:
    """Container for module-specific loggers."""
    
    @staticmethod
    def pdf_processor() -> "logger":
        return get_logger("pdf_processor")
    
    @staticmethod
    def embeddings() -> "logger":
        return get_logger("embeddings")
    
    @staticmethod
    def vector_store() -> "logger":
        return get_logger("vector_store")
    
    @staticmethod
    def rag() -> "logger":
        return get_logger("rag")
    
    @staticmethod
    def llm() -> "logger":
        return get_logger("llm")
    
    @staticmethod
    def api() -> "logger":
        return get_logger("api")


# Example usage
if __name__ == "__main__":
    # Set up logging
    setup_logging(log_level="DEBUG", log_file="test.log")
    
    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test module-specific logger
    pdf_logger = Loggers.pdf_processor()
    pdf_logger.info("Processing PDF file")
    
    # Test exception logging
    try:
        1 / 0
    except Exception as e:
        logger.exception("An error occurred")