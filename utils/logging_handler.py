import logging
import os
import sys
from pathlib import Path


def log_path_solver(log_path: str | os.PathLike | None) -> str | None:
    """
    Solves the log path by ensuring the directory exists.

    Args:
        log_path: The desired log file path.

    Returns:
        The resolved log file path or None if not provided.
    """
    if log_path is None:
        return None

    log_path_obj = Path(log_path).resolve()
    log_path_obj.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    return str(log_path_obj)


class NameFilter(logging.Filter):
    """A filter that filters out log records from specified logger names."""

    def __init__(self, names_to_filter: list[str]):
        super().__init__()
        self.names_to_filter = names_to_filter

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Determines if a log record should be processed.

        Args:
            record: The log record.

        Returns:
            False if the record's name starts with any of the filtered names,
            True otherwise.
        """
        if not self.names_to_filter:
            return True
        return not any(record.name.startswith(name) for name in self.names_to_filter)


class ColorFormatter(logging.Formatter):
    """A logging formatter that adds color to the output."""

    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__(fmt)
        self.FORMATS = {
            logging.DEBUG: self.GREY + fmt + self.RESET,
            logging.INFO: self.GREEN + fmt + self.RESET,
            logging.WARNING: self.YELLOW + fmt + self.RESET,
            logging.ERROR: self.RED + fmt + self.RESET,
            logging.CRITICAL: self.BOLD_RED + fmt + self.RESET,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_training_logger(
    name: str,
    stdout: bool = True,
    log_path: str | os.PathLike | None = None,
    filter_names: list[str] | None = None,
    level: int = logging.INFO,
    stdout_level: int | None = None,
    file_level: int | None = None,
) -> logging.Logger:
    """
    Gets a logger with specified handlers.

    Args:
        name: The name of the logger.
        stdout: If True, logs will be output to stdout.
        log_path: If specified, logs will be written to this file.
        filter_names: A list of logger names to filter out.
        level: The default logging level for handlers.
        stdout_level: The logging level for stdout. Defaults to `level`.
        file_level: The logging level for the log file. Defaults to `level`.

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels; handlers will filter
    logger.propagate = False  # Prevent double logging

    # Clear existing handlers
    logger.handlers.clear()

    base_format = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    plain_formatter = logging.Formatter(base_format)
    color_formatter = ColorFormatter(base_format)

    names_to_filter = filter_names or []
    name_filter = NameFilter(names_to_filter)

    if stdout:
        effective_stdout_level = stdout_level or level
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(color_formatter)
        handler.setLevel(effective_stdout_level)
        handler.addFilter(name_filter)
        logger.addHandler(handler)

    if log_path:
        log_path_solver(log_path)
        effective_file_level = file_level or level
        handler = logging.FileHandler(log_path)
        handler.setFormatter(plain_formatter)
        handler.setLevel(effective_file_level)
        handler.addFilter(name_filter)
        logger.addHandler(handler)

    return logger


def setup_root_logger(
    stdout: bool = True,
    log_path: str | None = None,
    filter_names: list[str] | None = None,
    level: int = logging.INFO,
    stdout_level: int | None = None,
    file_level: int | None = None,
):
    """
    Sets up the root logger.

    Args:
        stdout: If True, logs will be output to stdout.
        log_path: If specified, logs will be written to this file.
        filter_names: A list of logger names to filter out.
        level: The default logging level for handlers.
        stdout_level: The logging level for stdout. Defaults to `level`.
        file_level: The logging level for the log file. Defaults to `level`.
    """
    # Determine effective levels, defaulting to the main 'level'
    effective_stdout_level = stdout_level or level
    effective_file_level = file_level or level

    # Collect all active levels
    active_levels = []
    if stdout:
        active_levels.append(effective_stdout_level)
    if log_path:
        active_levels.append(effective_file_level)

    # Root logger must be set to the lowest level to not filter out messages
    # before they reach the handlers.
    min_level = min(active_levels) if active_levels else level

    logger = logging.getLogger()
    logger.setLevel(min_level)
    logger.handlers.clear()

    base_format = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    plain_formatter = logging.Formatter(base_format)
    color_formatter = ColorFormatter(base_format)

    names_to_filter = filter_names or []
    name_filter = NameFilter(names_to_filter)

    if stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(color_formatter)
        handler.setLevel(effective_stdout_level)
        handler.addFilter(name_filter)
        logger.addHandler(handler)

    if log_path:
        log_path_solver(log_path)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(plain_formatter)
        handler.setLevel(effective_file_level)
        handler.addFilter(name_filter)
        logger.addHandler(handler)
