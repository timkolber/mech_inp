import logging
import os
from pathlib import Path

log_output_directory = Path(__file__).parent.parent.parent / "outputs" / "logs"


def setup_logging(
    log_filename: str = "main.log", log_level: int = logging.INFO
) -> logging.Logger:
    """Set up logging to console and file"""
    log_format = "%(asctime)s - %(filename)s:%(funcName)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    os.makedirs(log_output_directory, exist_ok=True)

    logger = logging.getLogger("main")
    logger.setLevel(log_level)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(log_format, date_format))

        fh = logging.FileHandler(str(log_output_directory / log_filename), mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(log_format, date_format))

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
