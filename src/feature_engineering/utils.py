"""Logging and output-folder utilities."""

from __future__ import annotations

import logging
from pathlib import Path

from config import LOG_FILE, RESULT_PATH


def create_folders() -> None:
    Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)


def setup_logger() -> logging.Logger:
    create_folders()

    logger = logging.getLogger("ecg_feature_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Pipeline started")
    return logger
