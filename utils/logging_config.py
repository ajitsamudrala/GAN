
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def create_logger(name, filename='/var/log/gan.log', logger_format=None, max_bytes=10485760, backup_count=5):
    """
    :param name: str, name of the logger
    :param filename: str, path to store the logs
    :param logger_format: str, format for displaying logs
    :return: obj
    """
    log_dir = Path(filename).resolve().parents[0]
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if not logger_format:
        logger_format = '%(asctime)s:%(name)s:L%(lineno)d:%(levelname)s:%(message)s'
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    rotating_file_handler = RotatingFileHandler(filename, maxBytes=max_bytes, backupCount=backup_count)
    formatter = logging.Formatter(logger_format, datefmt='%Y-%m-%d %H:%M:%S')
    rotating_file_handler.setFormatter(formatter)
    logger.addHandler(rotating_file_handler)
