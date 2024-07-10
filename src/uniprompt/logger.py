import logging
import os
from datetime import datetime


def setup_logger(log_file=None):
    logger = logging.getLogger("UniPrompt")
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create file handler and set level to debug
    if log_file is None:
        logging_dir = "logs"
        os.makedirs(logging_dir, exist_ok=True)
        log_file = os.path.join(logging_dir, f"uniprompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    else:
        logging_dir = os.path.dirname(log_file)
        os.makedirs(logging_dir, exist_ok=True)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
