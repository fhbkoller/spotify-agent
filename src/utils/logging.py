# src/utils/logging.py
import logging
import sys
import os # <-- Import os

# Define the log file path
LOG_FILE_PATH = "data/data_prep.log"

def setup_logger():
    """
    Sets up a centralized logger that writes to both a file for debugging
    and the console for general information.
    """
    # --- FIX: Ensure the log directory exists ---
    log_dir = os.path.dirname(LOG_FILE_PATH)
    if log_dir: # Check if the path is not empty
        os.makedirs(log_dir, exist_ok=True)
    # --- End FIX ---

    # Define the format for our log messages
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # --- File Handler ---
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # Get the root logger.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # Capture all messages at the root level

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Create a logger instance that can be imported directly by other modules.
logger = setup_logger()