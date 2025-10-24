import logging
import sys

def setup_logger():
    """
    Sets up a centralized logger that writes to both a file for debugging
    and the console for general information.
    """
    # Define the format for our log messages
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # --- File Handler ---
    # This handler writes all messages (DEBUG level and above) to a file.
    # mode='w' ensures the log is fresh for each run.
    file_handler = logging.FileHandler("data/data_prep.log", mode='w')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)

    # --- Console Handler ---
    # This handler writes informational messages (INFO level and above) to the console.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # Get the root logger.
    # All other loggers in the application will inherit from this.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # Capture all messages at the root level

    # Add the handlers to the logger, but only if they haven't been added before.
    # This prevents duplicate log messages if the setup is called more than once.
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Create a logger instance that can be imported directly by other modules.
logger = setup_logger()