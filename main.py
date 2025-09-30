# In main.py

import sys
import logging
from spotify_agent import IntelligentShuffler
from dotenv import load_dotenv

def setup_logging():
    """
    Configures a robust, explicit logger.
    - File log captures EVERYTHING (DEBUG level) and is explicitly UTF-8.
    - Console log shows INFO and above.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 1. File Handler: Explicitly set encoding to UTF-8
    file_handler = logging.FileHandler('spotify_agent.log', mode='w', encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # 2. Console Handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def main():
    load_dotenv()
    setup_logging()
    
    logger = logging.getLogger(__name__)
    
    if len(sys.argv) < 2:
        logger.critical("Usage: python main.py <spotify_playlist_url_or_id>")
        sys.exit(1)

    playlist_input = sys.argv[1]
    
    if "playlist/" in playlist_input:
        playlist_id = playlist_input.split("playlist/")[1].split("?")[0]
    else:
        playlist_id = playlist_input

    logger.info(f"Targeting playlist ID: {playlist_id}")

    agent = IntelligentShuffler(playlist_id)
    
    try:
        agent.initialize()
        agent.run()
    except KeyboardInterrupt:
        logger.info("\nShutting down agent. Goodbye!")
    except Exception:
        # exc_info=True will print the full stack trace to the console and the log file.
        logger.critical("A FATAL ERROR OCCURRED. The application will now exit.", exc_info=True)

if __name__ == "__main__":
    main()