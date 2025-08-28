import logging
import os
from datetime import datetime
import sys # Import sys for StreamHandler

# Name of the folder where logs will be stored
LOG_DIR = "logs"
# Create the log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Build a log file path like: logs/log_2023-03-15.log (changes daily)
LOG_FILE = os.path.join(
    LOG_DIR,
    f"log_{datetime.now().strftime('%Y-%m-%d')}.log"
)

# Configure the ROOT logger once for the whole application
# Create a formatter to define the log message format
formatter = logging.Formatter('%(asctime)s %(levelname)s - %(module)s - %(message)s')

# Create a FileHandler to write logs to a file
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG) # Changed to DEBUG for detailed output

# Create a StreamHandler to write logs to the console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG) # Changed to DEBUG for detailed output

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG) # Set overall root logger level to DEBUG

# Clear existing handlers to prevent duplicate output if logger is re-initialized
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Add both handlers to the root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

def get_logger(name):
    """
    Returns a named logger that inherits the root logger's configuration above
    use different names per module (e.g., __name__) to identify sources
    """
    logger = logging.getLogger(name)
    # The level for individual loggers will be inherited from the root_logger
    # unless specifically overridden here for a particular named logger.
    return logger

# Example usage within logger.py itself (for testing)
if __name__ == "__main__":
    test_logger = get_logger(__name__)
    test_logger.info("This message should appear in both console and log file.")
    test_logger.debug("This is a debug message and should also appear.")
    test_logger.error("An example error message.")
