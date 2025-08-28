import sys
import logging # Assuming you'd want to use logging within exceptions too

# Initialize logger for this module if needed, otherwise rely on root logger
# from src.logger import get_logger
# logging = get_logger(__name__)

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        # Pass the original error_message along with the detailed one
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    def get_detailed_error_message(self, error_message, error_detail:sys):
        _, _, exc_tb = error_detail.exc_info() # Captures exception type, value, and traceback object

        if exc_tb is None: # Added check for NoneType traceback
            return f"Error: {error_message}"

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        
        # Format the error message to include file, line, and original message
        detailed_error = f"Error in {file_name} , line {line_number} : {error_message}"
        return detailed_error

    def __str__(self):
        return self.error_message
