import sys
import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Returns a detailed error message including:
      - Python script filename
      - Line number
      - Original error message
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in script: [{file_name}], "
        f"line number: [{line_number}], "
        f"error message: [{str(error)}]"
    )

"""
Custom exceptions for the RAG pipeline
"""

class CustomException(Exception):
    """Base exception for custom errors"""
    pass

class CustomError(Exception):
    """Raised when a specific error condition occurs"""
    pass

class DataLoadError(Exception):
    """Raised when data loading fails"""
    pass

class ModelInitializationError(Exception):
    """Raised when model initialization fails"""
    pass

class QueryProcessingError(Exception):
    """Raised when query processing fails"""
    pass


class CustomException(Exception):
    """
    Custom exception class for detailed and uniform error reporting.
    Logs the error automatically when raised.
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        detailed_message = error_message_detail(error_message, error_detail)
        super().__init__(detailed_message)
        logging.error(detailed_message)


# Optional: configure a default logger (you can adjust or remove this)
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("error.log"),
        logging.StreamHandler()
    ]
)
