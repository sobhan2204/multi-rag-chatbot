"""
Compatibility shim for modules that still import from 'exceptions'
This redirects all imports to custom_exceptions.py
"""

from custom_exceptions import *

# Define PendingDeprecationWarning
# It should be a built-in, but we define it here for compatibility
class PendingDeprecationWarning(Warning):
    """Warning about features which will be deprecated in the future."""
    pass

# Re-export everything
__all__ = ['CustomException', 'CustomError', 'DataLoadError', 
           'ModelInitializationError', 'QueryProcessingError',
           'PendingDeprecationWarning']