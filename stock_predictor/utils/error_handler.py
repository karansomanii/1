import functools
import traceback
from .logger import Logger

logger = Logger()

def handle_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    return wrapper 