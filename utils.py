"""Utility functions for the LightsSyncHdmi application."""
import functools
import time


def retry_on_error(max_attempts=3, delay=0):
    """
    Decorator to retry a function on exception.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds
    
    Returns:
        The decorated function that will retry on error
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Error in {func.__name__} (attempt {attempt + 1}/{max_attempts}): {e}")
                    if attempt < max_attempts - 1 and delay > 0:
                        time.sleep(delay)
            
            # If all retries failed, raise the last exception or print error
            print(f"Failed to execute {func.__name__} after {max_attempts} attempts.")
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator
