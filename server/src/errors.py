# Create a Custom timeout error
class ModelTimeoutError(Exception):
    def __init__(self, model: str, original_error: Exception = None):
        self.model = model
        self.original_error = original_error
        message = f"Timeout occurred for model: {model}"
        if original_error:
            message += f" - Original error: {str(original_error)}"
        super().__init__(message)
