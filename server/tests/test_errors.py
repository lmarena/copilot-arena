import pytest
from src.errors import ModelTimeoutError


def test_model_timeout_error_basic():
    """Test ModelTimeoutError with just a model name"""
    model = "gpt-4"
    error = ModelTimeoutError(model)

    assert error.model == model
    assert error.original_error is None
    assert str(error) == f"Timeout occurred for model: {model}"


def test_model_timeout_error_with_original():
    """Test ModelTimeoutError with both model name and original error"""
    model = "gpt-4"
    original_error = ValueError("API timeout")
    error = ModelTimeoutError(model, original_error)

    assert error.model == model
    assert error.original_error == original_error
    assert (
        str(error)
        == f"Timeout occurred for model: {model} - Original error: API timeout"
    )


def test_model_timeout_error_attributes():
    """Test that ModelTimeoutError maintains expected attributes"""
    model = "gpt-4"
    original_error = ValueError("API timeout")
    error = ModelTimeoutError(model, original_error)

    # Test that we can access all attributes
    assert hasattr(error, "model")
    assert hasattr(error, "original_error")

    # Test that it's properly subclassed from Exception
    assert isinstance(error, Exception)
