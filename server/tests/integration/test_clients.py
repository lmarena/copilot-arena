import pytest
from apis.base_client import State, LLMOptions
from apis.clients import (
    AnthropicClient,
    DeepseekFimClient,
    MistralClient,
    OpenAIClient,
    GeminiClient,
    FireworksClient,
)


@pytest.fixture
def state():
    # return State(prefix="def add(", suffix="def subtract(a, b):")
    return State(
        prefix='from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n              ',
        suffix=" - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
    )


@pytest.fixture
def options():
    return LLMOptions(
        temperature=0.3,
        max_tokens=1024,
        top_p=0.9,
        max_lines=5,
        prompt_index=0,
    )


@pytest.mark.asyncio
async def test_anthropic_client(state, options):
    client = AnthropicClient()
    for model in client.models:
        result = await client.create(state, model, options)
        assert isinstance(result, str)
        print(f"AnthropicClient response for {model}: {result}")


@pytest.mark.asyncio
async def test_deepseek_fim_client(state, options):
    client = DeepseekFimClient()
    for model in client.models:
        result = await client.create(state, model, options)
        assert isinstance(result, str)
        print(f"DeepseekFimClient response for {model}: {result}")


@pytest.mark.asyncio
async def test_mistral_client(state, options):
    client = MistralClient()
    for model in client.models:
        result = await client.create(state, model, options)
        assert isinstance(result, str)
        print(f"MistralClient response for {model}: {result}")


@pytest.mark.asyncio
async def test_openai_client(state, options):
    client = OpenAIClient()
    for model in client.models:
        result = await client.create(state, model, options)
        assert isinstance(result, str)
        print(f"OpenAIClient response for {model}: {result}")


@pytest.mark.asyncio
async def test_gemini_client(state, options):
    client = GeminiClient()
    for model in client.models:
        result = await client.create(state, model, options)
        assert isinstance(result, str)
        print(f"GeminiClient response for {model}: {result}")


@pytest.mark.asyncio
async def test_fireworks_ai_client(state, options):
    client = FireworksClient()
    for model in client.models:
        result = await client.create(state, model, options)
        assert isinstance(result, str)
        print(f"FireworksAIClient response for {model}: {result}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
