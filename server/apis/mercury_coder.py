import os
from openai import AsyncOpenAI, APITimeoutError
from apis.base_client import IBaseClient, LLMOptions, State, LLMResponse
from apis.utils import (
    generate_prompt_for_model,
    generate_stop_tokens_for_model,
    generate_start_phrases_for_model,
    parse_response,
    get_prompt_options_for_model,
)
from prompts.prompt_generator import PromptGenerator
from constants import MAX_RETRIES, TIMEOUT
from src.errors import ModelTimeoutError
from typing import List, Dict, Callable, Any
import random

try:
    from config.api_config import MERCURY_CODER_API_KEY
except ImportError:
    MERCURY_CODER_API_KEY = os.getenv("MERCURY_CODER_API_KEY")


class MercuryCoderClient(IBaseClient):
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=MERCURY_CODER_API_KEY,
            base_url="https://alpha.inceptionlabs.ai/v1",
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        self.models = ["mercury-coder-mini"]
        self._prompt_generators = {
            "mercury-coder-mini": [
                PromptGenerator("templates/mercury_coder.yaml"),
            ],
        }
        self.model_name_map = {
            "mercury-coder-mini": "anonymous-titan",
        }

    async def stream(
        self,
        state: State,
        model: str,
        options: LLMOptions,
    ) -> LLMResponse:
        raise NotImplementedError("Mercury Coder streaming not implemented.")

    async def create(
        self,
        state: State,
        model: str,
        options: LLMOptions,
    ):
        if model not in self.models:
            raise ValueError(f"Model {model} is not supported.")

        prompt = self.generate_prompt_for_model(state, model, options.prompt_index)
        stop_tokens = self.generate_stop_tokens_for_model(model, options.prompt_index)
        start_phrases = self.generate_start_phrases_for_model(
            model, options.prompt_index
        )

        try:
            response = await self.client.completions.create(
                model=self.model_name_map[model],
                prompt=prompt,
                suffix=state.suffix,
                max_tokens=options.max_tokens,
                stream=False,
                stop=stop_tokens,
            )

            # Move this into a util function called parse_response
            response = response.choices[0].text
            completion = parse_response(
                response,
                options.max_lines,
                stop_tokens,
                start_phrases,
                state,
                get_prompt_options_for_model(
                    self._prompt_generators, options.prompt_index, model
                ),
            )
            return LLMResponse(raw_text=response, text=completion)
        except APITimeoutError as e:
            raise ModelTimeoutError(model=model, original_error=e)

    def generate_prompt_for_model(self, state: State, model: str, prompt_index: int):
        return generate_prompt_for_model(
            self._prompt_generators,
            prompt_index,
            state,
            model,
        )

    def generate_stop_tokens_for_model(self, model: str, prompt_index):
        return generate_stop_tokens_for_model(
            self._prompt_generators, prompt_index, model
        )

    def generate_start_phrases_for_model(self, model: str, prompt_index):
        return generate_start_phrases_for_model(
            self._prompt_generators, prompt_index, model
        )
