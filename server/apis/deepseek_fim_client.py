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

try:
    from config.api_config import DEEPSEEK_API_KEY
except ImportError:
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")


class DeepseekFimClient(IBaseClient):
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/beta",
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        self.models = ["deepseek-coder-v3-fim"]
        self._prompt_generators = {
            "deepseek-coder-v3-fim": [
                PromptGenerator("templates/deepseek_fim.yaml"),
            ],
        }

    async def stream(
        self,
        state: State,
        model: str,
        options: LLMOptions,
    ) -> LLMResponse:
        raise NotImplementedError("Deepseek streaming not implemented.")

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
                model="deepseek-coder",  # Hardcoded model as it shares the same name with deepseek-coder (instruct)
                prompt=prompt,
                suffix=state.suffix,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
                top_p=options.top_p,
                stop=stop_tokens,
                stream=False,
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
