import os
from openai import AsyncOpenAI, APITimeoutError
from apis.base_client import IBaseClient, LLMOptions, State
from prompts.prompt_generator import PromptGenerator
from apis.utils import (
    generate_prompt_for_model,
    generate_stop_tokens_for_model,
    parse_response,
    generate_start_phrases_for_model,
    get_prompt_options_for_model,
)
from constants import MAX_RETRIES, TIMEOUT
from src.errors import ModelTimeoutError

try:
    from config.api_config import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class OpenAIClient(IBaseClient):
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            max_retries=MAX_RETRIES,
            timeout=TIMEOUT,
        )
        self.models = [
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20",
        ]
        chat_prompt_generators = [
            PromptGenerator("templates/chat_psm_overlap.yaml"),
            PromptGenerator("templates/edit/chat_edit.yaml"),
        ]

        self._prompt_generators = {
            "gpt-4o-mini-2024-07-18": chat_prompt_generators,
            "gpt-4o-2024-08-06": chat_prompt_generators,
            "gpt-4o-2024-11-20": chat_prompt_generators,
        }

    async def stream(
        self,
        state: State,
        model: str,
        options: LLMOptions,
    ):
        raise NotImplementedError("OpenAI streaming not implemented.")

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

        try:
            response = await self.client.chat.completions.create(
                messages=prompt,
                model=model,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
                top_p=options.top_p,
                stop=stop_tokens[:4],  # openai only allows 4 stop tokens
                stream=False,
            )

            response = response.choices[0].message.content
            completion = parse_response(
                response,
                options.max_lines,
                stop_tokens,
                generate_start_phrases_for_model(
                    self._prompt_generators,
                    options.prompt_index,
                    model,
                ),
                state,
                get_prompt_options_for_model(
                    self._prompt_generators, options.prompt_index, model
                ),
            )

            return completion
        except APITimeoutError as e:
            raise ModelTimeoutError(model=model, original_error=e)

    def generate_prompt_for_model(self, state: State, model: str, prompt_index: int):
        return generate_prompt_for_model(
            self._prompt_generators,
            prompt_index,
            state,
            model,
        )

    def generate_stop_tokens_for_model(self, model: str, prompt_index: int):
        return generate_stop_tokens_for_model(
            self._prompt_generators,
            prompt_index,
            model,
        )


async def main():
    client = OpenAIClient()
    state = State(prefix="def add(", suffix="\n\ndef subtract(a, b):")
    prompt_index = 0
    options = LLMOptions(
        temperature=0.5,
        max_tokens=1024,
        top_p=1.0,
        max_lines=5,
        prompt_index=prompt_index,
    )

    for model in client.models:
        print("Prompt:")
        print(client.generate_prompt_for_model(state, model, prompt_index=prompt_index))
        print("Response:")
        response = await client.create(state, model, options)
        print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
