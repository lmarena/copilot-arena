import os
from openai import AsyncOpenAI
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

try:
    from config.api_config import FIREWORKS_API_KEY
except ImportError:
    FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")


class FireworksClient(IBaseClient):
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=FIREWORKS_API_KEY,  # Use Fireworks API Key
            base_url="https://api.fireworks.ai/inference/v1",  # Set Fireworks API Base
            max_retries=MAX_RETRIES,
            timeout=TIMEOUT,
        )
        self.models = [
            "llama-3.1-70b-instruct",
            "llama-3.1-405b-instruct",
            "qwen-2.5-coder-32b-instruct",
        ]
        chat_prompt_generators = [
            PromptGenerator("templates/chat_psm_overlap.yaml"),
            PromptGenerator("templates/edit/chat_edit.yaml"),
        ]

        self._prompt_generators = {
            "llama-3.1-70b-instruct": chat_prompt_generators,
            "llama-3.1-405b-instruct": chat_prompt_generators,
            "qwen-2.5-coder-32b-instruct": chat_prompt_generators,
        }

        self.model_name_map = {
            "llama-3.1-70b-instruct": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "llama-3.1-405b-instruct": "accounts/fireworks/models/llama-v3p1-405b-instruct",
            "qwen-2.5-coder-32b-instruct": "accounts/fireworks/models/qwen2p5-coder-32b-instruct",
        }

    async def stream(
        self,
        state: State,
        model: str,
        options: LLMOptions,
    ):
        raise NotImplementedError("Fireworks AI streaming not implemented.")

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

        response = await self.client.chat.completions.create(
            messages=prompt,
            model=self.model_name_map[model],  # Map to the official name
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            top_p=options.top_p,
            stop=stop_tokens[:4],
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
    client = FireworksClient()
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
