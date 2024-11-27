import os
import asyncio
from mistralai import Mistral
from apis.base_client import IBaseClient, LLMOptions, State
from apis.utils import (
    generate_prompt_for_model,
    generate_stop_tokens_for_model,
    generate_start_phrases_for_model,
    parse_response,
    get_prompt_options_for_model,
)
from prompts.prompt_generator import PromptGenerator
from constants import TIMEOUT

try:
    from config.api_config import MISTRAL_API_KEY
except ImportError:
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")


class MistralClient(IBaseClient):
    def __init__(self) -> None:
        self.client = Mistral(api_key=MISTRAL_API_KEY, timeout_ms=TIMEOUT * 1000)
        self.models = ["codestral-2405"]
        self._prompt_generators = {
            "codestral-2405": [
                PromptGenerator("templates/codestral.yaml"),
            ]
        }

    async def stream(
        self,
        state: State,
        model: str,
        options: LLMOptions,
    ):
        raise NotImplementedError("Mistral streaming not implemented.")

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

        response = await self.client.fim.complete_async(
            model=model,
            prompt=prompt,
            suffix=state.suffix,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            top_p=options.top_p,
            stop=stop_tokens,
        )

        response = response.choices[0].message.content
        completion = parse_response(
            response,
            options.max_lines,
            stop_tokens,
            generate_start_phrases_for_model(
                self._prompt_generators, options.prompt_index, model
            ),
            state,
            get_prompt_options_for_model(
                self._prompt_generators, options.prompt_index, model
            ),
        )

        # Handle white spaces for codestral (from Continue)
        if len(completion) >= 2 and completion[0] == " " and completion[1] != " ":
            if state.prefix.endswith(" ") and state.suffix.startswith("\n"):
                completion = completion[1:]

        return completion

    def generate_prompt_for_model(self, state: State, model: str, prompt_index: int):
        return generate_prompt_for_model(
            self._prompt_generators, prompt_index, state, model
        )

    def generate_stop_tokens_for_model(self, model: str, prompt_index: int):
        return generate_stop_tokens_for_model(
            self._prompt_generators, prompt_index, model
        )


async def main():
    client = MistralClient()
    state = State(
        prefix="def is_odd(n): \n return n % 2 == 1 \ndef test_is_odd():", suffix=""
    )
    prompt_index = 1
    options = LLMOptions(
        temperature=0.5,
        max_tokens=1024,
        top_p=1.0,
        max_lines=5,
        prompt_index=prompt_index,
    )
    print(await client.create(state, model="codestral-2405", options=options))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
