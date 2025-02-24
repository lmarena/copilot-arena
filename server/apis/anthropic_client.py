import os
from anthropic import AsyncAnthropic
from apis.base_client import IBaseClient, LLMOptions, State, LLMResponse
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
    from config.api_config import ANTHROPIC_API_KEY
except ImportError:
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


class AnthropicClient(IBaseClient):
    def __init__(self) -> None:
        self.client = AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY, max_retries=MAX_RETRIES, timeout=TIMEOUT
        )
        self.models = [
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-7-sonnet-20250219",
        ]
        chat_prompt_generators = [
            PromptGenerator("templates/chat_psm_overlap.yaml"),
            PromptGenerator("templates/edit/chat_edit.yaml"),
        ]
        # sonnet_prompt_generators = chat_prompt_generators
        # sonnet_prompt_generators[0] = PromptGenerator("templates/chat_spm_overlap.yaml")
        self._prompt_generators = {
            "claude-3-haiku-20240307": chat_prompt_generators,
            "claude-3-5-sonnet-20240620": chat_prompt_generators,
            "claude-3-5-sonnet-20241022": chat_prompt_generators,
            "claude-3-7-sonnet-20250219": chat_prompt_generators,
        }

    async def stream(
        self,
        state: State,
        model: str,
        options: LLMOptions,
    ):
        raise NotImplementedError("Anthropic streaming not implemented.")

    async def create(
        self,
        state: State,
        model: str,
        options: LLMOptions,
    ) -> LLMResponse:
        if model not in self.models:
            raise ValueError(f"Model {model} is not supported.")

        prompt = self.generate_prompt_for_model(state, model, options.prompt_index)
        stop_tokens = self.generate_stop_tokens_for_model(model, options.prompt_index)

        system_message, filtered_prompt = self.process_messages(prompt)

        response = await self.client.messages.create(
            model=model,
            messages=filtered_prompt,
            system=system_message,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            top_p=options.top_p,
            stop_sequences=self.filter_whitespaces(stop_tokens),
            stream=False,
        )

        if len(response.content) == 0:
            return ""

        response = response.content[0].text
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

        return LLMResponse(raw_text=response, text=completion)

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

    def filter_whitespaces(self, stop_sequences):
        """Filter out stop sequences that consist only of whitespace."""
        return [seq for seq in stop_sequences if seq.strip()]

    def process_messages(self, messages):
        system_message = []
        filtered_messages = []

        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            else:
                filtered_messages.append(message)

        return system_message, filtered_messages


async def main():
    client = AnthropicClient()
    prompt_index = 1
    state = State(prefix="def add(", suffix="def subtract(a, b):")
    options = LLMOptions(
        temperature=0.5, max_tokens=1024, top_p=1.0, max_lines=5, prompt_index=1
    )

    for model in client.models:
        print("Prompt:")
        print(client.generate_prompt_for_model(state, model, prompt_index))
        print("Response:")
        print(await client.create(state, model, options))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
