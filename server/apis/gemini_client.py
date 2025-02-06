import os
import google.generativeai as genai
from apis.base_client import IBaseClient, LLMOptions, State, LLMResponse
from prompts.prompt_generator import PromptGenerator
from apis.utils import (
    generate_prompt_for_model,
    generate_stop_tokens_for_model,
    parse_response,
    generate_start_phrases_for_model,
    get_prompt_options_for_model,
)

try:
    from config.api_config import GOOGLE_API_KEY
except ImportError:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


class GeminiClient(IBaseClient):
    def __init__(self) -> None:
        genai.configure(api_key=GOOGLE_API_KEY)
        self.models = [
            # "gemini-1.5-flash-001",
            "gemini-1.5-flash-002",
            # "gemini-1.5-pro-001",
            "gemini-1.5-pro-002",
            # "gemini-1.5-flash-exp-0827",
            # "gemini-1.5-pro-exp-0827",
            # "gemini-2.0-flash-exp",
            "gemini-2.0-flash-001",
            "gemini-2.0-pro-exp-02-05",
        ]
        chat_prompt_generators = [
            PromptGenerator("templates/chat_psm_overlap.yaml"),
            PromptGenerator("templates/edit/chat_edit.yaml"),
        ]
        self._prompt_generators = {
            "gemini-1.5-flash-001": chat_prompt_generators,
            "gemini-1.5-flash-002": chat_prompt_generators,
            "gemini-1.5-pro-001": chat_prompt_generators,
            "gemini-1.5-pro-002": chat_prompt_generators,
            "gemini-1.5-flash-exp-0827": chat_prompt_generators,
            "gemini-1.5-pro-exp-0827": chat_prompt_generators,
            "gemini-2.0-flash-exp": chat_prompt_generators,
            "gemini-2.0-flash-001": chat_prompt_generators,
            "gemini-2.0-pro-exp-02-05": chat_prompt_generators,
        }

    async def stream(
        self,
        state: State,
        model: str,
        options: LLMOptions,
    ):
        raise NotImplementedError("Gemini streaming not implemented.")

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

        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_message[0] if len(system_message) > 0 else None,
        )

        response = await gemini_model.generate_content_async(
            [
                {"role": msg["role"], "parts": [msg["content"]]}
                for msg in filtered_prompt
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=options.temperature,
                max_output_tokens=options.max_tokens,
                top_p=options.top_p,
                stop_sequences=self.filter_whitespaces(stop_tokens)[
                    :5
                ],  # Gemini has max 5 stop sequences
            ),
        )

        if not response.text:
            return ""

        completion = parse_response(
            response.text,
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

        return LLMResponse(raw_text=response.text, text=completion)

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
                system_message.append(message["content"])
            else:
                # Gemini calls assistants model
                if message["role"] == "assistant":
                    message["role"] = "model"
                filtered_messages.append(message)

        return system_message, filtered_messages


async def main():
    client = GeminiClient()
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
