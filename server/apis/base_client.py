from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class LLMOptions:
    temperature: float = 0.5
    max_tokens: int = 1024
    top_p: float = 1
    max_lines: int = 5
    prompt_index: int = 0


@dataclass
class State:
    prefix: str = ""
    suffix: str = ""
    code_to_edit: str = ""
    user_input: str = ""
    language: str = ""
    full_prefix: str = ""
    full_suffix: str = ""


@dataclass
class IBaseClient(ABC):
    """
    Interface for a model client that can stream or return responses based on a given prompt and model configuration.
    """

    models: List[str] = field(default_factory=list)

    @abstractmethod
    async def stream(
        self,
        state: State,
        model: str,
        options: LLMOptions,
    ):
        """
        Stream responses from a model based on the provided prompt and configuration parameters.

        :param prompt: The prompt to send to the model.
        :param model: The model identifier to use for generating responses.
        :param temperature: Controls randomness in the response generation.
        :param max_tokens: The maximum number of tokens that the model should generate.
        :param top_p: Controls the diversity of the response via nucleus sampling.
        """
        pass

    @abstractmethod
    async def create(
        self,
        state: State,
        model: str,
        options: LLMOptions,
    ):
        """
        Create a response from a model based on the provided prompt and configuration parameters.

        :param prompt: The prompt to send to the model.
        :param model: The model identifier to use for generating responses.
        :param temperature: Controls randomness in the response generation.
        :param max_tokens: The maximum number of tokens that the model should generate.
        :param top_p: Controls the diversity of the response via nucleus sampling.
        """
        pass

    @abstractmethod
    def generate_prompt_for_model(
        self, state: State, model: str, prompt_index: int
    ) -> str:
        """
        Generate a prompt for a given model based on the provided state.

        :param state: The state to generate the prompt for.
        :param model: The model identifier to use for generating the prompt.
        """
        pass

    @abstractmethod
    def generate_stop_tokens_for_model(
        self, model: str, prompt_index: int
    ) -> List[str]:
        """
        Generate stop tokens for a given model.

        :param model: The model identifier to use for generating stop tokens.
        """
        pass
