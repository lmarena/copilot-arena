import yaml
from typing import List, Any, Dict, Union


class PromptGenerator:
    def __init__(self, yaml_file: str):
        with open(yaml_file, "r") as file:
            self.config = yaml.safe_load(file)

        if "messages" not in self.config and "prompt" not in self.config:
            raise ValueError("Neither 'messages' nor 'prompt' found in config file.")
        if "stop_tokens" not in self.config:
            raise ValueError("Stop tokens not found in config file.")
        if "args" not in self.config:
            raise ValueError("Args not found in config file.")

        self.messages = self.config.get("messages")
        self.prompt = self.config.get("prompt")
        if self.prompt:
            self.prompt = self.prompt.rstrip("\n")
        self.stop_tokens = self.config.get("stop_tokens", [])
        self.args = self.config.get("args", [])
        self.options = self.config.get("options", [])
        self.start_phrases = self.config.get("start_phrases", [])

    def generate_prompt(self, **kwargs: Any) -> Union[str, List[Dict[str, str]]]:
        try:
            if self.messages:
                return self._generate_messages_prompt(**kwargs)
            elif self.prompt:
                return self._generate_single_prompt(self.prompt, **kwargs)
            else:
                raise ValueError("No valid prompt format found in config.")
        except Exception as e:
            raise ValueError(f"Error generating prompt: {e}")

    def _generate_messages_prompt(self, **kwargs: Any) -> List[Dict[str, str]]:
        formatted_messages = []
        for message in self.messages:
            role = message["role"]
            content = self._generate_single_prompt(
                message["content"].rstrip("\n"), **kwargs
            )

            formatted_messages.append({"role": role, "content": content})

        return formatted_messages

    def _generate_single_prompt(self, prompt, **kwargs: Any) -> str:
        format_dict = {}
        for arg in self.args:
            if arg in kwargs:
                format_dict[arg] = kwargs[arg]
        return prompt.format(**format_dict)

    def get_stop_tokens(self) -> List[str]:
        return self.stop_tokens

    def get_start_phrases(self) -> List[str]:
        return self.start_phrases

    def get_args(self) -> List[str]:
        return self.args
