from typing import Dict
from prompts.prompt_generator import PromptGenerator
from apis.base_client import State


def generate_prompt_for_model(
    prompt_generators: Dict[str, PromptGenerator],
    prompt_index: int,
    state: State,
    model: str,
) -> str:
    prompt_generator = prompt_generators[model][prompt_index]
    return prompt_generator.generate_prompt(
        prefix=state.prefix,
        suffix=state.suffix,
        language=state.language,
        code_to_edit=state.code_to_edit,
        user_input=state.user_input,
    )


def generate_stop_tokens_for_model(
    prompt_generators: Dict[str, PromptGenerator],
    prompt_index: int,
    model: str,
) -> list:
    prompt_generator: PromptGenerator = prompt_generators[model][prompt_index]
    return prompt_generator.get_stop_tokens()


def generate_start_phrases_for_model(
    prompt_generators: Dict[str, PromptGenerator],
    prompt_index: int,
    model: str,
) -> list:
    prompt_generator: PromptGenerator = prompt_generators[model][prompt_index]
    return prompt_generator.get_start_phrases()


def get_prompt_options_for_model(
    prompt_generators: Dict[str, PromptGenerator],
    prompt_index: int,
    model: str,
):
    prompt_generator: PromptGenerator = prompt_generators[model][prompt_index]
    return prompt_generator.options


def longest_suffix_prefix(a, b):
    max_length = min(len(a), len(b))
    for length in range(max_length, 0, -1):
        suffix_a = a[-length:]
        prefix_b = b[:length]
        if suffix_a == prefix_b:
            return (len(a) - length, 0, length)
    return (len(a), 0, 0)


def longest_suffix_suffix(a, b):
    max_length = min(len(a), len(b))
    for length in range(max_length, 0, -1):
        suffix_a = a[-length:]
        suffix_b = b[-length:]
        if suffix_a == suffix_b:
            return (len(a) - length, len(b) - length, length)
    return (len(a), len(b), 0)


def remove_overlap(prefix: str, suffix: str, mid: str):
    # Normalize line endings in all inputs
    prefix = prefix.replace("\r\n", "\n").replace("\r", "\n")
    suffix = suffix.replace("\r\n", "\n").replace("\r", "\n")
    mid = mid.replace("\r\n", "\n").replace("\r", "\n")

    # Store original suffix length for later adjustment
    original_suffix_len = len(suffix)

    prefix_match = longest_suffix_prefix(prefix, mid)
    if prefix_match[2] != 0:
        mid = mid[prefix_match[2] :]

    suffix_match = longest_suffix_prefix(mid, suffix)
    if suffix_match[2] != 0:
        mid = mid[: -suffix_match[2]]

    suffix_match = longest_suffix_suffix(mid, suffix)
    if suffix_match[2] != 0:
        mid = mid[: -suffix_match[2]]

    # Adjust the mid string based on the difference in suffix length
    suffix_len_diff = original_suffix_len - len(suffix)
    if suffix_len_diff > 0:
        mid = mid[:-suffix_len_diff]
    elif suffix_len_diff < 0:
        mid += "\n" * (-suffix_len_diff)

    return mid


def parse_response(
    completion,
    max_lines: int,
    stop_tokens: list,
    start_phrases: list,
    state: State,
    prompt_options: list = [],
):
    completion = "\n".join(completion.split("\n")[: max_lines + 1])

    # Check for start phrases and code snippet markers
    start_index = -1
    for start_phrase in start_phrases:
        start_index = completion.find(start_phrase)
        if start_index != -1:
            completion = completion[start_index + len(start_phrase) :]
            break

    # If no start phrase found, check for code snippet markers
    start_index = -1
    code_snippet_markers = [
        f"```{lang}\n"
        for lang in [
            "python",
            "typescript",
            "javascript",
            "markdown",
            "html",
            "css",
            "java",
            "c",
            "cpp",
            "csharp",
            "go",
            "rust",
            "swift",
            "kotlin",
            "ruby",
            "php",
            "haskell",
        ]
    ]
    for marker in code_snippet_markers:
        start_index = completion.find(marker)
        if start_index != -1:
            completion = completion[start_index + len(marker) :]
            break

    # Remove ending code snippet marker
    if start_index != -1:  # Found a start code snippet
        end_marker = "\n```"
        end_index = completion.rfind(end_marker)
        if end_index != -1:
            completion = completion[:end_index]

    # Rest of the function remains the same
    for stop_token in stop_tokens:
        stop_index = completion.find(stop_token)
        if stop_index != -1:
            completion = completion[:stop_index]
            break

    prefix = state.prefix
    suffix = state.suffix

    if "overlap" in prompt_options:
        # strip ending newline from suffix and completion (but not any starting newlines)
        completion = completion.rstrip("\n")
        suffix = suffix.rstrip("\n")
        completion = remove_overlap(prefix, suffix, completion)

    return completion
