from enum import StrEnum
import logging


class PrivacySetting(StrEnum):
    PRIVATE = "Private"
    DEBUG = "Debug"
    RESEARCH = "Research"


class PrivacySensitiveKeys(StrEnum):
    PROMPT = "prompt"
    PREFIX = "prefix"
    FULL_PREFIX = "full_prefix"
    SUFFIX = "suffix"
    FULL_SUFFIX = "full_suffix"
    RAW_RESPONSE = "raw_response"
    RAW_COMPLETION = "raw_completion"
    RESPONSE = "response"
    COMPLETION = "completion"
    CODE_TO_EDIT = "code_to_edit"
    USER_INPUT = "user_input"


def clean_data(data: dict):
    cleaned_data = {}
    for key, value in data.items():
        if key not in PrivacySensitiveKeys:
            if isinstance(value, dict):
                cleaned_data[key] = clean_data(value)
            elif isinstance(value, list):
                cleaned_data[key] = [
                    (clean_data(item) if isinstance(item, dict) else item)
                    for item in value
                ]
            else:
                cleaned_data[key] = value

    return cleaned_data


def privacy_aware_log(
    data: dict,
    privacy_setting: PrivacySetting,
    logger: logging.Logger,
    level: int = logging.INFO,
):
    if privacy_setting == PrivacySetting.PRIVATE:
        data = clean_data(data)
    logger.log(level, data)
