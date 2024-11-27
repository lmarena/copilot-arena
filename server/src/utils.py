import yaml
import base64
import os
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv


def get_settings():
    config_path = "config/app_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    # Fall back to environment variable

    load_dotenv(find_dotenv())
    encoded_config = os.getenv("APP_CONFIG_YAML")
    print("encoded config")
    print(encoded_config)

    # Decode base64 encoded YAML
    try:
        yaml_content = base64.b64decode(encoded_config).decode("utf-8")
        return yaml.safe_load(yaml_content)
    except Exception as e:
        raise ValueError(f"Failed to decode configuration: {str(e)}")


def get_models_by_tags(tags: List, models: List, tag_to_models: Dict):
    if not tags or len(tags) == 0:
        return models
    tagged_models = set()
    for tag in tags:
        tagged_models.update(tag_to_models.get(tag, set()))
    return list(tagged_models)


def get_cost(prompt_token_length: int, response_token_length: int, model: str):
    config = get_settings()
    model_config = config["models"][model]
    return (
        prompt_token_length * model_config["input_cost"]
        + response_token_length * model_config["output_cost"]
    ) / 1000000
