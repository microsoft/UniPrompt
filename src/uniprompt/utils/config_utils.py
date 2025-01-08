from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration file: {e}")
