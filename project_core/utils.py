"""General project utilities"""
from pathlib import Path
from typing import Dict

from yaml import dump, safe_load


def load_yaml_config(config_path: Path) -> Dict:
    """Load YAML config file and return a parsed, ready to consume, dictionary

    :param config_path: location of the config file
    :return: config dictionary
    """
    with open(file=config_path, mode='r') as stream:
        config = safe_load(stream)
    return config


def write_yaml_config(config: Dict, path: Path):
    """Write config object (dictionary) to a YAML config file

    :config: config object
    :param config_path: location to write the config file
    """
    with path.open(mode='w') as stream:
        dump(config, stream)
