"""General model management utilities"""
import pickle
from pathlib import Path
from typing import Any, Dict

from project_core.utils import write_yaml_config


def load_model(model_path: Path) -> Any:
    """Load model from path

    :param model_path: path to model file
    :return: loaded model
    """
    model = pickle.load(open(model_path, 'rb'))
    return model


def save_model(path: Path, model: Any, config: Dict, metadata: Dict):
    """Save model object, config, and metadata to output directory

    :param path: Path to output directory
    :param model: model object
    :param config: config object used for training this model
    :param metadata: extra metadata, such as performance, features, package version, .. etc
    """
    path.mkdir(parents=True, exist_ok=True)
    # FIXME
    with (path / 'model.pkl').open(mode='wb') as f:
        pickle.dump(obj=model, file=f)

    write_yaml_config(config=config, path=path / 'config.yaml')
    write_yaml_config(config=metadata, path=path / 'metadata.yaml')
