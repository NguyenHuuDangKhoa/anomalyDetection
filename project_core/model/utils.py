"""General model management utilities"""
import pickle
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from project_core.utils import write_yaml_config
from project_core.model.definitions.models import iforest
import structlog

logger = structlog.getLogger(__name__)

def load_model(model_path: Path, model_name: str = 'model.pkl') -> Any:
    """Load model from path

    :param model_path: path to model file
    :return: loaded model
    """
    model = pickle.load(open(model_path / model_name, 'rb'))
    return model

def save_model(path: Path, model: Any, config: Dict, metadata: Dict, model_name: str ='model.pkl') -> None:
    """Save model object, config, and metadata to output directory

    :param path: Path to output directory
    :param model: model object
    :param config: config object used for training this model
    :param metadata: extra metadata, such as performance, features, package version, .. etc
    """
    path.mkdir(parents=True, exist_ok=True)
    with (path / model_name).open(mode='wb') as f:
        pickle.dump(obj=model, file=f)

    write_yaml_config(config=config, path=path / 'config.yaml')
    write_yaml_config(config=metadata, path=path / 'metadata.yaml')

def train_classification_model(arguments: List, config: List) -> Any:
    """
    This function is responsible for concatenate all datasets if
    the option to train model on full dataset for deployment purposes is selected.
    Then it also trains and return the model
    param arguments: a list of directories, names of all datasets
    param config: a list of parameters' values for training model
    return: trained model
    """
    if config.get('train_on_full_dataset', False) and arguments.input_test_data_path:
    # To train the model on the full dataset for deployment purposes only
        logger.info('Train on full dataset')
        X_test = pd.read_feather(path=arguments.input_test_data_path / arguments.input_X_test_data_file_name)
        y_test = pd.read_feather(path=arguments.input_test_data_path / arguments.input_y_test_data_file_name)
        X_train = pd.concat(objs=[X_train, X_test])
        y_train = pd.concat(objs=[y_train, y_test])
    X_train, y_train = load_datasets(X_train_path=f'{arguments.input_train_data_path}/{arguments.input_X_train_data_file_name}',
                                        y_train_path=f'{arguments.input_train_data_path}/{arguments.input_y_train_data_file_name}')
    model = iforest(X_train=X_train, y_train=y_train)
    return model

def load_datasets(X_train_path: Path, y_train_path: Path) -> pd.DataFrame:
    """
    This function loads input and target training datasets
    param X_train_path: Path to X_train dataset
    param y_train_path: Path to y_train dataset
    :return: pandas DataFrames
    """
    logger.info('Load X_train')
    X_train = pd.read_feather(path=X_train_path)
    X_train.set_index('local_ts_start', inplace=True)

    logger.info('Load y_train')
    y_train = pd.read_feather(path=y_train_path)
    y_train.set_index('local_ts_start', inplace=True)
    return X_train, y_train
