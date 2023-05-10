"""
The purpose of train_model is take in the processed data and do the following (if needed):
1. Handle model selection
2. Handle hyper parameter selection
3. Handle metric logging to experiment tracking service
4. Make sure the model has .predict, if not wrap it in a class that exposes a predict method.
5. Return trained model that has a predict method
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, List, Text

import pandas as pd
import structlog

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from project_core.model.definitions.models import iforest
from project_core import __version__ as project_core_version
from project_core.model.utils import save_model
from project_core.utils import load_yaml_config

logger = structlog.getLogger(__name__)


def parse_arguments(arguments_list: List) -> Namespace:
    """Parse arguments list"""
    parser = ArgumentParser(__name__)

    parser.add_argument('--input_train_data_path', type=Path, help='Training data path')
    parser.add_argument('--input_X_train_data_file_name', type=Text, help='X Training data file name')
    parser.add_argument('--input_y_train_data_file_name', type=Text, help='y Training data file name')
    parser.add_argument('--input_test_data_path', type=Path, help='Testing data path')
    parser.add_argument('--input_X_test_data_file_name', type=Text, help='X Testing data file name')
    parser.add_argument('--input_y_test_data_file_name', type=Text, help='y Testing data file name')
    parser.add_argument('--output_model_path', type=Path, help='output path (model, config, metadata)')
    parser.add_argument('--config_path', type=Path, help='Path to model training config file')
    parser.add_argument('--model_option', type=Text,
                        choices=['isolation_forest', 'other_models'],
                        help='Model to be trained')
    return parser.parse_args(arguments_list)


def load_datasets(X_train_path: Path, y_train_path: Path):
    logger.info('Load X_train')
    X_train = pd.read_feather(path=X_train_path)
    X_train.set_index('local_ts_start', inplace=True)

    logger.info('Load y_train')
    y_train = pd.read_feather(path=y_train_path)
    y_train.set_index('local_ts_start', inplace=True)
    return X_train, y_train


def run(arguments_list: List = None):
    """Run step"""
    arguments = parse_arguments(arguments_list=arguments_list)
    config = load_yaml_config(config_path=Path(__file__).parent / arguments.config_path)

    X_train, y_train = load_datasets(X_train_path=f'{arguments.input_train_data_path}/{arguments.input_X_train_data_file_name}',
                                     y_train_path=f'{arguments.input_train_data_path}/{arguments.input_y_train_data_file_name}')

    if config.get('train_on_full_dataset', False) and arguments.input_test_data_path:
        # To train the model on the full dataset for deployment purposes only
        logger.info('Train on full dataset')
        X_test = pd.read_feather(path=arguments.input_test_data_path / arguments.input_X_test_data_file_name)
        y_test = pd.read_feather(path=arguments.input_test_data_path / arguments.input_y_test_data_file_name)
        X_train = pd.concat(objs=[X_train, X_test])
        y_train = pd.concat(objs=[y_train, y_test])

    logger.info('Start training model')
    if arguments.model_option == 'isolation_forest':
        model = iforest(X_train=X_train, y_train=y_train)
    else:
        # TODO
        # code for other models here
        pass
    logger.info('Finish training model')

    save_model(
        path=arguments.output_model_path,
        model=model,
        config=config,
        metadata={
            'model.n_features_in_': model.n_features_in_,
            'project_core_version': project_core_version,
        }
    )
    logger.info('Finish saving model')


if __name__ == '__main__':
    run()
