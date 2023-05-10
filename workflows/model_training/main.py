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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from project_core.model.definitions.models import iforest
from project_core.model.utils import load_datasets, train_classification_model
from project_core import __version__ as project_core_version
from project_core.model.utils import save_model
from project_core.model.utils import load_model
from project_core.utils import load_yaml_config
from project_core.data.processing import processing_nbm_rul

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
    parser.add_argument('--input_hold_data_path', type=Path, help='Held data path')
    parser.add_argument('--output_model_path', type=Path, help='output path (model, config, metadata)')
    parser.add_argument('--config_path', type=Path, help='Path to model training config file')
    parser.add_argument('--model_option', type=Text,
                        choices=['one_class_classification', 'other_models'],
                        help='Model to be trained')
    return parser.parse_args(arguments_list)


def run(arguments_list: List = None):
    """Run step"""
    arguments = parse_arguments(arguments_list=arguments_list)
    config = load_yaml_config(config_path=Path(__file__).parent / arguments.config_path)
    
    logger.info('Start training model')
    if arguments.model_option == 'one_class_classification':
        try:
            model = train_classification_model(arguments=arguments, config=config)
        except FileNotFoundError:
            logger.info('Please enter the paths for datasets manually.')
            arguments.input_train_data_path = Path(input('Enter the path for train dataset: '))
            arguments.input_test_data_path = Path(input('Enter the path for test dataset: '))
            model = train_classification_model(arguments=arguments, config=config)
    elif arguments.model_option == 'other_models':
        #TODO
        pass


if __name__ == '__main__':
    run()